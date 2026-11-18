from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from namellm.constants import LABELS  # noqa: E402
except ModuleNotFoundError:
    LABELS = sorted(
        [
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "claude-fable-5",
            "claude-opus-4-6",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "qwen3.8-max",
            "glm-5.3",
            "glm-5.3-flash",
            "kimi-k3",
        ]
    )


class TextRows(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.ids = frame["id"].astype(str).tolist()
        self.texts = frame["text"].fillna("").astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.ids[index], self.texts[index]


def make_collator(tokenizer, max_length: int):
    def collate(items: list[tuple[str, str]]) -> dict[str, object]:
        ids, texts = zip(*items)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {"ids": ids, **encoded}

    return collate


class MultiDepthHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        selected_layers: list[int],
        projection_size: int = 256,
    ):
        super().__init__()
        self.selected_layers = selected_layers
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, projection_size),
                )
                for _ in selected_layers
            ]
        )
        width = projection_size * len(selected_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(0.2),
            nn.Linear(width, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, len(LABELS)),
        )

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.float()
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        blocks = []
        for layer, projection in zip(
            self.selected_layers,
            self.projections,
            strict=True,
        ):
            hidden = hidden_states[layer].float()
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts
            blocks.append(projection(pooled))
        return self.classifier(torch.cat(blocks, dim=-1))


class AttributionModel(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_size: int,
        selected_layers: list[int],
    ):
        super().__init__()
        self.backbone = backbone
        self.head = MultiDepthHead(hidden_size, selected_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return self.head(output.hidden_states, attention_mask)


def load_model(
    model_dir: str,
    checkpoint_dir: str,
    selected_layers: list[int],
    num_layers: int,
) -> tuple[AttributionModel, AutoTokenizer]:
    config = json.loads(
        (Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8")
    )
    if config["labels"] != LABELS:
        raise ValueError("Checkpoint labels do not match competition labels")
    if config["selected_layers"] != selected_layers:
        raise ValueError("Selected layers do not match the checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        model_dir,
        device_map={"": 0},
        attn_implementation="sdpa",
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )
    if not hasattr(backbone, "layers"):
        raise AttributeError("Qwen backbone has no layers attribute")
    if num_layers > len(backbone.layers):
        raise ValueError("Requested layer count exceeds the model depth")
    backbone.layers = nn.ModuleList(list(backbone.layers)[:num_layers])
    backbone.config.num_hidden_layers = len(backbone.layers)
    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(
        backbone,
        Path(checkpoint_dir) / "adapter",
        is_trainable=False,
    )
    model = AttributionModel(
        backbone,
        int(backbone.config.hidden_size),
        selected_layers,
    ).to("cuda")
    model.head.load_state_dict(
        torch.load(
            Path(checkpoint_dir) / "head.pt",
            map_location="cuda",
            weights_only=False,
        )
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict(
    model: AttributionModel,
    tokenizer,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> tuple[list[str], np.ndarray]:
    loader = DataLoader(
        TextRows(frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer, max_length),
        num_workers=0,
        pin_memory=True,
    )
    all_ids: list[str] = []
    all_logits = []
    for index, batch in enumerate(loader, start=1):
        ids = list(batch.pop("ids"))
        encoded = {
            name: value.to("cuda")
            for name, value in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**encoded)
        all_ids.extend(ids)
        all_logits.append(logits.float().cpu().numpy())
        if index == 1 or index % 50 == 0:
            print(
                f"batch={index}/{len(loader)} "
                f"rows={len(all_ids)}/{len(frame)} "
                f"cuda={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
                flush=True,
            )
    return all_ids, np.concatenate(all_logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-layers", default="2,4,6,8,12,16,20,24")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected_layers = [
        int(value) for value in args.selected_layers.split(",")
    ]
    frame = pd.read_csv(Path(args.data) / "test.csv")
    frame["text"] = frame["text"].fillna("").astype(str)
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_model(
        args.model_dir,
        args.checkpoint,
        selected_layers,
        args.num_layers,
    )
    ids, logits = predict(
        model,
        tokenizer,
        frame,
        args.batch_size,
        args.max_length,
    )
    if ids != frame["id"].astype(str).tolist():
        raise RuntimeError("Prediction IDs do not preserve test order")
    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=-1,
    ).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(ids),
        labels=np.asarray(LABELS),
        logits=logits,
        probabilities=probabilities,
    )
    print(
        json.dumps(
            {
                "rows": len(ids),
                "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
