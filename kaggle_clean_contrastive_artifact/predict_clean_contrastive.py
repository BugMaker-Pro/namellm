from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


LABELS = sorted(
    [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "glm-5.3",
        "glm-5.3-flash",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "kimi-k3",
        "qwen3.8-max",
    ]
)
FAMILIES = ("claude", "gemini", "glm", "gpt", "kimi", "qwen")
QKV_LAYERS = (
    ("k", 2, (32, 64), (96, 128)),
    ("k", 3, (32, 64), (96, 128)),
    ("v", 20, (64, 96),),
    ("v", 23, (64, 96),),
    ("q", 35, (0, 32),),
)


def family(model: str) -> str:
    return model.split("-")[0].replace("qwen3.8", "qwen")


VENDOR_FOR_CLASS = np.asarray(
    [FAMILIES.index(family(label)) for label in LABELS],
    dtype=np.int64,
)


class MaskedAttentionPool(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        values = self.projection(values)
        score = self.score(values).squeeze(-1)
        score = score.masked_fill(~mask, -1e4)
        weight = score.softmax(dim=-1)
        return torch.einsum("bt,btd->bd", weight, values)


class LayerwiseHierarchyEncoder(nn.Module):
    def __init__(
        self,
        hidden_width: int,
        qkv_width: int,
        d_model: int,
        heads: int,
        dropout: float,
        max_length: int,
    ):
        super().__init__()
        hidden_model = d_model
        self.layer8_norm = nn.LayerNorm(hidden_width)
        self.layer36_norm = nn.LayerNorm(hidden_width)
        self.layer8_projection = nn.Linear(hidden_width, hidden_model)
        self.layer36_projection = nn.Linear(hidden_width, hidden_model)
        self.hidden_norm = nn.LayerNorm(hidden_model * 2)
        hidden_layer = nn.TransformerEncoderLayer(
            d_model=hidden_model * 2,
            nhead=heads,
            dim_feedforward=hidden_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hidden_encoder = nn.TransformerEncoder(
            hidden_layer,
            num_layers=1,
            norm=nn.LayerNorm(hidden_model * 2),
        )
        self.hidden_pool = MaskedAttentionPool(
            hidden_model * 2,
            hidden_model,
            dropout,
        )
        qkv_hidden = d_model // 2
        self.qkv_pools = nn.ModuleList(
            [
                MaskedAttentionPool(
                    sum(stop - start for start, stop in slices),
                    qkv_hidden,
                    dropout,
                )
                for _, _, *slices in QKV_LAYERS
            ]
        )
        self.layer_embedding = nn.Parameter(
            torch.randn(len(QKV_LAYERS), qkv_hidden) * 0.02
        )
        layer_encoder = nn.TransformerEncoderLayer(
            d_model=qkv_hidden,
            nhead=heads,
            dim_feedforward=qkv_hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(
            layer_encoder,
            num_layers=1,
            norm=nn.LayerNorm(qkv_hidden),
        )
        pooled_width = hidden_model + qkv_hidden * 2
        self.output_norm = nn.LayerNorm(pooled_width)
        self.group3 = nn.Linear(pooled_width, 3)
        self.vendor = nn.Linear(pooled_width, len(FAMILIES))
        self.expert = nn.Linear(pooled_width, len(LABELS))
        self.vendor_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "vendor_for_class",
            torch.as_tensor(VENDOR_FOR_CLASS),
        )

    def forward(
        self,
        layer8: torch.Tensor,
        layer36: torch.Tensor,
        qkv: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.cat(
            [
                self.layer8_projection(self.layer8_norm(layer8)),
                self.layer36_projection(self.layer36_norm(layer36)),
            ],
            dim=-1,
        )
        hidden = self.hidden_encoder(
            self.hidden_norm(hidden),
            src_key_padding_mask=~mask,
        )
        hidden_pooled = self.hidden_pool(hidden, mask)
        layer_pooled = []
        for pool, (_, _, *slices) in zip(
            self.qkv_pools,
            QKV_LAYERS,
        ):
            selected = torch.cat(
                [
                    qkv[:, :, start:stop]
                    for start, stop in slices
                ],
                dim=-1,
            )
            layer_pooled.append(pool(selected, mask))
        layer_values = torch.stack(layer_pooled, dim=1)
        layer_values = self.layer_encoder(
            layer_values + self.layer_embedding.unsqueeze(0)
        )
        layer_mean = layer_values.mean(dim=1)
        layer_max = layer_values.max(dim=1).values
        pooled = self.output_norm(
            torch.cat(
                [hidden_pooled, layer_mean, layer_max],
                dim=-1,
            )
        )
        expert = self.expert(pooled)
        vendor = self.vendor(pooled)
        return (
            expert
            + self.vendor_scale.clamp(0.25, 4.0)
            * vendor[:, self.vendor_for_class]
        )


def load_encoders(
    checkpoint_root: Path,
) -> list[LayerwiseHierarchyEncoder]:
    encoders = []
    for fold in range(5):
        path = checkpoint_root / f"fold{fold}" / "model_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if list(state["labels"]) != LABELS:
            raise ValueError(
                f"Checkpoint label order differs at {path}: "
                f"{state['labels']}"
            )
        config = state["config"]
        encoder = LayerwiseHierarchyEncoder(
            2560,
            640,
            int(config["d_model"]),
            int(config["heads"]),
            float(config["dropout"]),
            384,
        )
        encoder.load_state_dict(state["encoder"], strict=True)
        encoders.append(encoder.to("cuda").eval())
    return encoders


def qkv_features(
    captured: dict[str, torch.Tensor],
    local_row: int,
    length: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
) -> torch.Tensor:
    values = []
    for _, layer, *_ in QKV_LAYERS:
        q = captured[f"{layer}:q_proj"].float().view(
            batch_size,
            -1,
            num_heads,
            head_dim,
        )
        k = captured[f"{layer}:k_proj"].float().view(
            batch_size,
            -1,
            num_kv_heads,
            head_dim,
        )
        v = captured[f"{layer}:v_proj"].float().view(
            batch_size,
            -1,
            num_kv_heads,
            head_dim,
        )
        repeats = num_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
        qn = torch.linalg.vector_norm(q, dim=-1)
        kn = torch.linalg.vector_norm(k, dim=-1)
        vn = torch.linalg.vector_norm(v, dim=-1)
        qk = (
            torch.nn.functional.normalize(q, dim=-1)
            * torch.nn.functional.normalize(k, dim=-1)
        ).sum(dim=-1)
        layer_values = torch.stack([qn, kn, vn, qk], dim=3)
        values.append(
            layer_values[local_row, :length].reshape(length, -1)
        )
    return torch.cat(values, dim=1)


def predict_frame(
    frame: pd.DataFrame,
    model_path: str,
    checkpoint_root: Path,
    batch_size: int = 4,
    max_length: int = 384,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoders = load_encoders(checkpoint_root)
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    backbone = getattr(model, "model", model)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer in sorted({item[1] for item in QKV_LAYERS}):
        attention = backbone.layers[layer - 1].self_attn
        for projection in ("q_proj", "k_proj", "v_proj"):
            module = getattr(attention, projection)

            def hook(
                _module,
                _inputs,
                output,
                key=f"{layer}:{projection}",
            ):
                captured[key] = (
                    output[0] if isinstance(output, tuple) else output
                )

            handles.append(module.register_forward_hook(hook))

    texts = frame["text"].fillna("").astype(str).tolist()
    probabilities = np.zeros(
        (len(frame), len(LABELS)),
        dtype=np.float64,
    )
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                stop = min(start + batch_size, len(texts))
                batch = tokenizer(
                    texts[start:stop],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {
                    name: value.to("cuda")
                    for name, value in batch.items()
                }
                captured.clear()
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model(
                        **batch,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                lengths = batch["attention_mask"].sum(dim=1)
                rows = []
                for local_row in range(stop - start):
                    length = int(lengths[local_row])
                    rows.append(
                        (
                            result.hidden_states[8][
                                local_row, :length
                            ].float(),
                            result.hidden_states[36][
                                local_row, :length
                            ].float(),
                            qkv_features(
                                captured,
                                local_row,
                                length,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                stop - start,
                            ),
                        )
                    )
                batch_max = max(len(row[0]) for row in rows)
                layer8 = torch.zeros(
                    (len(rows), batch_max, 2560),
                    dtype=torch.float32,
                    device="cuda",
                )
                layer36 = torch.zeros_like(layer8)
                qkv = torch.zeros(
                    (len(rows), batch_max, 640),
                    dtype=torch.float32,
                    device="cuda",
                )
                mask = torch.zeros(
                    (len(rows), batch_max),
                    dtype=torch.bool,
                    device="cuda",
                )
                for local_row, (value8, value36, value_qkv) in enumerate(rows):
                    length = len(value8)
                    layer8[local_row, :length] = value8
                    layer36[local_row, :length] = value36
                    qkv[local_row, :length] = value_qkv
                    mask[local_row, :length] = True
                fold_probabilities = []
                for encoder in encoders:
                    logits = encoder(layer8, layer36, qkv, mask)
                    fold_probabilities.append(
                        logits.float().softmax(dim=1)
                    )
                probabilities[start:stop] = (
                    torch.stack(fold_probabilities).mean(dim=0)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if stop % 100 == 0 or stop == len(texts):
                    print(
                        f"clean contrastive {stop}/{len(texts)}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
        del model, encoders
        torch.cuda.empty_cache()
    ids = frame["id"].astype(str).to_numpy()
    return ids, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    ids, probabilities = predict_frame(
        frame,
        args.model,
        Path(args.checkpoint_root),
        args.batch_size,
        args.max_length,
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite clean contrastive probabilities")
    np.savez_compressed(
        args.output,
        ids=ids,
        labels=np.asarray(LABELS),
        probabilities=probabilities,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
