from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


class TextRows(Dataset):
    def __init__(self, texts: list[str], row_indices: np.ndarray):
        self.texts = texts
        self.row_indices = row_indices

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.row_indices[index])


def make_collator(tokenizer, max_length: int):
    def collate(items: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, row_indices = zip(*items)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["row_indices"] = torch.tensor(row_indices, dtype=torch.long)
        return encoded

    return collate


def resolve_amp_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def summarize_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layers: list[int],
) -> np.ndarray:
    mask = attention_mask.bool()
    mask_float = mask.unsqueeze(-1).to(torch.float32)
    counts = mask_float.sum(dim=1).clamp_min(1.0)
    lengths = mask.sum(dim=1)
    batch_indices = torch.arange(mask.shape[0], device=mask.device)
    blocks: list[torch.Tensor] = []

    for layer_index in layers:
        hidden = hidden_states[layer_index].float()
        mean = (hidden * mask_float).sum(dim=1) / counts
        centered = (hidden - mean.unsqueeze(1)) * mask_float
        variance = centered.square().sum(dim=1) / counts
        std = variance.sqrt()
        last = hidden[batch_indices, lengths - 1]

        differences = (hidden[:, 1:] - hidden[:, :-1]).abs()
        adjacent_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        adjacent_counts = (
            adjacent_mask.to(torch.float32).sum(dim=1).clamp_min(1.0)
        )
        delta = (differences * adjacent_mask).sum(dim=1) / adjacent_counts

        blocks.extend([mean, std, last, delta])

    return torch.cat(blocks, dim=-1).cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--prompt-column", default="prompt_id")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16"],
        default="auto",
    )
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",")]
    device = torch.device("cuda")
    amp_dtype = resolve_amp_dtype(device, args.precision)
    print(
        f"precision={str(amp_dtype).removeprefix('torch.')} "
        f"compute_capability={torch.cuda.get_device_capability(device)}",
        flush=True,
    )

    frame = pd.read_csv(args.input_csv)
    if args.limit:
        frame = frame.iloc[: args.limit].copy()
    row_indices = np.arange(len(frame), dtype=np.int64)
    if args.label_column in frame.columns:
        label_to_index = {label: index for index, label in enumerate(LABELS)}
        labels = np.array(
            [label_to_index[value] for value in frame[args.label_column]]
        )
    else:
        labels = np.full(len(frame), -1, dtype=np.int64)
    if args.prompt_column in frame.columns:
        prompt_ids = frame[args.prompt_column].to_numpy()
    else:
        prompt_ids = np.full(len(frame), "", dtype=object)
    if args.id_column in frame.columns:
        ids = frame[args.id_column].to_numpy()
    else:
        ids = row_indices.astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader(
        TextRows(frame[args.text_column].tolist(), row_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer, args.max_length),
        num_workers=0,
    )

    model = AutoModel.from_pretrained(
        args.model,
        dtype=amp_dtype,
        device_map={"": device.index or 0},
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    hidden_size = int(model.config.hidden_size)
    features = np.empty(
        (len(frame), len(layers) * 4 * hidden_size),
        dtype=np.float32,
    )
    started = time.time()
    completed = 0

    with torch.inference_mode():
        for batch in loader:
            positions = batch.pop("row_indices").numpy()
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(
                    **batch,
                    output_hidden_states=True,
                    return_dict=True,
                )
            features[positions] = summarize_hidden_states(
                output.hidden_states,
                batch["attention_mask"],
                layers,
            )
            completed += len(positions)
            if completed % 50 == 0 or completed == len(frame):
                elapsed = time.time() - started
                remaining = elapsed / completed * (len(frame) - completed)
                print(
                    f"{completed}/{len(frame)} elapsed={elapsed:.1f}s "
                    f"remaining={remaining:.1f}s",
                    flush=True,
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=features,
        y=labels,
        ids=ids,
        prompt_ids=prompt_ids,
        row_indices=row_indices,
        layers=np.array(layers),
    )
    print(
        f"saved {output_path} shape={features.shape} "
        f"peak_cuda_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
