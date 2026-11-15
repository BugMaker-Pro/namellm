from __future__ import annotations

import argparse
import json
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


def unwrap_sequence_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(
        f"Unsupported embedding model output type: {type(output).__name__}"
    )


def pool_embeddings(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if hidden.ndim == 2:
        return hidden.float()
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden shape (B, T, H), got {hidden.shape}")
    mask = attention_mask.bool()
    if pooling == "cls":
        return hidden[:, 0].float()
    mask_float = mask.unsqueeze(-1).float()
    counts = mask_float.sum(dim=1).clamp_min(1.0)
    return (hidden.float() * mask_float).sum(dim=1) / counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--prompt-column", default="prompt_id")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--episode-mode",
        action="store_true",
        help="Add a singleton episode dimension for LUAR-style models.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if input_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(input_path)
    else:
        frame = pd.read_csv(input_path)
    if args.limit:
        frame = frame.iloc[: args.limit].copy().reset_index(drop=True)
    frame[args.text_column] = (
        frame[args.text_column].fillna("").astype(str)
    )
    row_indices = np.arange(len(frame), dtype=np.int64)
    if args.label_column in frame.columns:
        label_to_index = {label: index for index, label in enumerate(LABELS)}
        labels = frame[args.label_column].map(label_to_index).to_numpy(
            dtype=np.int64
        )
        if np.isnan(labels).any():
            raise ValueError("Input CSV contains unknown model labels")
    else:
        labels = np.full(len(frame), -1, dtype=np.int64)
    if args.prompt_column in frame.columns:
        prompt_ids = frame[args.prompt_column].to_numpy()
    else:
        prompt_ids = np.full(len(frame), "", dtype=object)
    if args.id_column in frame.columns:
        ids = frame[args.id_column].astype(str).to_numpy()
    else:
        ids = row_indices.astype(str)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_dtype = (
        resolve_amp_dtype(device, args.precision)
        if device.type == "cuda"
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {
        "dtype": amp_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        model_kwargs["device_map"] = {"": device.index or 0}
    model = AutoModel.from_pretrained(
        args.model,
        **model_kwargs,
    ).eval()
    if device.type != "cuda":
        model.to(device)
    model.config.use_cache = False

    loader = DataLoader(
        TextRows(frame[args.text_column].tolist(), row_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer, args.max_length),
        num_workers=0,
    )
    features: np.ndarray | None = None
    started = time.time()
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            positions = batch.pop("row_indices").numpy()
            batch = {name: value.to(device) for name, value in batch.items()}
            if args.episode_mode:
                batch = {
                    name: value
                    for name, value in batch.items()
                    if name in {"input_ids", "attention_mask"}
                }
                batch["input_ids"] = batch["input_ids"].unsqueeze(1)
                batch["attention_mask"] = batch["attention_mask"].unsqueeze(1)
            with torch.autocast(
                "cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                output = model(**batch)
            hidden = unwrap_sequence_output(output)
            pooled = pool_embeddings(
                hidden,
                batch["attention_mask"],
                args.pooling,
            )
            pooled_np = pooled.cpu().numpy().astype(np.float32)
            if features is None:
                features = np.empty(
                    (len(frame), pooled_np.shape[1]),
                    dtype=np.float32,
                )
            if features.shape[1] != pooled_np.shape[1]:
                raise ValueError("Embedding width changed between batches")
            features[positions] = pooled_np
            completed += len(positions)
            if completed % 250 == 0 or completed == len(frame):
                elapsed = time.time() - started
                remaining = elapsed / completed * (len(frame) - completed)
                print(
                    f"{completed}/{len(frame)} elapsed={elapsed:.1f}s "
                    f"remaining={remaining:.1f}s",
                    flush=True,
                )
    if features is None:
        raise RuntimeError("No embeddings were extracted")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=features,
        y=labels,
        ids=ids,
        prompt_ids=prompt_ids,
        row_indices=row_indices,
        pooling=np.array(args.pooling),
        model=np.array(args.model),
    )
    metadata = {
        "model": args.model,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "rows": len(frame),
        "embedding_dim": int(features.shape[1]),
        "precision": str(amp_dtype).removeprefix("torch."),
        "trust_remote_code": args.trust_remote_code,
        "episode_mode": args.episode_mode,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    peak = (
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
        if device.type == "cuda"
        else "n/a"
    )
    print(
        f"saved {output} shape={features.shape} peak_cuda={peak}",
        flush=True,
    )


if __name__ == "__main__":
    main()
