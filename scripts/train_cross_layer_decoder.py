from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV
from namellm.style import style_frame


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_length: int = 512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]].to(values.dtype)


class CrossLayerDataset(Dataset):
    def __init__(
        self,
        query_dir: Path,
        key_dir: Path,
        indices: np.ndarray,
        styles: np.ndarray | None,
    ):
        self.query_storage = np.load(query_dir / "sequences.npy", mmap_mode="r")
        self.key_storage = np.load(key_dir / "sequences.npy", mmap_mode="r")
        self.query_metadata = np.load(
            query_dir / "metadata.npz",
            allow_pickle=True,
        )
        self.key_metadata = np.load(
            key_dir / "metadata.npz",
            allow_pickle=True,
        )
        if not np.array_equal(
            self.query_metadata["row_indices"],
            self.key_metadata["row_indices"],
        ):
            raise ValueError("Query and key sequence rows differ")
        if not np.array_equal(
            self.query_metadata["offsets"],
            self.key_metadata["offsets"],
        ):
            raise ValueError("Query and key tokenization offsets differ")
        self.indices = indices
        self.styles = styles

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        offsets = self.query_metadata["offsets"]
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        query = np.asarray(
            self.query_storage[start:stop],
            dtype=np.float32,
        )
        key = np.asarray(
            self.key_storage[start:stop],
            dtype=np.float32,
        )
        style = (
            self.styles[index]
            if self.styles is not None
            else np.zeros(0, dtype=np.float32)
        )
        return (
            query,
            key,
            int(self.query_metadata["y"][index]),
            style,
        )


def collate(items):
    queries, keys, labels, styles = zip(*items)
    max_length = max(len(query) for query in queries)
    hidden_size = queries[0].shape[1]
    query_batch = torch.zeros(
        (len(queries), max_length, hidden_size),
        dtype=torch.float32,
    )
    key_batch = torch.zeros_like(query_batch)
    mask = torch.zeros((len(queries), max_length), dtype=torch.bool)
    for row, (query, key) in enumerate(zip(queries, keys)):
        length = len(query)
        if len(key) != length:
            raise ValueError("Query and key lengths differ")
        query_batch[row, :length] = torch.from_numpy(query)
        key_batch[row, :length] = torch.from_numpy(key)
        mask[row, :length] = True
    return (
        query_batch,
        key_batch,
        mask,
        torch.tensor(labels),
        torch.from_numpy(np.stack(styles)),
    )


class CrossLayerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        style_size: int,
        d_model: int = 256,
        decoder_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        causal_query: bool = True,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.query_norm = nn.LayerNorm(input_size)
        self.key_norm = nn.LayerNorm(input_size)
        self.query_projection = nn.Linear(input_size, d_model)
        self.key_projection = nn.Linear(input_size, d_model)
        self.causal_query = causal_query
        self.position = SinusoidalPositionEncoding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        combined_size = d_model * 4
        self.style_branch = None
        if style_size:
            self.style_branch = nn.Sequential(
                nn.LayerNorm(style_size),
                nn.Linear(style_size, 128),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            combined_size += 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Dropout(0.25),
            nn.Linear(combined_size, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, len(LABELS)),
        )

    def forward(
        self,
        query_values: torch.Tensor,
        key_values: torch.Tensor,
        mask: torch.Tensor,
        styles: torch.Tensor,
    ) -> torch.Tensor:
        query = self.position(
            self.query_projection(self.query_norm(query_values))
        )
        key = self.position(
            self.key_projection(self.key_norm(key_values))
        )
        positions = torch.arange(
            query.shape[1],
            device=query.device,
        )
        causal_mask = (
            positions[None, :] > positions[:, None]
            if self.causal_query
            else None
        )
        decoded = self.decoder(
            query,
            key,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~mask,
        )
        masked = decoded.masked_fill(~mask.unsqueeze(-1), 0)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = masked.sum(dim=1) / counts
        maximum = decoded.masked_fill(
            ~mask.unsqueeze(-1),
            -1e4,
        ).max(dim=1).values
        last_index = mask.sum(dim=1) - 1
        last = decoded[
            torch.arange(decoded.shape[0], device=decoded.device),
            last_index,
        ]
        query_summary = (
            self.query_projection(self.query_norm(query_values))
            .masked_fill(~mask.unsqueeze(-1), 0)
            .sum(dim=1)
            / counts
        )
        pooled = torch.cat([mean, maximum, last, query_summary], dim=-1)
        if self.style_branch is not None:
            if styles.shape[1] == 0:
                raise ValueError("Style features are required")
            pooled = torch.cat([pooled, self.style_branch(styles)], dim=-1)
        return self.classifier(pooled)


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for query, key, mask, labels, styles in loader:
            logits = model(
                query.to(device),
                key.to(device),
                mask.to(device),
                styles.to(device),
            )
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
            targets.append(labels)
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def balanced_accuracy(targets: np.ndarray, prediction: np.ndarray) -> float:
    recalls = [
        (prediction[targets == index] == index).mean()
        for index in range(len(LABELS))
        if (targets == index).any()
    ]
    return float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", default="features/qwen3_seq_fold0")
    parser.add_argument(
        "--key-dir",
        default="features/qwen3_seq_fold0_layer8",
    )
    parser.add_argument("--use-style", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--causal-query",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    query_dir = Path(args.query_dir)
    key_dir = Path(args.key_dir)
    metadata = np.load(query_dir / "metadata.npz", allow_pickle=True)
    train_indices = np.where(metadata["train_mask"])[0]
    valid_indices = np.where(metadata["valid_mask"])[0]
    styles = None
    if args.use_style:
        frame = pd.read_csv(TRAIN_CSV)
        row_indices = metadata["row_indices"]
        styles = style_frame(
            frame["text"].fillna("").astype(str)
        ).to_numpy(np.float32)[row_indices]
        mean = styles[metadata["train_mask"]].mean(axis=0)
        std = styles[metadata["train_mask"]].std(axis=0).clip(min=1e-6)
        styles = (styles - mean) / std

    train_loader = DataLoader(
        CrossLayerDataset(query_dir, key_dir, train_indices, styles),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        CrossLayerDataset(query_dir, key_dir, valid_indices, styles),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    device = torch.device("cuda")
    input_size = int(metadata["hidden_size"][0])
    model = CrossLayerDecoder(
        input_size=input_size,
        style_size=styles.shape[1] if styles is not None else 0,
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        dropout=args.dropout,
        causal_query=args.causal_query,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score = -1.0
    best_state = None
    best_probabilities = None
    best_targets = None
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for query, key, mask, labels, batch_styles in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    query.to(device),
                    key.to(device),
                    mask.to(device),
                    batch_styles.to(device),
                )
                loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities, targets = evaluate(model, valid_loader, device)
        score = balanced_accuracy(targets, probabilities.argmax(axis=1))
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"balanced_accuracy={score:.4f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            best_probabilities = probabilities
            best_targets = targets
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, best_probabilities)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "best_balanced_accuracy": best_score,
                "targets": best_targets.tolist(),
                "epochs": epoch,
                "elapsed": time.time() - started,
                "query_dir": str(query_dir),
                "key_dir": str(key_dir),
                "use_style": args.use_style,
                "d_model": args.d_model,
                "decoder_layers": args.decoder_layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "causal_query": args.causal_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_balanced_accuracy={best_score:.4f} saved={output}")


if __name__ == "__main__":
    main()
