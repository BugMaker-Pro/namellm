from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.labels[index])


def make_collator(tokenizer, max_length: int):
    def collate(items: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*items)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
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


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(**batch, labels=labels.to(device))
            logits = output.logits.float()
            losses.append(float(output.loss))
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.tolist())
    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)
    return (
        float(np.mean(losses)),
        balanced_accuracy_score(targets_array, predictions_array),
        targets_array,
        predictions_array,
        np.concatenate(probabilities),
    )


def train_fold(
    frame: pd.DataFrame,
    labels: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    args,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = make_collator(tokenizer, args.max_length)
    train_dataset = TextDataset(
        frame["text"].iloc[train_index].tolist(),
        labels[train_index],
    )
    valid_dataset = TextDataset(
        frame["text"].iloc[valid_index].tolist(),
        labels[valid_index],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16,
        init_scale=1024,
        growth_interval=2000,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    started = time.time()
    completed_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch_labels = batch.pop("labels").to(device)
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**batch).logits
                loss = criterion(logits, batch_labels) / args.grad_accum
            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.detach()) * args.grad_accum)
            if batch_index % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
                optimizer.zero_grad(set_to_none=True)
                if stepped:
                    scheduler.step()
                    completed_steps += 1
                if completed_steps % 20 == 0 or completed_steps == 1:
                    print(
                        f"step={completed_steps}/{total_steps} "
                        f"loss={np.mean(epoch_losses[-args.grad_accum:]):.4f} "
                        f"sec={time.time() - started:.1f}",
                        flush=True,
                    )
                if not stepped:
                    print(
                        f"step={completed_steps}/{total_steps} "
                        f"skipped_fp16_overflow scale={scaler.get_scale():.0f}",
                        flush=True,
                    )
        valid_loss, valid_score, _, _, _ = evaluate(
            model,
            valid_loader,
            device,
            amp_dtype,
        )
        print(
            f"epoch={epoch} train_loss={np.mean(epoch_losses):.4f} "
            f"valid_loss={valid_loss:.4f} valid_bacc={valid_score:.4f}",
            flush=True,
        )

    _, valid_score, targets, predictions, probabilities = evaluate(
        model,
        valid_loader,
        device,
        amp_dtype,
    )
    print(f"final valid_bacc={valid_score:.4f}", flush=True)
    output = pd.DataFrame(
        {
            "row_index": valid_index,
            "id": frame["id"].iloc[valid_index].to_numpy(),
            "prompt_id": frame["prompt_id"].iloc[valid_index].to_numpy(),
            "target": targets,
            "prediction": predictions,
        }
    )
    for label_index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probabilities[:, label_index]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    amp_dtype = resolve_amp_dtype(device, args.precision)
    print(
        f"precision={str(amp_dtype).removeprefix('torch.')} "
        f"compute_capability={torch.cuda.get_device_capability(device)}",
        flush=True,
    )

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    selected_folds = range(len(folds)) if args.fold < 0 else [args.fold]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}

    for fold in selected_folds:
        fold_output = output_dir / f"fold{fold}_valid.parquet"
        if args.resume and fold_output.exists():
            saved = pd.read_parquet(fold_output)
            score = balanced_accuracy_score(saved["target"], saved["prediction"])
            metrics[f"fold{fold}"] = float(score)
            print(f"fold={fold} resumed balanced_accuracy={score:.4f}", flush=True)
            continue

        train_index, valid_index = folds[fold]
        if args.train_limit:
            train_index = train_index[: args.train_limit]
        if args.validation_limit:
            valid_index = valid_index[: args.validation_limit]
        print(
            f"fold={fold} train={len(train_index)} valid={len(valid_index)}",
            flush=True,
        )
        output = train_fold(
            frame,
            labels,
            train_index,
            valid_index,
            args,
            device,
            amp_dtype,
        )
        output.to_parquet(fold_output, index=False)
        score = balanced_accuracy_score(output["target"], output["prediction"])
        metrics[f"fold{fold}"] = float(score)
        print(f"saved {fold_output}", flush=True)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
