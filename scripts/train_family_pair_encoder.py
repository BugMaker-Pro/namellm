from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(value: object) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip()


def text_key(value: object) -> str:
    return hashlib.md5(
        normalize_text(value).encode("utf-8")
    ).hexdigest()


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.labels[index])


def make_collator(tokenizer, max_length: int):
    def collate(items):
        texts, labels = zip(*items)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


class PairEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        dropout: float,
        local_files_only: bool,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


def predict(
    model: PairEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probabilities.append(
                torch.softmax(logits.float(), dim=1).cpu().numpy()
            )
            targets.append(batch["labels"].numpy())
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(targets, axis=0),
    )


def split_external(
    frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "prompt_group" in frame.columns:
        groups = frame["prompt_group"].fillna("").astype(str).to_numpy()
    else:
        groups = np.asarray(
            [f"row_{index}" for index in range(len(frame))],
            dtype=object,
        )
    blank = groups == ""
    groups[blank] = [
        f"row_{index}" for index in np.flatnonzero(blank)
    ]
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    validation_groups = set(
        unique[: max(1, int(len(unique) * validation_fraction))]
    )
    is_validation = np.isin(groups, list(validation_groups))
    validation = frame[is_validation].copy()
    training = frame[~is_validation].copy()
    if validation.empty or training.empty:
        raise ValueError("External prompt-group split is empty")
    return training, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="gemini-3.7-flash,gemini-3.8-flash",
    )
    parser.add_argument(
        "--pool",
        default=str(
            ROOT
            / "artifacts"
            / "open_pool_filtered"
            / "ntl_open_pool_filtered.parquet"
        ),
    )
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--model-path", default="answerdotai/ModernBERT-base")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "family_pair_gemini_37_38"),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-per-label", type=int, default=12000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Train with official rows from the other folds; test this fold.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)

    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    if len(labels) != 2 or any(label not in LABELS for label in labels):
        raise ValueError("--labels must contain two known competition labels")
    label_to_local = {label: index for index, label in enumerate(labels)}
    train = pd.read_csv(args.train_csv)
    train["text"] = train["text"].fillna("").astype(str).map(normalize_text)
    official_keys = set(train["text"].map(text_key))
    pool = pd.read_parquet(args.pool)
    pool["text"] = pool["text"].fillna("").astype(str).map(normalize_text)
    pool["_key"] = pool["text"].map(text_key)
    pool = pool[pool["model"].isin(labels)]
    pool = pool[~pool["_key"].isin(official_keys)].drop_duplicates("_key")
    pool = pool[pool["text"].ne("")]
    pool = (
        pool.sample(frac=1.0, random_state=args.seed)
        .groupby("model", group_keys=False)
        .head(args.max_per_label)
    )
    pool["target"] = pool["model"].map(label_to_local)
    training, validation = split_external(
        pool,
        args.validation_fraction,
        args.seed,
    )
    train_texts = training["text"].tolist()
    train_targets = training["target"].to_numpy(dtype=np.int64)
    valid_texts = validation["text"].tolist()
    valid_targets = validation["target"].to_numpy(dtype=np.int64)
    official = train[train["model"].isin(labels)].copy()
    official["target"] = official["model"].map(label_to_local)
    if args.fold >= 0:
        official_folds = list(
            GroupKFold(5).split(
                official,
                official["target"],
                official["prompt_id"],
            )
        )
        if args.fold >= len(official_folds):
            raise ValueError(f"Fold {args.fold} is out of range")
        official_train_index, official_test_index = official_folds[args.fold]
        official_train = official.iloc[official_train_index]
        official_test = official.iloc[official_test_index]
        train_texts = train_texts + official_train["text"].tolist()
        train_targets = np.concatenate(
            [
                train_targets,
                official_train["target"].to_numpy(dtype=np.int64),
            ]
        )
    else:
        official_train = official.iloc[[]]
        official_test = official
    official_texts = official_test["text"].tolist()
    official_targets = official_test["target"].to_numpy(dtype=np.int64)
    print(
        f"external_train={len(train_texts)} external_valid={len(valid_texts)} "
        f"official_train={len(official_train)} "
        f"official_test={len(official_texts)} fold={args.fold} "
        f"official_counts={official_test['model'].value_counts().to_dict()}",
        flush=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise RuntimeError("Tokenizer has no pad token")
    collator = make_collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        TextDataset(train_texts, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        TextDataset(valid_texts, valid_targets),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    official_loader = DataLoader(
        TextDataset(official_texts, official_targets),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    model = PairEncoder(
        args.model_path,
        args.dropout,
        args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / max(1, args.grad_accum)
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.1)),
        num_training_steps=max(1, total_updates),
    )
    counts = np.bincount(train_targets, minlength=2).astype(np.float32)
    class_weight = counts.sum() / np.clip(counts, 1, None)
    class_weight /= class_weight.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight, device=device)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    started = time.monotonic()
    completed_updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            (loss / args.grad_accum).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            if (
                batch_index % args.grad_accum == 0
                or batch_index == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1
            if args.max_updates and completed_updates >= args.max_updates:
                break
        valid_prob, valid_y = predict(model, valid_loader, device)
        official_prob, official_y = predict(model, official_loader, device)
        valid_score = float(
            balanced_accuracy_score(valid_y, valid_prob.argmax(axis=1))
        )
        official_score = float(
            balanced_accuracy_score(
                official_y,
                official_prob.argmax(axis=1),
            )
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "external_valid_balanced_accuracy": valid_score,
            "official_test_balanced_accuracy": official_score,
            "updates": completed_updates,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if args.max_updates and completed_updates >= args.max_updates:
            break
    if best_state is None:
        raise RuntimeError("No best state was selected")
    model.load_state_dict(best_state)
    valid_prob, valid_y = predict(model, valid_loader, device)
    official_prob, official_y = predict(model, official_loader, device)
    valid_score = float(
        balanced_accuracy_score(valid_y, valid_prob.argmax(axis=1))
    )
    official_score = float(
        balanced_accuracy_score(
            official_y,
            official_prob.argmax(axis=1),
        )
    )
    if args.fold >= 0:
        fold_scores = [
            {
                "fold": args.fold,
                "balanced_accuracy": official_score,
            }
        ]
    else:
        fold_scores = []
        group_folds = list(
            GroupKFold(5).split(
                official,
                official["target"],
                official["prompt_id"],
            )
        )
        for fold, (_, valid_index) in enumerate(group_folds):
            fold_scores.append(
                {
                    "fold": fold,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            official_targets[valid_index],
                            official_prob[valid_index].argmax(axis=1),
                        )
                    ),
                }
            )
    full_official = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    for local_index, label in enumerate(labels):
        full_official[
            official_test.index,
            LABELS.index(label),
        ] = official_prob[:, local_index]
    np.save(output_dir / "official_pair_probs.npy", official_prob)
    np.save(output_dir / "official_pair_targets.npy", official_y)
    np.save(output_dir / "full_official_probs.npy", full_official)
    torch.save(
        {
            "labels": labels,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "external_valid_score": valid_score,
        },
        output_dir / "best.pt",
    )
    result = {
        "labels": labels,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "external_valid_balanced_accuracy": float(valid_score),
        "official_pair_balanced_accuracy": official_score,
        "official_folds": fold_scores,
        "history": history,
        "pool": args.pool,
        "model_path": args.model_path,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
