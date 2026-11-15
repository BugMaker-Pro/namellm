from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV, family  # noqa: E402


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, labels: np.ndarray):
        self.frame = frame
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return self.frame["text"].iloc[index], int(self.labels[index])


def collate(tokenizer, max_length: int):
    def inner(items):
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

    return inner


def resolve_amp_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class AttentionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attention_hidden = nn.Linear(hidden_size, 256)
        self.attention_gate = nn.Linear(hidden_size, 256)
        self.attention_output = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 4, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(LABELS)),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.norm(hidden)
        attention_hidden = torch.tanh(self.attention_hidden(values))
        attention_gate = torch.sigmoid(self.attention_gate(values))
        scores = self.attention_output(attention_hidden * attention_gate).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.einsum("bt,btd->bd", weights, values)
        masked = values.masked_fill(~mask.unsqueeze(-1), 0)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_pool = masked.sum(dim=1) / counts
        max_pool = values.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        last_index = mask.sum(dim=1) - 1
        last_pool = values[
            torch.arange(values.shape[0], device=values.device),
            last_index,
        ]
        pooled = torch.cat(
            [attention_pool, mean_pool, max_pool, last_pool],
            dim=-1,
        )
        return self.classifier(pooled)


class ModelWithAttentionHead(nn.Module):
    def __init__(self, backbone, head: AttentionHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = output.hidden_states[-1].to(self.head.norm.weight.dtype)
        return self.head(hidden, attention_mask.bool())


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**batch)
            predictions.extend(logits.float().argmax(dim=-1).cpu().tolist())
            targets.extend(labels.tolist())
    return (
        balanced_accuracy_score(targets, predictions),
        np.array(targets),
        np.array(predictions),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-4B-Base")
    parser.add_argument("--adapter", default="artifacts/qlora_fold0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="D:/tmp/namellm_attention_pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16"],
        default="auto",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    amp_dtype = resolve_amp_dtype(device, args.precision)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16,
        init_scale=1024,
        growth_interval=2000,
    )
    print(
        f"precision={str(amp_dtype).removeprefix('torch.')} "
        f"compute_capability={torch.cuda.get_device_capability(device)}",
        flush=True,
    )
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    train_idx, valid_idx = folds[args.fold]
    if args.validation_limit:
        valid_idx = valid_idx[: args.validation_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=amp_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        quantization_config=quantization_config,
        device_map={"": device.index or 0},
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    backbone = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
    head = AttentionHead(backbone.config.hidden_size).to(device)
    model = ModelWithAttentionHead(backbone, head).to(device)

    train_loader = DataLoader(
        TextDataset(frame, train_idx, labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate(tokenizer, args.max_length),
        num_workers=0,
    )
    valid_loader = DataLoader(
        TextDataset(frame, valid_idx, labels),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate(tokenizer, args.max_length),
        num_workers=0,
    )

    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_" in name or name.startswith("head."))
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps or int(updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    started = time.time()
    completed_steps = 0

    for epoch in range(1, math.ceil(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch_labels = batch.pop("labels").to(device)
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**batch)
                loss = criterion(logits, batch_labels) / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * args.grad_accum)
            if batch_index % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
                optimizer.zero_grad(set_to_none=True)
                if stepped:
                    scheduler.step()
                    completed_steps += 1
                if completed_steps % 5 == 0 or completed_steps == 1:
                    print(
                        f"step={completed_steps}/{total_steps} "
                        f"loss={np.mean(losses[-args.grad_accum:]):.4f} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
                if not stepped:
                    print(
                        f"step={completed_steps}/{total_steps} "
                        f"skipped_fp16_overflow scale={scaler.get_scale():.0f}",
                        flush=True,
                    )
                if args.max_steps and completed_steps >= args.max_steps:
                    break
        score, targets, predictions = evaluate(
            model,
            valid_loader,
            device,
            amp_dtype,
        )
        family_score = np.mean(
            [
                family(LABELS[predicted]) == family(LABELS[target])
                for predicted, target in zip(predictions, targets)
            ]
        )
        print(
            f"epoch={epoch} steps={completed_steps} "
            f"balanced_accuracy={score:.4f} family={family_score:.4f}",
            flush=True,
        )
        if args.max_steps and completed_steps >= args.max_steps:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(output_dir / "adapter")
    torch.save(head.state_dict(), output_dir / "attention_head.pt")
    np.save(output_dir / "valid_predictions.npy", predictions)
    np.save(output_dir / "valid_targets.npy", targets)
    print(f"saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()

