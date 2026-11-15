from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


@dataclass
class TokenizedRows(Dataset):
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, str | int]:
        return {"text": self.texts[index], "label": self.labels[index]}


def make_collator(tokenizer, max_length: int):
    def collate(items: list[dict[str, str | int]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [str(item["text"]) for item in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(item["label"]) for item in items],
            dtype=torch.long,
        )
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


@torch.inference_mode()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
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
    return (
        float(np.mean(losses)),
        balanced_accuracy_score(targets, predictions),
        np.asarray(targets),
        np.asarray(predictions),
        np.concatenate(probabilities),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-4B-Base")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="D:/tmp/namellm_qlora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantization",
        choices=["4bit", "none"],
        default="4bit",
    )
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
    print(
        f"fold={args.fold} train={len(train_idx)} valid={len(valid_idx)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = make_collator(tokenizer, args.max_length)
    train_dataset = TokenizedRows(
        frame["text"].iloc[train_idx].tolist(),
        labels[train_idx].tolist(),
    )
    valid_dataset = TokenizedRows(
        frame["text"].iloc[valid_idx].tolist(),
        labels[valid_idx].tolist(),
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
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model_kwargs: dict[str, object] = {
        "num_labels": len(LABELS),
        "device_map": {"": device.index or 0},
        "attn_implementation": "sdpa",
    }
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=amp_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = amp_dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    if args.quantization == "none":
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps or int(updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed_steps = 0
    for epoch in range(1, math.ceil(args.epochs) + 1):
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels")
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(**batch, labels=labels_batch.to(device))
                loss = output.loss / args.grad_accum
            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.detach()) * args.grad_accum)
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
                    elapsed = time.time() - started
                    print(
                        f"step={completed_steps}/{total_steps} "
                        f"loss={np.mean(epoch_losses[-args.grad_accum:]):.4f} "
                        f"sec={elapsed:.1f}",
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
        if completed_steps and (not args.max_steps or completed_steps < args.max_steps):
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
            model.train()
        if args.max_steps and completed_steps >= args.max_steps:
            break

    if completed_steps:
        valid_loss, valid_score, valid_targets, valid_predictions, valid_probabilities = evaluate(
            model,
            valid_loader,
            device,
            amp_dtype,
        )
        print(
            f"final valid_loss={valid_loss:.4f} valid_bacc={valid_score:.4f}",
            flush=True,
        )
        oof_frame = pd.DataFrame(
            {
                "row_index": valid_idx,
                "id": frame["id"].iloc[valid_idx].to_numpy(),
                "prompt_id": frame["prompt_id"].iloc[valid_idx].to_numpy(),
                "target": valid_targets,
                "prediction": valid_predictions,
            }
        )
        for label_index, label in enumerate(LABELS):
            oof_frame[f"prob_{label}"] = valid_probabilities[:, label_index]
        oof_frame.to_parquet(output_dir / f"fold{args.fold}_valid.parquet", index=False)
    print(
        f"peak_cuda_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}",
        flush=True,
    )
    model.save_pretrained(output_dir / f"fold{args.fold}")
    tokenizer.save_pretrained(output_dir / f"fold{args.fold}")
    print(f"saved adapter to {output_dir / f'fold{args.fold}'}", flush=True)


if __name__ == "__main__":
    main()
