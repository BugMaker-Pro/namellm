from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import peft.import_utils as peft_import_utils
import peft.tuners.lora.torchao as peft_lora_torchao
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


class TextRows(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels.astype(np.int64, copy=False)

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_dir: str,
    selected_layers: list[int],
    num_layers: int,
    load_in_4bit: bool,
    lora_r: int,
) -> tuple[AttributionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, object] = {
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
        "dtype": torch.float16,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    backbone = AutoModel.from_pretrained(model_dir, **model_kwargs)
    if not hasattr(backbone, "layers"):
        raise AttributeError("Qwen backbone has no layers attribute")
    max_layer = max(selected_layers)
    if max_layer >= len(backbone.layers) or num_layers > len(backbone.layers):
        raise ValueError("Requested layer exceeds the model depth")
    backbone.layers = nn.ModuleList(list(backbone.layers)[:num_layers])
    backbone.config.num_hidden_layers = len(backbone.layers)
    backbone.config.use_cache = False
    prepare_kwargs = {"use_gradient_checkpointing": True}
    if load_in_4bit:
        backbone = prepare_model_for_kbit_training(
            backbone,
            **prepare_kwargs,
        )
    else:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_import_utils.is_torchao_available = lambda: False
    peft_lora_torchao.is_torchao_available = lambda: False
    backbone = get_peft_model(backbone, lora)
    backbone.enable_input_require_grads()
    model = AttributionModel(
        backbone,
        int(backbone.config.hidden_size),
        selected_layers,
    ).to("cuda")
    for parameter in model.head.parameters():
        parameter.data = parameter.data.float()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"model={model_dir} layers={num_layers} "
        f"selected={selected_layers} trainable={trainable:,}",
        flush=True,
    )
    return model, tokenizer


@torch.inference_mode()
def evaluate(
    model: AttributionModel,
    loader: DataLoader,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {name: value.to("cuda") for name, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**batch)
        logits = logits.float()
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(labels.tolist())
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    return (
        float(balanced_accuracy_score(target_array, prediction_array)),
        target_array,
        prediction_array,
        np.concatenate(probabilities),
    )


def sparse_subset(
    frame: pd.DataFrame,
    responses_per_prompt: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for _, indices in frame.groupby("prompt_id", sort=True).indices.items():
        values = np.asarray(indices, dtype=np.int64)
        count = min(responses_per_prompt, len(values))
        selected.append(rng.choice(values, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def train_stage(
    model: AttributionModel,
    tokenizer,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    stage: int,
    epochs: int,
    learning_rate: float,
    hours: float,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    output_dir: Path,
    sparse_seed: int,
) -> dict[str, object]:
    train_loader = DataLoader(
        TextRows(
            train_frame["text"].tolist(),
            train_frame["label_index"].to_numpy(dtype=np.int64),
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collator(tokenizer, max_length),
        num_workers=0,
    )
    valid_loader = None
    sparse_loader = None
    if valid_frame is not None:
        valid_loader = DataLoader(
            TextRows(
                valid_frame["text"].tolist(),
                valid_frame["label_index"].to_numpy(dtype=np.int64),
            ),
            batch_size=max(1, batch_size * 2),
            shuffle=False,
            collate_fn=make_collator(tokenizer, max_length),
            num_workers=0,
        )
        sparse_indices = sparse_subset(valid_frame, 2, sparse_seed)
        sparse_frame = valid_frame.iloc[sparse_indices]
        sparse_loader = DataLoader(
            TextRows(
                sparse_frame["text"].tolist(),
                sparse_frame["label_index"].to_numpy(dtype=np.int64),
            ),
            batch_size=max(1, batch_size * 2),
            shuffle=False,
            collate_fn=make_collator(tokenizer, max_length),
            num_workers=0,
        )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=0.02,
    )
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = updates_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=max(1, total_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=1024,
        growth_interval=2000,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    deadline = time.monotonic() + hours * 3600
    completed = 0
    metrics: dict[str, object] = {"epochs": []}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.time()
        stop = False
        for batch_index, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to("cuda")
            batch = {name: value.to("cuda") for name, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**batch)
                loss = criterion(logits, labels) / grad_accum
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * grad_accum)
            if batch_index % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                stepped = scaler.get_scale() >= scale_before
                if stepped:
                    scheduler.step()
                    completed += 1
                if completed % 10 == 0 or completed == 1:
                    print(
                        f"stage={stage} epoch={epoch} step={completed} "
                        f"loss={np.mean(losses[-grad_accum:]):.4f} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
            if (max_steps > 0 and completed >= max_steps) or (
                time.monotonic() >= deadline
            ):
                stop = True
                break
        epoch_result: dict[str, object] = {
            "epoch": epoch,
            "completed_steps": completed,
            "mean_loss": float(np.mean(losses)) if losses else None,
        }
        if valid_loader is not None:
            full_score, targets, predictions, probabilities = evaluate(
                model,
                valid_loader,
            )
            sparse_score, sparse_targets, sparse_predictions, _ = evaluate(
                model,
                sparse_loader,
            )
            epoch_result.update(
                {
                    "valid_balanced_accuracy": full_score,
                    "sparse2_balanced_accuracy": sparse_score,
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                output_dir / f"stage{stage}_epoch{epoch}_probabilities.npy",
                probabilities,
            )
            np.save(
                output_dir / f"stage{stage}_epoch{epoch}_targets.npy",
                targets,
            )
            np.save(
                output_dir / f"stage{stage}_epoch{epoch}_predictions.npy",
                predictions,
            )
            np.save(
                output_dir / f"stage{stage}_epoch{epoch}_sparse_targets.npy",
                sparse_targets,
            )
            np.save(
                output_dir / f"stage{stage}_epoch{epoch}_sparse_predictions.npy",
                sparse_predictions,
            )
            print(
                f"stage={stage} epoch={epoch} "
                f"valid={full_score:.4f} sparse2={sparse_score:.4f}",
                flush=True,
            )
        metrics["epochs"].append(epoch_result)
        if stop:
            print(f"stage={stage} reached step/time budget", flush=True)
            break
    return metrics


def save_checkpoint(
    model: AttributionModel,
    tokenizer,
    output_dir: Path,
    selected_layers: list[int],
    num_layers: int,
    max_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    torch.save(model.head.state_dict(), output_dir / "head.pt")
    config = {
        "labels": LABELS,
        "selected_layers": selected_layers,
        "num_layers": num_layers,
        "max_length": max_length,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--selected-layers", default="2,4,6,8,12,16,20,24")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=2)
    parser.add_argument("--stage1-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--stage1-hours", type=float, default=3.0)
    parser.add_argument("--stage2-hours", type=float, default=2.0)
    parser.add_argument("--stage1-per-label", type=int, default=500)
    parser.add_argument("--stage1-max-steps", type=int, default=0)
    parser.add_argument("--stage2-max-steps", type=int, default=0)
    parser.add_argument("--sparse-seed", type=int, default=20261116)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    torch.cuda.empty_cache()
    gc.collect()
    print(
        f"gpu={torch.cuda.get_device_name(0)} "
        f"memory_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}",
        flush=True,
    )

    train = pd.read_csv(args.train_csv)
    train["text"] = train["text"].fillna("").astype(str)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    train["label_index"] = train["model"].map(label_to_index).astype(int)
    pool = pd.read_parquet(args.pool)
    pool["text"] = pool["text"].fillna("").astype(str)
    pool["label_index"] = pool["model"].map(label_to_index).astype(int)
    if args.stage1_per_label > 0:
        pool = (
            pool.groupby("model", group_keys=False, sort=True)
            .head(args.stage1_per_label)
            .reset_index(drop=True)
        )
    if args.smoke:
        pool = pool.groupby("model", group_keys=False).head(1).reset_index(
            drop=True
        )
        train = train.groupby("prompt_id", group_keys=False).head(1).reset_index(
            drop=True
        )
        args.stage1_max_steps = args.stage1_max_steps or 4
        args.stage2_max_steps = args.stage2_max_steps or 4
        args.stage1_epochs = 1
        args.stage2_epochs = 1
    folds = list(
        GroupKFold(5).split(
            train,
            train["label_index"],
            train["prompt_id"],
        )
    )
    train_index, valid_index = folds[args.fold]
    stage2_train = train.iloc[train_index].reset_index(drop=True)
    stage2_valid = train.iloc[valid_index].reset_index(drop=True)
    if args.smoke:
        stage2_train = stage2_train.groupby("prompt_id", group_keys=False).head(
            1
        ).reset_index(drop=True)
        stage2_valid = stage2_valid.groupby("prompt_id", group_keys=False).head(
            1
        ).reset_index(drop=True)
    selected_layers = [
        int(value) for value in args.selected_layers.split(",")
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"pool={len(pool)} stage2_train={len(stage2_train)} "
        f"stage2_valid={len(stage2_valid)} fold={args.fold}",
        flush=True,
    )
    model, tokenizer = build_model(
        args.model_dir,
        selected_layers,
        args.num_layers,
        not args.no_4bit,
        args.lora_r,
    )
    stage1_metrics = train_stage(
        model,
        tokenizer,
        pool,
        None,
        1,
        args.stage1_epochs,
        args.stage1_lr,
        args.stage1_hours,
        args.max_length,
        args.batch_size,
        args.grad_accum,
        args.stage1_max_steps,
        output_dir,
        args.sparse_seed,
    )
    save_checkpoint(
        model,
        tokenizer,
        output_dir / "stage1",
        selected_layers,
        args.num_layers,
        args.max_length,
    )
    stage2_metrics = train_stage(
        model,
        tokenizer,
        stage2_train,
        stage2_valid,
        2,
        args.stage2_epochs,
        args.stage2_lr,
        args.stage2_hours,
        args.max_length,
        args.batch_size,
        args.grad_accum,
        args.stage2_max_steps,
        output_dir,
        args.sparse_seed,
    )
    save_checkpoint(
        model,
        tokenizer,
        output_dir / "stage2",
        selected_layers,
        args.num_layers,
        args.max_length,
    )
    metrics = {
        "fold": args.fold,
        "pool_rows": len(pool),
        "stage2_train_rows": len(stage2_train),
        "stage2_valid_rows": len(stage2_valid),
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
