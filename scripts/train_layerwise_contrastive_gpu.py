from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.constants import (  # noqa: E402
    GROUP_BY_FAMILY,
    GROUPS,
    LABELS,
    TRAIN_CSV,
    family,
)
from train_hierarchical_vendor_router_gpu import (  # noqa: E402
    FAMILIES,
    balanced_accuracy,
)
from train_layerwise_token_hierarchy_gpu import (  # noqa: E402
    LayerwiseHierarchyEncoder,
)
from train_token_hierarchy_encoder_gpu import (  # noqa: E402
    SequenceDataset,
    bounds_by_row,
    class_weights,
    collate,
    evaluate,
)


PAIRS = (
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("gpt-5.6-luna", "gpt-5.6-terra"),
    ("kimi-k3", "gpt-5.5"),
    ("claude-fable-5", "glm-5.3-flash"),
)
VENDOR_FOR_CLASS = np.asarray(
    [FAMILIES.index(family(label)) for label in LABELS],
    dtype=np.int64,
)


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = F.normalize(embeddings, dim=-1)
    logits = values @ values.T / temperature
    logits = logits.masked_fill(
        torch.eye(len(values), device=values.device, dtype=torch.bool),
        -1e4,
    )
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    positive_count = same.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive = (
        log_prob.masked_fill(~same, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -positive[valid].mean()


class ContrastiveHierarchy(torch.nn.Module):
    def __init__(
        self,
        encoder: LayerwiseHierarchyEncoder,
        projection_dim: int,
        pair_heads: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(encoder.output_norm.normalized_shape[0], 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, projection_dim),
        )
        self.pair_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    encoder.output_norm.normalized_shape[0],
                    2,
                )
                for _ in range(pair_heads)
            ]
        )

    def forward(
        self,
        layer8: torch.Tensor,
        layer36: torch.Tensor,
        qkv: torch.Tensor,
        mask: torch.Tensor,
    ):
        pooled = self.encoder.encode(
            layer8,
            layer36,
            qkv,
            mask,
        )
        group = self.encoder.group3(pooled)
        vendor = self.encoder.vendor(pooled)
        expert = self.encoder.expert(pooled)
        final = (
            expert
            + self.encoder.vendor_scale.clamp(0.25, 4.0)
            * vendor[:, self.encoder.vendor_for_class]
        )
        pairs = torch.stack(
            [head(pooled) for head in self.pair_heads],
            dim=1,
        )
        return final, vendor, group, expert, pairs, pooled


def balanced_batches(
    train_rows: np.ndarray,
    labels: np.ndarray,
    classes_per_batch: int,
    samples_per_class: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = {
        label: np.flatnonzero(
            (labels[train_rows] == label)
        )
        for label in np.unique(labels[train_rows])
    }
    eligible = [
        label for label, rows in by_class.items() if len(rows) >= 1
    ]
    batches = []
    for _ in range(max(1, len(train_rows) // (
        classes_per_batch * samples_per_class
    ))):
        chosen = rng.choice(
            eligible,
            size=min(classes_per_batch, len(eligible)),
            replace=False,
        )
        rows = []
        for label in chosen:
            candidates = by_class[label]
            replace = len(candidates) < samples_per_class
            local = rng.choice(
                candidates,
                size=samples_per_class,
                replace=replace,
            )
            rows.extend(train_rows[local].tolist())
        batches.append(np.asarray(rows, dtype=np.int64))
    return batches


def pair_loss(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for pair_index, (left_name, right_name) in enumerate(PAIRS):
        left = LABELS.index(left_name)
        right = LABELS.index(right_name)
        mask = (labels == left) | (labels == right)
        if not mask.any():
            continue
        target = (labels[mask] == left).long()
        selected = pair_logits[mask, pair_index]
        losses.append(
            F.cross_entropy(selected, target)
        )
    if not losses:
        return pair_logits.sum() * 0.0
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--vendor-weight", type=float, default=0.25)
    parser.add_argument("--group-weight", type=float, default=0.15)
    parser.add_argument("--expert-weight", type=float, default=0.25)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--supcon-weight", type=float, default=0.20)
    parser.add_argument("--seed-base", type=int, default=20261122)
    parser.add_argument(
        "--qkv-file",
        default=(
            "features/qwen3_4b_qkv_token_layers_"
            "2_3_20_23_35.npz"
        ),
    )
    parser.add_argument(
        "--hidden8-dir",
        default="features/qwen3_base_hidden_layers_8_36/layer8",
    )
    parser.add_argument(
        "--hidden36-dir",
        default="features/qwen3_base_hidden_layers_8_36/layer36",
    )
    parser.add_argument(
        "--output",
        default="validation/layerwise_contrastive_fold0",
    )
    args = parser.parse_args()
    if args.d_model % args.heads != 0:
        raise ValueError("d_model must be divisible by heads")
    if (args.d_model // 2) % args.heads != 0:
        raise ValueError("d_model // 2 must be divisible by heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    frame = pd.read_csv(TRAIN_CSV)
    labels = frame["model"].map(
        {label: index for index, label in enumerate(LABELS)}
    ).to_numpy(np.int64)
    prompts = frame["prompt_id"].astype(str).to_numpy()
    vendor_labels = np.asarray(
        [FAMILIES.index(family(LABELS[index])) for index in labels],
        dtype=np.int64,
    )
    group3_labels = np.asarray(
        [
            GROUPS.index(GROUP_BY_FAMILY[family(LABELS[index])])
            for index in labels
        ],
        dtype=np.int64,
    )
    metadata = np.load(
        ROOT / args.hidden36_dir / "metadata.npz",
        allow_pickle=True,
    )
    qkv_data = np.load(ROOT / args.qkv_file, allow_pickle=True)
    layer8 = np.load(
        ROOT / args.hidden8_dir / "sequences.npy",
        mmap_mode="r",
    )
    layer36 = np.load(
        ROOT / args.hidden36_dir / "sequences.npy",
        mmap_mode="r",
    )
    hidden_bounds = bounds_by_row(
        metadata["offsets"].astype(np.int64),
        metadata["row_indices"].astype(np.int64),
        len(labels),
    )
    qkv_bounds = bounds_by_row(
        qkv_data["offsets"].astype(np.int64),
        qkv_data["row_indices"].astype(np.int64),
        len(labels),
    )
    qkv = qkv_data["sequences"]
    folds = list(
        GroupKFold(5).split(
            np.zeros((len(labels), 1), dtype=np.float32),
            labels,
            prompts,
        )
    )
    train_rows, valid_rows = folds[args.fold]
    torch.manual_seed(args.seed_base + args.fold)
    torch.cuda.manual_seed_all(args.seed_base + args.fold)
    train_dataset = SequenceDataset(
        layer8,
        layer36,
        qkv,
        hidden_bounds,
        qkv_bounds,
        labels,
        vendor_labels,
        group3_labels,
        np.arange(len(labels)),
        384,
    )
    valid_dataset = SequenceDataset(
        layer8,
        layer36,
        qkv,
        hidden_bounds,
        qkv_bounds,
        labels,
        vendor_labels,
        group3_labels,
        valid_rows,
        384,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    encoder = LayerwiseHierarchyEncoder(
        layer8.shape[1],
        qkv.shape[1],
        args.d_model,
        args.heads,
        args.dropout,
        384,
    ).to("cuda")
    model = ContrastiveHierarchy(
        encoder,
        args.projection_dim,
        len(PAIRS),
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    class_weight = class_weights(
        labels,
        train_rows,
        len(LABELS),
    )
    vendor_weight = class_weights(
        vendor_labels,
        train_rows,
        len(FAMILIES),
    )
    group_weight = class_weights(
        group3_labels,
        train_rows,
        len(GROUPS),
    )
    batches = balanced_batches(
        train_rows,
        labels,
        max(1, args.batch_size // 2),
        2,
        args.seed_base + args.fold,
    )
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch_rows in batches:
            # SequenceDataset indexes into rows, so use local row positions.
            train_indices = np.asarray(
                [
                    int(np.flatnonzero(train_rows == row)[0])
                    for row in batch_rows
                ],
                dtype=np.int64,
            )
            items = [
                train_dataset[int(index)]
                for index in train_indices
            ]
            (
                layer8_batch,
                layer36_batch,
                qkv_batch,
                mask,
                target,
                target_vendor,
                target_group,
            ) = collate(items)
            final, vendor, group, expert, pairs, pooled = model(
                layer8_batch.to("cuda"),
                layer36_batch.to("cuda"),
                qkv_batch.to("cuda"),
                mask.to("cuda"),
            )
            target = target.to("cuda")
            target_vendor = target_vendor.to("cuda")
            target_group = target_group.to("cuda")
            loss = F.cross_entropy(
                final,
                target,
                weight=class_weight,
                label_smoothing=args.label_smoothing,
            )
            loss += args.vendor_weight * F.cross_entropy(
                vendor,
                target_vendor,
                weight=vendor_weight,
            )
            loss += args.group_weight * F.cross_entropy(
                group,
                target_group,
                weight=group_weight,
            )
            loss += args.expert_weight * F.cross_entropy(
                expert,
                target,
                weight=class_weight,
            )
            loss += args.pair_weight * pair_loss(pairs, target)
            loss += args.supcon_weight * supcon_loss(
                model.projection(pooled),
                target,
                args.temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
        print(
            f"epoch={epoch + 1} loss={total / (len(batches) * args.batch_size):.4f}",
            flush=True,
        )
    final_probability, vendor_probability, group_probability = (
        evaluate(model.encoder, valid_loader)
    )
    metrics = {
        "fold": args.fold,
        "balanced_accuracy": balanced_accuracy(
            labels[valid_rows],
            final_probability.argmax(axis=1),
            len(LABELS),
        ),
        "vendor_balanced_accuracy": balanced_accuracy(
            vendor_labels[valid_rows],
            vendor_probability.argmax(axis=1),
            len(FAMILIES),
        ),
        "group3_balanced_accuracy": balanced_accuracy(
            group3_labels[valid_rows],
            group_probability.argmax(axis=1),
            len(GROUPS),
        ),
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "valid_probabilities.npy", final_probability)
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "config": vars(args),
            "fold": args.fold,
            "labels": LABELS,
        },
        output / "model_state.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
