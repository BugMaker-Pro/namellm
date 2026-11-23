from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from train_hierarchical_vendor_router_gpu import (  # noqa: E402
    balanced_accuracy,
)


DEFAULT_SEEDS = (
    "validation/layerwise_contrastive_fold{fold}",
    "validation/layerwise_contrastive_seed2_fold{fold}",
    "validation/layerwise_contrastive_seed3_fold{fold}",
)
DEFAULT_EXCLUDED = ("gpt-5.6-terra", "claude-fable-5")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(
            f"Expected probabilities shape (n, {len(LABELS)}), "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every probability row must have positive mass")
    return values / totals


def parse_weights(raw: str) -> tuple[float, ...]:
    weights = tuple(float(value) for value in raw.split(",") if value)
    if not weights or any(value <= 0 or value >= 1 for value in weights):
        raise argparse.ArgumentTypeError(
            "Weights must be comma-separated values between 0 and 1"
        )
    return weights


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(value for value in raw.split(",") if value)
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown labels: {', '.join(unknown)}"
        )
    return labels


def parse_seed_weights(raw: str) -> tuple[float, ...]:
    weights = tuple(float(value) for value in raw.split(",") if value)
    if not weights or any(value < 0 for value in weights):
        raise argparse.ArgumentTypeError(
            "Seed weights must be non-negative comma-separated values"
        )
    if sum(weights) <= 0:
        raise argparse.ArgumentTypeError(
            "At least one seed weight must be positive"
        )
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge layerwise contrastive seed OOF by GroupKFold, form a "
            "seed ensemble, and evaluate label-gated parent blends."
        )
    )
    parser.add_argument(
        "--seed-template",
        action="append",
        default=[],
        help=(
            "Fold directory template containing {fold}. Repeat for each "
            "model seed."
        ),
    )
    parser.add_argument(
        "--base",
        default="features/qwen3_embedding_stack_blend_oof.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/layerwise_contrastive_seed3_5fold",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=(0.14, 0.18, 0.20, 0.22),
    )
    parser.add_argument(
        "--exclude",
        type=parse_labels,
        default=DEFAULT_EXCLUDED,
        help="Do not apply the token ensemble when parent top-1 is in this set.",
    )
    parser.add_argument(
        "--seed-weights",
        type=parse_seed_weights,
        help="Optional non-negative seed weights; normalized to sum to one.",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=("mean", "log_mean"),
        default="mean",
    )
    args = parser.parse_args()

    templates = tuple(args.seed_template) or DEFAULT_SEEDS
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    blends = output / "blends"
    blends.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {
        label: index for index, label in enumerate(LABELS)
    }
    targets = frame["model"].map(label_to_index).to_numpy(np.int64)
    if np.isnan(targets).any():
        raise ValueError("Training data contains an unknown label")
    prompts = frame["prompt_id"].astype(str).to_numpy()
    folds = list(
        GroupKFold(5).split(
            np.zeros((len(frame), 1), dtype=np.float32),
            targets,
            prompts,
        )
    )

    seed_probabilities = []
    seed_summaries = []
    for seed_index, template in enumerate(templates):
        oof = np.full(
            (len(frame), len(LABELS)),
            np.nan,
            dtype=np.float64,
        )
        fold_metrics = []
        for fold, (_, valid_rows) in enumerate(folds):
            fold_dir = ROOT / template.format(fold=fold)
            probability_path = fold_dir / "valid_probabilities.npy"
            probabilities = normalize_rows(
                np.load(probability_path)
            )
            if len(probabilities) != len(valid_rows):
                raise ValueError(
                    f"{probability_path} has {len(probabilities)} rows, "
                    f"expected {len(valid_rows)}"
                )
            oof[valid_rows] = probabilities
            metrics_path = fold_dir / "metrics.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {}
            )
            fold_metrics.append(
                {
                    "fold": fold,
                    "rows": len(valid_rows),
                    "standalone_balanced_accuracy": balanced_accuracy(
                        targets[valid_rows],
                        probabilities.argmax(axis=1),
                        len(LABELS),
                    ),
                    "reported_balanced_accuracy": metrics.get(
                        "balanced_accuracy"
                    ),
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError(
                f"Seed {seed_index} did not cover every OOF row"
            )
        score = balanced_accuracy(
            targets,
            oof.argmax(axis=1),
            len(LABELS),
        )
        seed_probabilities.append(oof)
        np.save(output / f"seed{seed_index + 1}_oof.npy", oof)
        seed_summaries.append(
            {
                "name": f"seed{seed_index + 1}",
                "template": template,
                "balanced_accuracy": score,
                "fold_metrics": fold_metrics,
            }
        )

    if args.seed_weights is None:
        seed_weights = np.full(
            len(seed_probabilities),
            1.0 / len(seed_probabilities),
            dtype=np.float64,
        )
    else:
        if len(args.seed_weights) != len(seed_probabilities):
            raise ValueError(
                f"Received {len(args.seed_weights)} seed weights for "
                f"{len(seed_probabilities)} seeds"
            )
        seed_weights = np.asarray(args.seed_weights, dtype=np.float64)
        seed_weights /= seed_weights.sum()
    stacked = np.stack(seed_probabilities)
    if args.ensemble_method == "mean":
        ensemble = normalize_rows(
            np.tensordot(seed_weights, stacked, axes=(0, 0))
        )
    else:
        log_probabilities = np.log(
            np.clip(stacked, np.finfo(np.float64).tiny, None)
        )
        ensemble = normalize_rows(
            np.exp(
                np.tensordot(
                    seed_weights,
                    log_probabilities,
                    axes=(0, 0),
                )
            )
        )
    np.save(output / "oof_probabilities.npy", ensemble)
    ensemble_score = balanced_accuracy(
        targets,
        ensemble.argmax(axis=1),
        len(LABELS),
    )

    base = normalize_rows(np.load(ROOT / args.base))
    if base.shape != ensemble.shape:
        raise ValueError(
            f"Parent shape {base.shape} does not match {ensemble.shape}"
        )
    excluded_indices = {
        LABELS.index(label) for label in args.exclude
    }
    active = np.asarray(
        [
            prediction not in excluded_indices
            for prediction in base.argmax(axis=1)
        ],
        dtype=bool,
    )
    base_predictions = base.argmax(axis=1)
    base_score = balanced_accuracy(
        targets,
        base_predictions,
        len(LABELS),
    )

    weight_results = []
    per_class_records = []
    per_parent_class_records = []
    for weight in args.weights:
        candidate = base.copy()
        candidate[active] = normalize_rows(
            (1.0 - weight) * base[active]
            + weight * ensemble[active]
        )
        candidate_predictions = candidate.argmax(axis=1)
        score = balanced_accuracy(
            targets,
            candidate_predictions,
            len(LABELS),
        )
        name = f"gated_w{weight:.2f}"
        np.save(blends / f"{name}.npy", candidate)
        class_records = []
        parent_class_records = []
        for label_index, label in enumerate(LABELS):
            rows = targets == label_index
            base_correct = (
                base_predictions[rows] == label_index
            )
            candidate_correct = (
                candidate_predictions[rows] == label_index
            )
            gains = int(np.sum(~base_correct & candidate_correct))
            losses = int(np.sum(base_correct & ~candidate_correct))
            record = {
                "weight": weight,
                "label": label,
                "rows": int(rows.sum()),
                "base_correct": int(base_correct.sum()),
                "candidate_correct": int(candidate_correct.sum()),
                "gains": gains,
                "losses": losses,
                "net": gains - losses,
                "active_rows": int(active[rows].sum()),
            }
            class_records.append(record)
            per_class_records.append(record)
        for label_index, label in enumerate(LABELS):
            rows = base_predictions == label_index
            base_correct = (
                base_predictions[rows] == targets[rows]
            )
            candidate_correct = (
                candidate_predictions[rows] == targets[rows]
            )
            gains = int(np.sum(~base_correct & candidate_correct))
            losses = int(np.sum(base_correct & ~candidate_correct))
            record = {
                "weight": weight,
                "parent_label": label,
                "rows": int(rows.sum()),
                "base_correct": int(base_correct.sum()),
                "candidate_correct": int(candidate_correct.sum()),
                "gains": gains,
                "losses": losses,
                "net": gains - losses,
                "active_rows": int(active[rows].sum()),
            }
            parent_class_records.append(record)
            per_parent_class_records.append(record)
        weight_results.append(
            {
                "name": name,
                "weight": weight,
                "balanced_accuracy": score,
                "delta": score - base_score,
                "active_rows": int(active.sum()),
                "changed_predictions": int(
                    np.sum(candidate_predictions != base_predictions)
                ),
                "class_deltas": {
                    row["label"]: row["net"]
                    for row in class_records
                    if row["net"] != 0
                },
                "parent_class_deltas": {
                    row["parent_label"]: row["net"]
                    for row in parent_class_records
                    if row["net"] != 0
                },
            }
        )

    summary = {
        "train_rows": len(frame),
        "labels": len(LABELS),
        "base": args.base,
        "base_balanced_accuracy": base_score,
        "excluded_parent_predictions": list(args.exclude),
        "active_rows": int(active.sum()),
        "seeds": seed_summaries,
        "seed_weights": seed_weights.tolist(),
        "ensemble_method": args.ensemble_method,
        "seed_ensemble_balanced_accuracy": ensemble_score,
        "weights": weight_results,
    }
    (output / "seed_ensemble_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(per_class_records).to_csv(
        output / "seed_ensemble_class_deltas.csv",
        index=False,
    )
    pd.DataFrame(per_parent_class_records).to_csv(
        output / "seed_ensemble_parent_class_deltas.csv",
        index=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
