from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


EXCLUDED = (
    "gpt-5.6-terra",
    "claude-fable-5",
    "gpt-5.6-luna",
)


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


def parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value)
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(
            "Values must be non-negative comma-separated numbers"
        )
    return values


def top_two_margin(values: np.ndarray) -> np.ndarray:
    top_two = np.partition(values, -2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search label-independent confidence and parent-margin "
            "gates for a contrastive token ensemble."
        )
    )
    parser.add_argument(
        "--base",
        default="features/qwen3_embedding_stack_blend_oof.npy",
    )
    parser.add_argument(
        "--ensemble",
        default=(
            "validation/layerwise_contrastive_seed2_5fold/"
            "seed_ensemble_oof.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="validation/layerwise_contrastive_confidence_gate",
    )
    parser.add_argument(
        "--weights",
        type=parse_floats,
        default=(0.18, 0.20, 0.22, 0.24),
    )
    parser.add_argument(
        "--token-confidences",
        type=parse_floats,
        default=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    )
    parser.add_argument(
        "--parent-margins",
        type=parse_floats,
        default=(0.05, 0.10, 0.15, 0.25),
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {
        label: index for index, label in enumerate(LABELS)
    }
    targets = frame["model"].map(label_to_index).to_numpy(np.int64)
    base = normalize_rows(np.load(ROOT / args.base))
    ensemble = normalize_rows(np.load(ROOT / args.ensemble))
    if base.shape != ensemble.shape or len(base) != len(frame):
        raise ValueError("Parent and ensemble probability shapes do not match")

    base_prediction = base.argmax(axis=1)
    token_prediction = ensemble.argmax(axis=1)
    token_confidence = ensemble.max(axis=1)
    parent_margin = top_two_margin(base)
    excluded_indices = {
        LABELS.index(label) for label in EXCLUDED
    }
    eligible = np.asarray(
        [
            prediction not in excluded_indices
            for prediction in base_prediction
        ],
        dtype=bool,
    )
    same_prediction = token_prediction == base_prediction
    changed_prediction = ~same_prediction

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidates = output / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    base_score = float(
        balanced_accuracy_score(targets, base_prediction)
    )
    records = []

    for weight in args.weights:
        modes = [
            ("parent", None, None),
            ("changed", None, None),
        ]
        confidence_modes = [
            (mode, threshold, margin)
            for mode in ("confidence", "safe_change")
            for threshold in args.token_confidences
            for margin in args.parent_margins
        ]
        configurations = modes + confidence_modes
        for mode, threshold, margin in configurations:
            if mode == "parent":
                active = eligible
                name = f"w{weight:.2f}_parent"
                confidence_gate = None
                margin_gate = None
            elif mode == "changed":
                active = eligible & changed_prediction
                name = f"w{weight:.2f}_changed"
                confidence_gate = None
                margin_gate = None
            else:
                confidence_ok = token_confidence >= threshold
                margin_ok = parent_margin <= margin
                allow_change = confidence_ok & margin_ok
                active = eligible & (
                    allow_change
                    if mode == "confidence"
                    else same_prediction | allow_change
                )
                name = (
                    f"w{weight:.2f}_{mode}"
                    f"_c{int(threshold * 100):02d}"
                    f"_m{int(margin * 100):02d}"
                )
                confidence_gate = threshold
                margin_gate = margin
            candidate = base.copy()
            if active.any():
                candidate[active] = normalize_rows(
                    (1.0 - weight) * base[active]
                    + weight * ensemble[active]
                )
            prediction = candidate.argmax(axis=1)
            score = float(
                balanced_accuracy_score(targets, prediction)
            )
            records.append(
                {
                    "name": name,
                    "mode": mode,
                    "weight": weight,
                    "token_confidence": confidence_gate,
                    "parent_margin": margin_gate,
                    "active_rows": int(active.sum()),
                    "changed_predictions": int(
                        np.sum(prediction != base_prediction)
                    ),
                    "balanced_accuracy": score,
                    "delta": score - base_score,
                    "gains": int(
                        np.sum(
                            (base_prediction != targets)
                            & (prediction == targets)
                        )
                    ),
                    "losses": int(
                        np.sum(
                            (base_prediction == targets)
                            & (prediction != targets)
                        )
                    ),
                    "_candidate": candidate,
                }
            )

    ranked = sorted(
        records,
        key=lambda record: record["balanced_accuracy"],
        reverse=True,
    )
    for record in ranked[: args.top_k]:
        candidate = record.pop("_candidate")
        np.save(candidates / f"{record['name']}.npy", candidate)
    for record in ranked[args.top_k :]:
        record.pop("_candidate")

    columns = [
        "name",
        "mode",
        "weight",
        "token_confidence",
        "parent_margin",
        "active_rows",
        "changed_predictions",
        "balanced_accuracy",
        "delta",
        "gains",
        "losses",
    ]
    table = pd.DataFrame(ranked)[columns]
    table.to_csv(output / "confidence_gate_grid.csv", index=False)
    summary = {
        "base": args.base,
        "ensemble": args.ensemble,
        "base_balanced_accuracy": base_score,
        "excluded_parent_predictions": list(EXCLUDED),
        "configurations": len(records),
        "saved_candidates": min(args.top_k, len(records)),
        "top_candidates": table.head(20).to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
