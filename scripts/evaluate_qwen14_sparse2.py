from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(f"Unexpected probability shape {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Probability rows must have positive mass")
    return values / totals


def score(targets: np.ndarray, predictions: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(targets, predictions))


def sample_two(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for prompt, rows in frame.groupby("prompt_id", sort=True).indices.items():
        del prompt
        rows = np.asarray(rows, dtype=np.int64)
        selected.append(
            rng.choice(rows, size=min(2, len(rows)), replace=False)
        )
    return np.sort(np.concatenate(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--targets")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20261116)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument(
        "--weights",
        default="0.05,0.10,0.15,0.20,0.30,0.50",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-seed-output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.train_csv)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["model"].map(label_to_index).to_numpy(dtype=np.int64)
    folds = list(
        GroupKFold(args.outer_folds).split(
            frame,
            labels,
            frame["prompt_id"],
        )
    )
    _, valid_index = folds[args.fold]
    valid_frame = frame.iloc[valid_index].reset_index(drop=True)
    qwen = normalize(np.load(args.probabilities))
    if qwen.shape != (len(valid_index), len(LABELS)):
        raise ValueError(
            f"Qwen probabilities shape {qwen.shape}, "
            f"expected {(len(valid_index), len(LABELS))}"
        )
    if args.targets:
        targets = np.load(args.targets).astype(np.int64)
        expected = labels[valid_index]
        if not np.array_equal(targets, expected):
            raise ValueError("Qwen target array does not match fold rows")

    baseline_all = normalize(np.load(args.baseline))
    if len(baseline_all) != len(frame):
        raise ValueError("Baseline must cover all training rows")
    baseline = baseline_all[valid_index]
    valid_labels = labels[valid_index]
    weights = [float(value) for value in args.weights.split(",") if value]

    seed_rows = []
    variants = {
        **{f"qwen_w{weight:g}": weight for weight in weights},
        "qwen_only": None,
    }
    scores = {
        "baseline": [],
        **{name: [] for name in variants},
    }
    bootstrap_delta = {name: [] for name in variants}
    for seed_offset in range(args.seeds):
        seed = args.seed_start + seed_offset
        rng = np.random.default_rng(seed)
        sampled = sample_two(valid_frame, rng)
        sampled_labels = valid_labels[sampled]
        sampled_prompts = valid_frame["prompt_id"].to_numpy()[sampled]
        baseline_probabilities = baseline[sampled]
        qwen_probabilities = qwen[sampled]
        baseline_predictions = baseline_probabilities.argmax(axis=1)
        scores["baseline"].append(
            score(sampled_labels, baseline_predictions)
        )
        predictions_by_variant = {}
        for name, weight in variants.items():
            if weight is None:
                probabilities = qwen_probabilities
            else:
                probabilities = np.log(
                    np.clip(baseline_probabilities, 1e-12, None)
                ) + weight * np.log(
                    np.clip(qwen_probabilities, 1e-12, None)
                )
            predictions = probabilities.argmax(axis=1)
            predictions_by_variant[name] = predictions
            scores[name].append(score(sampled_labels, predictions))

        unique_prompts = np.unique(sampled_prompts)
        prompt_rows = {
            prompt: np.flatnonzero(sampled_prompts == prompt)
            for prompt in unique_prompts
        }
        for name, predictions in predictions_by_variant.items():
            deltas = []
            for _ in range(args.bootstrap_replicates):
                chosen = rng.choice(
                    unique_prompts,
                    size=len(unique_prompts),
                    replace=True,
                )
                rows = np.concatenate(
                    [prompt_rows[prompt] for prompt in chosen]
                )
                deltas.append(
                    score(
                        sampled_labels[rows],
                        predictions[rows],
                    )
                    - score(
                        sampled_labels[rows],
                        baseline_predictions[rows],
                    )
                )
            bootstrap_delta[name].append(np.asarray(deltas))

        for candidate, values in scores.items():
            seed_rows.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "balanced_accuracy": values[-1],
                }
            )

    baseline_scores = np.asarray(scores["baseline"])
    report: dict[str, object] = {
        "fold": args.fold,
        "valid_rows": int(len(valid_index)),
        "seeds": args.seeds,
        "responses_per_prompt": 2,
        "baseline": {
            "path": args.baseline,
            "mean": float(baseline_scores.mean()),
            "std": float(baseline_scores.std(ddof=1)),
        },
        "candidates": {},
    }
    for name in variants:
        values = np.asarray(scores[name])
        deltas = values - baseline_scores
        bootstrap_values = np.stack(bootstrap_delta[name])
        bootstrap_means = bootstrap_values.mean(axis=0)
        report["candidates"][name] = {
            "score": {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "p05": float(np.quantile(values, 0.05)),
            },
            "paired_delta": {
                "mean": float(deltas.mean()),
                "std": float(deltas.std(ddof=1)),
                "p05": float(np.quantile(deltas, 0.05)),
                "win_rate": float(np.mean(deltas > 0)),
            },
            "bootstrap_delta": {
                "mean": float(bootstrap_means.mean()),
                "p025": float(np.quantile(bootstrap_means, 0.025)),
                "p975": float(np.quantile(bootstrap_means, 0.975)),
            },
        }
    best_name = max(
        variants,
        key=lambda name: report["candidates"][name]["score"]["mean"],
    )
    best = report["candidates"][best_name]
    report["best_variant"] = best_name
    report["reader_gate"] = {
        "new_reader_passed": bool(
            best["paired_delta"]["mean"] >= 0.004
            and best["paired_delta"]["win_rate"] >= 0.80
            and best["bootstrap_delta"]["p025"] > 0
        ),
        "fallback_passed": bool(
            best["paired_delta"]["mean"] >= 0.003
            and best["paired_delta"]["win_rate"] >= 0.80
            and best["bootstrap_delta"]["p025"] > 0
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_seed = Path(args.per_seed_output)
    per_seed.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(per_seed, index=False)
    print(json.dumps(report, indent=2))
    print(f"saved {output}")
    print(f"saved {per_seed}")


if __name__ == "__main__":
    main()
