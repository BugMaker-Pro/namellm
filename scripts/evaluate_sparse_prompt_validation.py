from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


DEFAULT_BRANCHES = {
    "compact": "features/compact_baseline_oof.npy",
    "base": "features/summary_funnel_power_blend.npy",
    "e512": "features/qwen3_embedding_4b_oof.npy",
    "e256": "features/qwen3_embedding_4b_pca256_oof.npy",
    "layer4": "features/qwen3_embedding_4b_layers4_oof.npy",
    "gru7": "features/qwen3_embedding_4b_layer_sequence7_oof.npy",
    "fusion": "features/qwen3_embedding_fusion7_oof.npy",
    "cluster_fusion": (
        "features/qwen3_embedding_cluster_balanced_fusion_oof.npy"
    ),
    "rich_stack": "features/qwen3_embedding_rich_stack_balanced_oof.npy",
    "stack_blend": "features/qwen3_embedding_stack_blend_oof.npy",
}


def normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(
            f"Expected probabilities with shape (n, {len(LABELS)}), "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Probability rows must have positive mass")
    return values / totals


def center_probabilities(
    probabilities: np.ndarray,
    clusters: np.ndarray,
    alpha: float,
) -> np.ndarray:
    normalized = normalize_rows(probabilities)
    logs = np.log(np.clip(normalized, 1e-12, None))
    centered = logs.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= alpha * logs[mask].mean(
            axis=0,
            keepdims=True,
        )
    centered -= centered.max(axis=1, keepdims=True)
    result = np.exp(centered)
    return result / result.sum(axis=1, keepdims=True)


def balance_from_clusters(
    probabilities: np.ndarray,
    clusters: np.ndarray,
    center_alpha: float,
    strength: float,
    iterations: int,
    min_cluster_size: int,
) -> np.ndarray:
    normalized = normalize_rows(probabilities)
    logs = np.log(np.clip(normalized, 1e-12, None))
    centered = logs.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= center_alpha * logs[mask].mean(
            axis=0,
            keepdims=True,
        )
    centered -= centered.max(axis=1, keepdims=True)
    result = np.exp(centered)
    result /= result.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if mask.sum() < min_cluster_size:
                continue
            block = result[mask]
            target = len(block) / len(LABELS)
            column_scale = target / np.clip(
                block.sum(axis=0, keepdims=True),
                1e-9,
                None,
            )
            block *= np.power(np.clip(column_scale, 1e-6, None), strength)
            block /= np.clip(block.sum(axis=1, keepdims=True), 1e-9, None)
            result[mask] = block
    return result


def sample_prompt_rows(
    groups: np.ndarray,
    responses_per_prompt: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    order = np.argsort(groups, kind="stable")
    ordered_groups = groups[order]
    boundaries = (
        np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1
    )
    for indices in np.split(order, boundaries):
        count = min(responses_per_prompt, len(indices))
        sampled.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(sampled))


def decision_probabilities(
    probabilities: np.ndarray,
    clusters: np.ndarray,
    mode: str,
    center_alpha: float,
    strength: float,
    iterations: int,
    min_cluster_size: int,
) -> np.ndarray:
    if mode == "raw":
        return normalize_rows(probabilities)
    if mode == "center":
        return center_probabilities(probabilities, clusters, center_alpha)
    if mode == "balance":
        return balance_from_clusters(
            probabilities,
            clusters,
            center_alpha,
            strength,
            iterations,
            min_cluster_size,
        )
    raise ValueError(f"Unknown decision mode: {mode}")


def present_class_recall(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[list[float], list[int]]:
    present = np.unique(targets)
    recalls = recall_score(
        targets,
        predictions,
        labels=np.arange(len(LABELS)),
        average=None,
        zero_division=0,
    )
    return (
        [float(value) for value in recalls[present]],
        [int(value) for value in present],
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {value!r}"
        )
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {value!r}"
        )
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        action="append",
        type=parse_branch,
        default=[],
        help="Additional NAME=PATH probability branch.",
    )
    parser.add_argument(
        "--drop-default-branches",
        action="store_true",
        help="Evaluate only branches supplied with --branch.",
    )
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument(
        "--sample-sizes",
        default="1,2,3,4,6,8,18",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20261116)
    parser.add_argument("--center-alpha", type=float, default=0.5)
    parser.add_argument("--balance-strength", type=float, default=0.625)
    parser.add_argument("--balance-iterations", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument(
        "--modes",
        default="raw,center,balance",
    )
    parser.add_argument(
        "--output",
        default="features/sparse_prompt_validation.json",
    )
    parser.add_argument(
        "--per-seed-output",
        default="",
        help="Optional JSON path for all per-seed scores.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["model"].map(label_to_index).to_numpy(dtype=np.int64)
    if np.isnan(labels).any():
        raise ValueError("Input CSV contains unknown model labels")
    groups = frame["prompt_id"].astype(str).to_numpy()
    texts = frame["text"].fillna("").astype(str).tolist()
    sample_sizes = [
        int(value) for value in args.sample_sizes.split(",") if value
    ]
    modes = [value.strip() for value in args.modes.split(",") if value]
    unknown_modes = set(modes) - {"raw", "center", "balance"}
    if unknown_modes:
        raise ValueError(f"Unknown modes: {sorted(unknown_modes)}")
    if "raw" not in modes:
        modes.insert(0, "raw")

    branch_paths: dict[str, Path] = {}
    if not args.drop_default_branches or not args.branch:
        branch_paths.update(
            {name: ROOT / path for name, path in DEFAULT_BRANCHES.items()}
        )
    for name, path in args.branch:
        branch_paths[name] = (
            path if path.is_absolute() else ROOT / path
        )
    if not branch_paths:
        raise ValueError("No probability branches were provided")

    branches = {
        name: normalize_rows(np.load(path))
        for name, path in branch_paths.items()
    }
    expected_shape = (len(frame), len(LABELS))
    for name, values in branches.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}, expected {expected_shape}"
            )

    per_seed: dict[str, dict[str, dict[str, list[float]]]] = {
        str(size): {
            name: {mode: [] for mode in modes}
            for name in branches
        }
        for size in sample_sizes
    }
    per_class: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        str(size): {
            name: {mode: [] for mode in modes}
            for name in branches
        }
        for size in sample_sizes
    }
    coverage: dict[str, list[float]] = defaultdict(list)

    for seed_offset in range(args.seeds):
        seed = args.seed_start + seed_offset
        rng = np.random.default_rng(seed)
        for size in sample_sizes:
            sampled = sample_prompt_rows(groups, size, rng)
            sampled_texts = [texts[index] for index in sampled]
            sampled_labels = labels[sampled]
            sampled_clusters = blind_prompt_clusters(sampled_texts, 0.90)
            coverage[str(size)].append(
                float(len(np.unique(sampled_labels)) / len(LABELS))
            )
            for name, probabilities in branches.items():
                for mode in modes:
                    decided = decision_probabilities(
                        probabilities[sampled],
                        sampled_clusters,
                        mode,
                        args.center_alpha,
                        args.balance_strength,
                        args.balance_iterations,
                        args.min_cluster_size,
                    )
                    predictions = decided.argmax(axis=1)
                    score = balanced_accuracy_score(
                        sampled_labels,
                        predictions,
                    )
                    per_seed[str(size)][name][mode].append(float(score))
                    recalls, present = present_class_recall(
                        sampled_labels,
                        predictions,
                    )
                    per_class[str(size)][name][mode].append(
                        np.asarray(recalls, dtype=np.float64)
                    )

    summary: dict[str, object] = {
        "metric": "balanced_accuracy",
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "sample_sizes": sample_sizes,
        "modes": modes,
        "coverage": {
            size: summarize(values) for size, values in coverage.items()
        },
        "branches": {},
    }
    for size in sample_sizes:
        key = str(size)
        size_summary: dict[str, object] = {}
        for name in branches:
            raw_scores = np.asarray(per_seed[key][name]["raw"])
            branch_summary: dict[str, object] = {}
            for mode in modes:
                scores = np.asarray(per_seed[key][name][mode])
                recalls = np.concatenate(
                    per_class[key][name][mode],
                    axis=0,
                )
                mode_summary = summarize(scores.tolist())
                mode_summary["mean_class_recall"] = float(
                    recalls.mean()
                )
                if mode == "raw":
                    mode_summary["delta_vs_raw"] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "win_rate": 0.0,
                    }
                else:
                    delta = scores - raw_scores
                    mode_summary["delta_vs_raw"] = {
                        "mean": float(delta.mean()),
                        "std": (
                            float(delta.std(ddof=1))
                            if len(delta) > 1
                            else 0.0
                        ),
                        "win_rate": float(np.mean(delta > 0)),
                    }
                branch_summary[mode] = mode_summary
            size_summary[name] = branch_summary
        summary["branches"][key] = size_summary

    print("SPARSE PROMPT VALIDATION")
    for size in sample_sizes:
        print(f"\nsample_size={size} per prompt")
        print(
            f"{'branch':<18} {'mode':<8} {'mean':>8} "
            f"{'std':>8} {'delta':>8} {'win':>7}"
        )
        for name in branches:
            for mode in modes:
                result = summary["branches"][str(size)][name][mode]
                print(
                    f"{name:<18} {mode:<8} "
                    f"{result['mean']:>8.4f} "
                    f"{result['std']:>8.4f} "
                    f"{result['delta_vs_raw']['mean']:>8.4f} "
                    f"{result['delta_vs_raw']['win_rate']:>7.2f}"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved {output}")
    if args.per_seed_output:
        per_seed_output = Path(args.per_seed_output)
        per_seed_output.parent.mkdir(parents=True, exist_ok=True)
        per_seed_output.write_text(
            json.dumps(per_seed, indent=2),
            encoding="utf-8",
        )
        print(f"saved {per_seed_output}")


if __name__ == "__main__":
    main()
