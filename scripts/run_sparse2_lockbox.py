from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


Candidate = Callable[[pd.DataFrame, pd.DataFrame, int, int], np.ndarray]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {value!r}"
        )
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path


def parse_callable(value: str) -> tuple[str, Candidate]:
    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=MODULE:FUNCTION, got {value!r}"
        )
    name, target = value.split("=", 1)
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{target} is not callable")
    return name, function


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
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
        raise ValueError("Probability rows must have positive mass")
    return values / totals


def balanced_score(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(targets, predictions))


def make_fold_assignment(
    frame: pd.DataFrame,
    labels: np.ndarray,
    outer_folds: int,
) -> np.ndarray:
    assignment = np.full(len(frame), -1, dtype=np.int16)
    groups = frame["prompt_id"].astype(str).to_numpy()
    splitter = GroupKFold(outer_folds)
    for fold, (_, valid) in enumerate(
        splitter.split(frame, labels, groups)
    ):
        assignment[valid] = fold
    if np.any(assignment < 0):
        raise RuntimeError("Some rows did not receive an outer fold")
    return assignment


def sample_two_rows(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    responses_per_prompt: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    groups = frame["prompt_id"].astype(str).to_numpy()
    for prompt in sorted(np.unique(groups)):
        rows = np.flatnonzero(groups == prompt)
        count = min(responses_per_prompt, len(rows))
        selected.append(rng.choice(rows, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def bootstrap_prompt_delta(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    prompts: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    unique_prompts = np.unique(prompts)
    by_prompt = {
        prompt: np.flatnonzero(prompts == prompt)
        for prompt in unique_prompts
    }
    deltas = []
    for _ in range(replicates):
        sampled_prompts = rng.choice(
            unique_prompts,
            size=len(unique_prompts),
            replace=True,
        )
        rows = np.concatenate(
            [by_prompt[prompt] for prompt in sampled_prompts]
        )
        base_score = balanced_score(
            labels[rows],
            baseline_predictions[rows],
        )
        candidate_score = balanced_score(
            labels[rows],
            candidate_predictions[rows],
        )
        deltas.append(candidate_score - base_score)
    return np.asarray(deltas, dtype=np.float64)


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def protocol_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_candidate(
    probabilities: np.ndarray,
) -> Candidate:
    def predict(
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        seed: int,
        fold: int,
    ) -> np.ndarray:
        del train_frame, seed, fold
        return probabilities[valid_frame.index.to_numpy(dtype=np.int64)]

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument(
        "--protocol",
        default="validation/sparse2_protocol.json",
    )
    parser.add_argument(
        "--baseline-static",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--candidate-static",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--baseline-callable",
        action="append",
        type=parse_callable,
        default=[],
        metavar="NAME=MODULE:FUNCTION",
    )
    parser.add_argument(
        "--candidate-callable",
        action="append",
        type=parse_callable,
        default=[],
        metavar="NAME=MODULE:FUNCTION",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--responses-per-prompt", type=int)
    parser.add_argument(
        "--output",
        default="validation/sparse2_results.json",
    )
    parser.add_argument(
        "--per-seed-output",
        default="validation/sparse2_per_seed.csv",
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outer_folds = int(protocol["outer_folds"])
    seeds = int(args.seeds or protocol["seeds"])
    seed_start = int(protocol["seed_start"])
    responses_per_prompt = int(
        args.responses_per_prompt
        or protocol["responses_per_prompt"]
    )
    bootstrap_replicates = int(
        args.bootstrap_replicates
        or protocol["bootstrap_replicates"]
    )

    frame = pd.read_csv(args.input_csv)
    if "prompt_id" not in frame or "model" not in frame:
        raise ValueError("Input CSV must include prompt_id and model")
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["model"].map(label_to_index).to_numpy(dtype=np.int64)
    if np.isnan(labels).any():
        raise ValueError("Input CSV contains unknown labels")

    candidates: dict[str, Candidate] = {}
    for name, path in args.baseline_static + args.candidate_static:
        probabilities = normalize_probabilities(np.load(path))
        if probabilities.shape != (len(frame), len(LABELS)):
            raise ValueError(
                f"{name} has shape {probabilities.shape}, "
                f"expected {(len(frame), len(LABELS))}"
            )
        candidates[name] = static_candidate(
            probabilities
        )
    for name, function in (
        args.baseline_callable + args.candidate_callable
    ):
        candidates[name] = function
    if args.baseline not in candidates:
        raise ValueError(f"Baseline candidate {args.baseline!r} is missing")
    if len(candidates) < 2:
        raise ValueError("At least one candidate is required")
    fold_assignment = make_fold_assignment(frame, labels, outer_folds)

    all_scores: dict[str, list[float]] = {
        name: [] for name in candidates
    }
    all_seed_deltas: dict[str, list[float]] = {}
    prompt_bootstrap_deltas: dict[str, list[np.ndarray]] = {}
    per_seed_rows = []

    for seed_offset in range(seeds):
        seed = seed_start + seed_offset
        rng = np.random.default_rng(seed)
        sampled = sample_two_rows(
            frame,
            rng,
            responses_per_prompt,
        )
        seed_probabilities = {
            name: np.zeros((len(sampled), len(LABELS)), dtype=np.float64)
            for name in candidates
        }
        sample_position = {
            int(source_index): position
            for position, source_index in enumerate(sampled)
        }
        for fold in range(outer_folds):
            train_index = np.flatnonzero(
                fold_assignment != fold
            )
            valid_source_index = sampled[
                fold_assignment[sampled] == fold
            ]
            if not len(valid_source_index):
                continue
            train_frame = frame.iloc[train_index].copy()
            valid_frame = frame.iloc[valid_source_index].copy()
            for name, candidate in candidates.items():
                predicted = normalize_probabilities(
                    candidate(
                        train_frame,
                        valid_frame,
                        seed,
                        fold,
                    )
                )
                if predicted.shape != (len(valid_frame), len(LABELS)):
                    raise ValueError(
                        f"{name} fold {fold} returned {predicted.shape}, "
                        f"expected {(len(valid_frame), len(LABELS))}"
                    )
                for local_row, source_index in enumerate(
                    valid_source_index
                ):
                    seed_probabilities[name][
                        sample_position[int(source_index)]
                    ] = predicted[local_row]

        sampled_labels = labels[sampled]
        sampled_prompts = frame["prompt_id"].astype(str).to_numpy()[sampled]
        predictions = {
            name: probabilities.argmax(axis=1)
            for name, probabilities in seed_probabilities.items()
        }
        for name in candidates:
            score = balanced_score(
                sampled_labels,
                predictions[name],
            )
            all_scores[name].append(score)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "candidate": name,
                    "balanced_accuracy": score,
                    "rows": len(sampled),
                    "present_classes": len(
                        np.unique(sampled_labels)
                    ),
                }
            )

        baseline_predictions = predictions[args.baseline]
        bootstrap_rng = np.random.default_rng(seed + 1_000_000)
        for name in candidates:
            if name == args.baseline:
                continue
            all_seed_deltas.setdefault(name, []).append(
                all_scores[name][-1] - all_scores[args.baseline][-1]
            )
            prompt_bootstrap_deltas.setdefault(name, []).append(
                bootstrap_prompt_delta(
                    sampled_labels,
                    baseline_predictions,
                    predictions[name],
                    sampled_prompts,
                    bootstrap_rng,
                    bootstrap_replicates,
                )
            )

    baseline_scores = np.asarray(all_scores[args.baseline])
    result: dict[str, object] = {
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_digest(protocol_path),
            "outer_folds": outer_folds,
            "seeds": seeds,
            "seed_start": seed_start,
            "responses_per_prompt": responses_per_prompt,
            "bootstrap_replicates": bootstrap_replicates,
            "overrides": {
                "seeds": args.seeds is not None,
                "responses_per_prompt": (
                    args.responses_per_prompt is not None
                ),
                "bootstrap_replicates": (
                    args.bootstrap_replicates is not None
                ),
            },
        },
        "baseline": args.baseline,
        "class_coverage": {
            "present_classes": int(len(np.unique(labels))),
            "expected_classes": len(LABELS),
        },
        "candidates": {},
    }
    for name, scores in all_scores.items():
        score_summary = quantile_summary(np.asarray(scores))
        candidate_result: dict[str, object] = {
            "score": score_summary,
        }
        if name != args.baseline:
            deltas = np.asarray(all_seed_deltas[name])
            bootstrap_means = np.mean(
                np.stack(prompt_bootstrap_deltas[name]),
                axis=0,
            )
            candidate_result["paired_delta"] = quantile_summary(deltas)
            candidate_result["paired_delta"]["win_rate"] = float(
                np.mean(deltas > 0)
            )
            candidate_result["bootstrap_delta_mean"] = {
                "mean": float(bootstrap_means.mean()),
                "p025": float(np.quantile(bootstrap_means, 0.025)),
                "p975": float(np.quantile(bootstrap_means, 0.975)),
            }
            acceptance = protocol.get("acceptance", {})
            mean_delta = candidate_result["paired_delta"]["mean"]
            win_rate = candidate_result["paired_delta"]["win_rate"]
            ci_low = candidate_result["bootstrap_delta_mean"]["p025"]
            candidate_result["acceptance"] = {
                name: {
                    "min_delta": threshold,
                    "passed": bool(
                        mean_delta >= threshold
                        and win_rate
                        >= float(
                            acceptance.get(
                                "minimum_win_rate",
                                0.0,
                            )
                        )
                        and (
                            ci_low > 0
                            if acceptance.get(
                                "bootstrap_ci_lower_must_exceed_zero",
                                False,
                            )
                            else True
                        )
                    ),
                }
                for name, threshold in acceptance.items()
                if name.endswith("_delta")
            }
        result["candidates"][name] = candidate_result

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    per_seed_path = Path(args.per_seed_output)
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_seed_rows).to_csv(
        per_seed_path,
        index=False,
    )
    print(json.dumps(result, indent=2))
    print(f"saved {output_path}")
    print(f"saved {per_seed_path}")


if __name__ == "__main__":
    main()
