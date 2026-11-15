from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def balance_probabilities(
    probabilities: np.ndarray,
    texts: list[str],
    center_alpha: float,
    strength: float,
    iterations: int,
    min_cluster_size: int,
) -> np.ndarray:
    clusters = blind_prompt_clusters(texts, 0.90)
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(normalized, 1e-12, None))
    centered = log_probabilities.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= (
            center_alpha
            * log_probabilities[mask].mean(axis=0, keepdims=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--prompt-column", default="prompt_id")
    parser.add_argument("--center-alpha", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.625)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["all", "validation"],
        default="all",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    probabilities = np.load(args.probabilities)
    if len(frame) != len(probabilities):
        raise ValueError("Probability rows do not align with the input CSV")

    if args.mode == "all":
        output_probabilities = balance_probabilities(
            probabilities,
            frame[args.text_column].tolist(),
            args.center_alpha,
            args.strength,
            args.iterations,
            args.min_cluster_size,
        )
    else:
        if args.label_column not in frame.columns:
            raise ValueError("Validation mode requires labels")
        if args.prompt_column not in frame.columns:
            raise ValueError("Validation mode requires prompt groups")
        label_to_index = {
            label: index for index, label in enumerate(LABELS)
        }
        labels = np.array(
            [label_to_index[value] for value in frame[args.label_column]]
        )
        output_probabilities = np.zeros_like(probabilities)
        folds = GroupKFold(5).split(
            frame,
            labels,
            frame[args.prompt_column],
        )
        for fold, (_, valid_index) in enumerate(folds):
            output_probabilities[valid_index] = balance_probabilities(
                probabilities[valid_index],
                frame[args.text_column].iloc[valid_index].tolist(),
                args.center_alpha,
                args.strength,
                args.iterations,
                args.min_cluster_size,
            )
            score = balanced_accuracy_score(
                labels[valid_index],
                output_probabilities[valid_index].argmax(axis=1),
            )
            print(f"fold={fold} valid={score:.4f}", flush=True)

    if args.label_column in frame.columns:
        label_to_index = {
            label: index for index, label in enumerate(LABELS)
        }
        labels = np.array(
            [label_to_index[value] for value in frame[args.label_column]]
        )
        score = balanced_accuracy_score(
            labels,
            output_probabilities.argmax(axis=1),
        )
        print(f"OOF={score:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, output_probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
