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


def center_log_probabilities(
    probabilities: np.ndarray,
    texts: list[str],
    alpha: float,
) -> np.ndarray:
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(normalized, 1e-12, None))
    clusters = blind_prompt_clusters(texts, 0.90)
    centered = log_probabilities.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= (
            alpha
            * log_probabilities[mask].mean(axis=0, keepdims=True)
        )
    centered -= centered.max(axis=1, keepdims=True)
    result = np.exp(centered)
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--prompt-column", default="prompt_id")
    parser.add_argument("--alpha", type=float, default=0.75)
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
        output_probabilities = center_log_probabilities(
            probabilities,
            frame[args.text_column].tolist(),
            args.alpha,
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
            output_probabilities[valid_index] = center_log_probabilities(
                probabilities[valid_index],
                frame[args.text_column].iloc[valid_index].tolist(),
                args.alpha,
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
