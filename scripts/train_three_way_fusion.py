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

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="features/summary_funnel_power_blend.npy",
    )
    parser.add_argument(
        "--embedding-512",
        default="features/qwen3_embedding_4b_oof.npy",
    )
    parser.add_argument(
        "--embedding-256",
        default="features/qwen3_embedding_4b_pca256_oof.npy",
    )
    parser.add_argument(
        "--output",
        default="features/qwen3_embedding_three_way_oof.npy",
    )
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    base = np.load(args.base)
    embedding_512 = np.load(args.embedding_512)
    embedding_256 = np.load(args.embedding_256)
    if not (
        base.shape
        == embedding_512.shape
        == embedding_256.shape
        == (len(frame), len(LABELS))
    ):
        raise ValueError("Fusion arrays do not align with training data")

    powers = [0.5, 0.75, 1.0, 1.25]
    powered_512 = {
        power: np.power(embedding_512, power) for power in powers
    }
    powered_256 = {
        power: np.power(embedding_256, power) for power in powers
    }
    weight_candidates = []
    for base_weight in np.arange(0.4, 0.81, 0.05):
        for weight_512 in np.arange(0.0, 0.41, 0.05):
            weight_256 = 1.0 - base_weight - weight_512
            if -1e-9 <= weight_256 <= 0.4 + 1e-9:
                weight_candidates.append(
                    (
                        round(float(base_weight), 2),
                        round(float(weight_512), 2),
                        round(float(weight_256), 2),
                    )
                )
    candidates = [
        (power_512, power_256, base_weight, weight_512, weight_256)
        for power_512 in powers
        for power_256 in powers
        for base_weight, weight_512, weight_256 in weight_candidates
    ]

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    probabilities = np.zeros_like(base)
    selections = []
    for fold, (train_index, valid_index) in enumerate(folds):
        train_labels = labels[train_index]
        base_train = base[train_index]
        powered_512_train = {
            power: values[train_index] for power, values in powered_512.items()
        }
        powered_256_train = {
            power: values[train_index] for power, values in powered_256.items()
        }
        inner_folds = list(
            GroupKFold(4).split(
                base[train_index],
                train_labels,
                frame["prompt_id"].iloc[train_index],
            )
        )
        best: tuple[float, tuple[float, float, float, float, float]] | None = None
        for candidate in candidates:
            (
                power_512,
                power_256,
                base_weight,
                weight_512,
                weight_256,
            ) = candidate
            inner_predictions = np.zeros(len(train_index), dtype=int)
            for _, inner_valid in inner_folds:
                blended = (
                    base_weight * base_train[inner_valid]
                    + weight_512
                    * powered_512_train[power_512][inner_valid]
                    + weight_256
                    * powered_256_train[power_256][inner_valid]
                )
                inner_predictions[inner_valid] = blended.argmax(axis=1)
            score = balanced_accuracy_score(train_labels, inner_predictions)
            if best is None or score > best[0]:
                best = (float(score), candidate)

        if best is None:
            raise RuntimeError("Fusion search produced no candidate")
        inner_score, candidate = best
        (
            power_512,
            power_256,
            base_weight,
            weight_512,
            weight_256,
        ) = candidate
        probabilities[valid_index] = (
            base_weight * base[valid_index]
            + weight_512 * powered_512[power_512][valid_index]
            + weight_256 * powered_256[power_256][valid_index]
        )
        valid_score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        selections.append(candidate)
        print(
            f"fold={fold} inner={inner_score:.4f} valid={valid_score:.4f} "
            f"powers=({power_512},{power_256}) weights="
            f"({base_weight},{weight_512},{weight_256})",
            flush=True,
        )

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"nested_fusion OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
