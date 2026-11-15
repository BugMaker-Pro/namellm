from __future__ import annotations

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


def center(
    probabilities: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(normalized, 1e-12, None))
    centered = log_probabilities.copy()
    for group_id in np.unique(groups):
        mask = groups == group_id
        centered[mask] -= (
            alpha
            * log_probabilities[mask].mean(axis=0, keepdims=True)
        )
    centered -= centered.max(axis=1, keepdims=True)
    result = np.exp(centered)
    return result / result.sum(axis=1, keepdims=True)


def balance(
    probabilities: np.ndarray,
    groups: np.ndarray,
    strength: float,
    iterations: int,
) -> np.ndarray:
    result = probabilities.copy()
    for _ in range(iterations):
        for group_id in np.unique(groups):
            mask = groups == group_id
            if mask.sum() < 8:
                continue
            block = result[mask]
            target = len(block) / len(LABELS)
            scale = target / np.clip(
                block.sum(axis=0, keepdims=True),
                1e-9,
                None,
            )
            block *= np.power(np.clip(scale, 1e-6, None), strength)
            block /= np.clip(block.sum(axis=1, keepdims=True), 1e-9, None)
            result[mask] = block
    return result


def main() -> None:
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    raw = np.load(
        ROOT / "features" / "qwen3_embedding_fusion7_oof.npy"
    )
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))

    for keep_rate in [1.0, 0.9, 0.8, 0.7]:
        baseline_scores = []
        balanced_scores = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            baseline_predictions = np.zeros(len(frame), dtype=int)
            balanced_predictions = np.zeros(len(frame), dtype=int)
            keep_mask = np.zeros(len(frame), dtype=bool)
            for _, valid_index in folds:
                groups = blind_prompt_clusters(
                    frame["text"].iloc[valid_index].tolist(),
                    0.90,
                )
                local_keep = np.ones(len(valid_index), dtype=bool)
                if keep_rate < 1:
                    for group_id in np.unique(groups):
                        positions = np.where(groups == group_id)[0]
                        local_keep[positions] = (
                            rng.random(len(positions)) < keep_rate
                        )
                retained_index = valid_index[local_keep]
                retained_groups = groups[local_keep]
                keep_mask[valid_index] = local_keep
                centered = center(
                    raw[retained_index],
                    retained_groups,
                    0.5,
                )
                baseline_predictions[retained_index] = centered.argmax(axis=1)
                balanced_predictions[retained_index] = balance(
                    centered,
                    retained_groups,
                    0.625,
                    5,
                ).argmax(axis=1)
            baseline_scores.append(
                balanced_accuracy_score(
                    labels[keep_mask],
                    baseline_predictions[keep_mask],
                )
            )
            balanced_scores.append(
                balanced_accuracy_score(
                    labels[keep_mask],
                    balanced_predictions[keep_mask],
                )
            )
        differences = np.array(balanced_scores) - np.array(baseline_scores)
        print(
            f"keep={keep_rate:.1f} "
            f"centered={np.mean(baseline_scores):.4f} "
            f"balanced={np.mean(balanced_scores):.4f} "
            f"delta={np.mean(differences):+.4f} "
            f"wins={np.mean(differences > 0):.2f}"
        )


if __name__ == "__main__":
    main()
