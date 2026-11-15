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
        "--primary",
        default="features/qwen3_embedding_cluster_balanced_fusion_oof.npy",
    )
    parser.add_argument(
        "--stack",
        default="features/qwen3_embedding_rich_stack_balanced_oof.npy",
    )
    parser.add_argument("--weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument(
        "--output",
        default="features/qwen3_embedding_stack_blend_oof.npy",
    )
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    primary = np.load(args.primary)
    stack = np.load(args.stack)
    expected_shape = (len(frame), len(LABELS))
    if primary.shape != expected_shape or stack.shape != expected_shape:
        raise ValueError("Blend arrays do not align with the training data")
    weights = [float(value) for value in args.weights.split(",")]

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    probabilities = np.zeros_like(primary)
    for fold, (train_index, valid_index) in enumerate(folds):
        inner_folds = list(
            GroupKFold(4).split(
                frame.iloc[train_index],
                labels[train_index],
                frame["prompt_id"].iloc[train_index],
            )
        )
        best: tuple[float, float] | None = None
        for weight in weights:
            inner_predictions = np.zeros(len(train_index), dtype=int)
            for _, inner_valid in inner_folds:
                blended = (
                    (1.0 - weight) * primary[train_index][inner_valid]
                    + weight * stack[train_index][inner_valid]
                )
                inner_predictions[inner_valid] = blended.argmax(axis=1)
            score = balanced_accuracy_score(
                labels[train_index],
                inner_predictions,
            )
            if best is None or score > best[0]:
                best = (float(score), weight)
        if best is None:
            raise RuntimeError("Blend search produced no candidate")
        inner_score, weight = best
        probabilities[valid_index] = (
            (1.0 - weight) * primary[valid_index]
            + weight * stack[valid_index]
        )
        valid_score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        print(
            f"fold={fold} inner={inner_score:.4f} valid={valid_score:.4f} "
            f"stack_weight={weight:.2f}",
            flush=True,
        )

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"stack_blend OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
