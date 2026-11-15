from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


SUMMARY_PATH = ROOT / "features" / "summary_oof.npy"
FUNNEL_PATH = ROOT / "features" / "funnel_oof.npy"
OUTPUT_PATH = ROOT / "features" / "summary_funnel_power_blend.npy"


def blend(
    summary: np.ndarray,
    funnel: np.ndarray,
    summary_power: float,
    funnel_power: float,
    summary_weight: float,
) -> np.ndarray:
    return (
        summary_weight * np.power(summary, summary_power)
        + (1.0 - summary_weight) * np.power(funnel, funnel_power)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    summary = np.load(SUMMARY_PATH)
    funnel = np.load(FUNNEL_PATH)
    if summary.shape != funnel.shape or summary.shape != (len(frame), len(LABELS)):
        raise ValueError("OOF probability arrays do not align with training data")

    equal = summary + funnel
    equal_score = balanced_accuracy_score(labels, equal.argmax(axis=1))
    print(f"equal_blend balanced_accuracy={equal_score:.4f}", flush=True)

    stack_features = np.hstack(
        [np.log(summary + 1e-9), np.log(funnel + 1e-9)]
    )
    stack_probabilities = np.zeros_like(summary)
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    for train_index, valid_index in folds:
        scaler = StandardScaler().fit(stack_features[train_index])
        model = LogisticRegression(C=0.03, max_iter=3000)
        model.fit(scaler.transform(stack_features[train_index]), labels[train_index])
        stack_probabilities[valid_index] = model.predict_proba(
            scaler.transform(stack_features[valid_index])
        )
    stack_score = balanced_accuracy_score(
        labels,
        stack_probabilities.argmax(axis=1),
    )
    print(f"logistic_stack balanced_accuracy={stack_score:.4f}", flush=True)

    powers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    grid = [
        (summary_power, funnel_power, summary_weight)
        for summary_power in powers
        for funnel_power in powers
        for summary_weight in weights
    ]
    probabilities = np.zeros_like(summary)
    selections: list[tuple[float, float, float, float]] = []
    for train_index, valid_index in folds:
        train_summary = summary[train_index]
        train_funnel = funnel[train_index]
        train_labels = labels[train_index]
        inner_folds = GroupKFold(4).split(
            train_summary,
            train_labels,
            frame["prompt_id"].iloc[train_index],
        )
        inner_folds = list(inner_folds)

        best: tuple[float, float, float, float] | None = None
        for summary_power, funnel_power, summary_weight in grid:
            predictions = np.zeros(len(train_index), dtype=int)
            for _, inner_valid in inner_folds:
                candidate = blend(
                    train_summary[inner_valid],
                    train_funnel[inner_valid],
                    summary_power,
                    funnel_power,
                    summary_weight,
                )
                predictions[inner_valid] = candidate.argmax(axis=1)
            score = balanced_accuracy_score(train_labels, predictions)
            if best is None or score > best[0]:
                best = (
                    score,
                    summary_power,
                    funnel_power,
                    summary_weight,
                )

        if best is None:
            raise RuntimeError("Hyperparameter selection produced no candidate")
        _, summary_power, funnel_power, summary_weight = best
        selections.append(
            (summary_power, funnel_power, summary_weight, best[0])
        )
        probabilities[valid_index] = blend(
            summary[valid_index],
            funnel[valid_index],
            summary_power,
            funnel_power,
            summary_weight,
        )

    power_score = balanced_accuracy_score(
        labels,
        probabilities.argmax(axis=1),
    )
    print(f"nested_power_blend balanced_accuracy={power_score:.4f}", flush=True)
    for fold, (summary_power, funnel_power, weight, inner_score) in enumerate(
        selections
    ):
        print(
            f"fold={fold} inner={inner_score:.4f} "
            f"summary_power={summary_power:.2f} "
            f"funnel_power={funnel_power:.2f} weight={weight:.2f}",
            flush=True,
        )

    if power_score <= equal_score:
        raise ValueError("Power-calibrated blend did not beat the equal blend")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
