from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV, family, group  # noqa: E402


COMPONENTS = {
    "base": ROOT / "features" / "summary_funnel_power_blend.npy",
    "e512": ROOT / "features" / "qwen3_embedding_4b_oof.npy",
    "e256": ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
    "layer4": ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
    "gru7": ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    "fusion": ROOT / "features" / "qwen3_embedding_fusion7_oof.npy",
}


def main() -> None:
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    probabilities = {
        name: np.load(path)
        for name, path in COMPONENTS.items()
    }
    predictions = {
        name: values.argmax(axis=1)
        for name, values in probabilities.items()
    }

    print("COMPONENT SCORES")
    for name, prediction in predictions.items():
        score = balanced_accuracy_score(labels, prediction)
        family_score = np.mean(
            [
                family(LABELS[predicted]) == family(actual)
                for predicted, actual in zip(prediction, frame["model"])
            ]
        )
        group_score = np.mean(
            [
                group(LABELS[predicted]) == group(actual)
                for predicted, actual in zip(prediction, frame["model"])
            ]
        )
        print(
            f"{name:<8} bacc={score:.4f} "
            f"family={family_score:.4f} group={group_score:.4f}"
        )

    fusion_predictions = predictions["fusion"]
    print("\nFUSION PER-CLASS RECALL")
    recalls = []
    for index, label in enumerate(LABELS):
        mask = labels == index
        recall = np.mean(fusion_predictions[mask] == index)
        recalls.append((recall, label, int(mask.sum())))
    for recall, label, count in sorted(recalls):
        print(f"{label:<28} recall={recall:.4f} n={count}")

    print("\nTOP CONFUSIONS")
    confusions: dict[tuple[int, int], int] = {}
    for actual, predicted in zip(labels, fusion_predictions):
        if actual != predicted:
            key = (int(actual), int(predicted))
            confusions[key] = confusions.get(key, 0) + 1
    for (actual, predicted), count in sorted(
        confusions.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(
            f"{LABELS[actual]:<28} -> {LABELS[predicted]:<28} n={count}"
        )

    print("\nPAIRWISE PREDICTION AGREEMENT")
    names = list(predictions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            agreement = np.mean(predictions[left] == predictions[right])
            print(f"{left:<8} {right:<8} agreement={agreement:.4f}")

    print("\nHEADROOM")
    correct_counts = np.sum(
        np.stack(
            [
                prediction == labels
                for name, prediction in predictions.items()
                if name != "fusion"
            ]
        ),
        axis=0,
    )
    print(
        f"fusion accuracy={np.mean(fusion_predictions == labels):.4f}, "
        f"oracle accuracy={np.mean(correct_counts > 0):.4f}"
    )
    for count in range(0, correct_counts.max() + 1):
        print(
            f"branches_correct={count} "
            f"rows={int(np.sum(correct_counts == count))}"
        )

    print("\nCORRECT ONLY BY COMPONENT")
    for name, prediction in predictions.items():
        if name == "fusion":
            continue
        correct = prediction == labels
        others = [
            other_prediction == labels
            for other_name, other_prediction in predictions.items()
            if other_name not in {name, "fusion"}
        ]
        unique = correct & ~np.any(others, axis=0)
        recovered = correct & ~(fusion_predictions == labels)
        print(
            f"{name:<8} unique_correct={int(unique.sum())} "
            f"missed_by_fusion={int(recovered.sum())}"
        )


if __name__ == "__main__":
    main()
