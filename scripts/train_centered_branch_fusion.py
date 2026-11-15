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


def centered_log_probabilities(
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
    return centered


def weight_candidates(branch_count: int) -> list[tuple[float, ...]]:
    increments = 20
    branch_max = 6
    candidates: list[tuple[float, ...]] = []
    branch_weights = [0] * branch_count

    def assign(index: int, remaining: int, base: int) -> None:
        if index == branch_count:
            if remaining == 0:
                candidates.append(
                    (base / increments,)
                    + tuple(value / increments for value in branch_weights)
                )
            return
        for value in range(min(branch_max, remaining) + 1):
            branch_weights[index] = value
            assign(index + 1, remaining - value, base)

    for base in range(8, 17):
        assign(0, increments - base, base)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-alpha", type=float, default=0.75)
    parser.add_argument(
        "--output",
        default="features/qwen3_embedding_centered_branch_fusion_oof.npy",
    )
    args = parser.parse_args()

    paths = {
        "base": ROOT / "features" / "summary_funnel_power_blend.npy",
        "e512": ROOT / "features" / "qwen3_embedding_4b_oof.npy",
        "e256": ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
        "layer4": ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
        "gru7": ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    }
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    raw_branches = [np.load(path) for path in paths.values()]
    centered_branches = [np.zeros_like(values) for values in raw_branches]
    for _, valid_index in GroupKFold(5).split(
        frame,
        labels,
        frame["prompt_id"],
    ):
        texts = frame["text"].iloc[valid_index].tolist()
        for index, values in enumerate(raw_branches):
            centered_branches[index][valid_index] = (
                centered_log_probabilities(
                    values[valid_index],
                    texts,
                    args.center_alpha,
                )
            )

    candidates = weight_candidates(len(centered_branches))
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    scores = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        best: tuple[float, tuple[float, ...]] | None = None
        for weights in candidates:
            blended = weights[0] * centered_branches[0][train_index]
            for weight, branch in zip(
                weights[1:],
                centered_branches[1:],
            ):
                blended += weight * branch[train_index]
            score = balanced_accuracy_score(
                labels[train_index],
                blended.argmax(axis=1),
            )
            if best is None or score > best[0]:
                best = (float(score), weights)

        if best is None:
            raise RuntimeError("Fusion search produced no candidate")
        inner_score, weights = best
        valid_scores = weights[0] * centered_branches[0][valid_index]
        for weight, branch in zip(weights[1:], centered_branches[1:]):
            valid_scores += weight * branch[valid_index]
        valid_scores -= valid_scores.max(axis=1, keepdims=True)
        valid_probabilities = np.exp(valid_scores)
        valid_probabilities /= valid_probabilities.sum(
            axis=1,
            keepdims=True,
        )
        scores[valid_index] = valid_probabilities
        valid_score = balanced_accuracy_score(
            labels[valid_index],
            valid_probabilities.argmax(axis=1),
        )
        weight_text = ",".join(
            f"{name}={weight:.2f}"
            for name, weight in zip(paths, weights)
        )
        print(
            f"fold={fold} inner={inner_score:.4f} valid={valid_score:.4f} "
            f"{weight_text}",
            flush=True,
        )

    score = balanced_accuracy_score(labels, scores.argmax(axis=1))
    print(f"centered_branch_fusion OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, scores)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
