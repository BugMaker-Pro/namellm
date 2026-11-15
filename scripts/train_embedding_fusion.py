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


def weight_candidates(branch_count: int) -> list[tuple[float, ...]]:
    increments = 20
    base_units = range(10, 16)
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

    for base in base_units:
        assign(0, increments - base, base)
    return candidates


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
        "--layer4",
        default="features/qwen3_embedding_4b_layers4_oof.npy",
    )
    parser.add_argument(
        "--layer-sequence",
        default="features/qwen3_embedding_4b_layer_sequence_oof.npy",
    )
    parser.add_argument(
        "--layer-sequence7",
        default="features/qwen3_embedding_4b_layer_sequence7_oof.npy",
    )
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default="features/qwen3_embedding_fusion_oof.npy",
    )
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    base = np.load(args.base)
    branch_paths = [
        args.embedding_512,
        args.embedding_256,
        args.layer4,
        args.layer_sequence,
        args.layer_sequence7,
    ]
    branches = []
    names = []
    for name, path in zip(
        ["e512", "e256", "layer4", "gru4", "gru7"],
        branch_paths,
    ):
        branch_path = Path(path)
        if not branch_path.exists():
            continue
        values = np.load(branch_path)
        branches.append(np.power(values, args.power))
        names.append(name)
    if not branches:
        raise ValueError("At least one embedding branch is required")

    expected_shape = (len(frame), len(LABELS))
    if base.shape != expected_shape or any(
        branch.shape != expected_shape for branch in branches
    ):
        raise ValueError("Fusion arrays do not align with training data")

    candidates = weight_candidates(len(branches))
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    probabilities = np.zeros_like(base)
    for fold, (train_index, valid_index) in enumerate(folds):
        train_labels = labels[train_index]
        best: tuple[float, tuple[float, ...]] | None = None
        for weights in candidates:
            blended = weights[0] * base[train_index]
            for weight, branch in zip(weights[1:], branches):
                blended += weight * branch[train_index]
            score = balanced_accuracy_score(
                train_labels,
                blended.argmax(axis=1),
            )
            if best is None or score > best[0]:
                best = (float(score), weights)

        if best is None:
            raise RuntimeError("Fusion search produced no candidate")
        inner_score, weights = best
        blended_valid = weights[0] * base[valid_index]
        for weight, branch in zip(weights[1:], branches):
            blended_valid += weight * branch[valid_index]
        probabilities[valid_index] = blended_valid
        valid_score = balanced_accuracy_score(
            labels[valid_index],
            blended_valid.argmax(axis=1),
        )
        weight_text = ",".join(
            f"{name}={weight:.2f}"
            for name, weight in zip(names, weights[1:])
        )
        print(
            f"fold={fold} inner={inner_score:.4f} valid={valid_score:.4f} "
            f"base={weights[0]:.2f} {weight_text}",
            flush=True,
        )

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"nested_embedding_fusion OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
