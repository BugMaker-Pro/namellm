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

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV, group  # noqa: E402


def squash_rows(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-alpha", type=float, default=0.75)
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument(
        "--output",
        default="D:/tmp/namellm_candidate_reranker_oof.npy",
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
    clusters = blind_prompt_clusters(frame["text"].tolist(), 0.90)
    class_groups = np.array([group(label) for label in LABELS])
    group_names = sorted(set(class_groups))
    group_to_index = {
        group_name: index for index, group_name in enumerate(group_names)
    }
    class_group_indices = np.array(
        [group_to_index[group_name] for group_name in class_groups]
    )

    branch_probabilities = []
    branch_centered = []
    branch_rank = []
    branch_top1 = []
    for path in paths.values():
        values = squash_rows(np.load(path))
        log_probabilities = np.log(np.clip(values, 1e-12, None))
        centered = log_probabilities.copy()
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            centered[mask] -= (
                args.center_alpha
                * log_probabilities[mask].mean(axis=0, keepdims=True)
            )
        order = np.argsort(-values, axis=1)
        ranks = np.empty_like(order, dtype=np.float32)
        np.put_along_axis(
            ranks,
            order,
            np.arange(len(LABELS), dtype=np.float32)[None, :],
            axis=1,
        )
        branch_probabilities.append(values.astype(np.float32))
        branch_centered.append(centered.astype(np.float32))
        branch_rank.append((ranks / (len(LABELS) - 1)).astype(np.float32))
        branch_top1.append((ranks == 0).astype(np.float32))

    candidate_features = np.stack(
        [
            *branch_probabilities,
            *branch_centered,
            *branch_rank,
            *branch_top1,
        ],
        axis=-1,
    )
    centered_stack = np.stack(branch_centered, axis=-1)
    probability_stack = np.stack(branch_probabilities, axis=-1)
    candidate_features = np.concatenate(
        [
            candidate_features,
            centered_stack.mean(axis=-1, keepdims=True),
            centered_stack.max(axis=-1, keepdims=True),
            centered_stack.min(axis=-1, keepdims=True),
            centered_stack.std(axis=-1, keepdims=True),
            probability_stack.mean(axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    class_one_hot = np.broadcast_to(
        np.eye(len(LABELS), dtype=np.float32)[None, :, :],
        (len(frame), len(LABELS), len(LABELS)),
    )
    group_one_hot = np.broadcast_to(
        np.eye(len(group_names), dtype=np.float32)[class_group_indices][
            None, :, :
        ],
        (len(frame), len(LABELS), len(group_names)),
    )
    candidate_features = np.concatenate(
        [candidate_features, class_one_hot, group_one_hot],
        axis=-1,
    )
    feature_count = candidate_features.shape[-1]
    flat_features = candidate_features.reshape(-1, feature_count)
    candidate_labels = (
        np.broadcast_to(labels[:, None], (len(frame), len(LABELS)))
        == np.arange(len(LABELS))[None, :]
    ).reshape(-1)

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    row_probabilities = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        train_rows = np.repeat(
            train_index,
            len(LABELS),
        ) * len(LABELS) + np.tile(np.arange(len(LABELS)), len(train_index))
        valid_rows = np.repeat(
            valid_index,
            len(LABELS),
        ) * len(LABELS) + np.tile(np.arange(len(LABELS)), len(valid_index))
        scaler = StandardScaler().fit(flat_features[train_rows])
        model = LogisticRegression(
            C=args.c,
            max_iter=3000,
            class_weight="balanced",
        )
        model.fit(
            scaler.transform(flat_features[train_rows]),
            candidate_labels[train_rows],
        )
        scores = model.decision_function(
            scaler.transform(flat_features[valid_rows])
        ).reshape(len(valid_index), len(LABELS))
        row_probabilities[valid_index] = np.exp(
            scores - scores.max(axis=1, keepdims=True)
        )
        row_probabilities[valid_index] /= row_probabilities[
            valid_index
        ].sum(axis=1, keepdims=True)
        score = balanced_accuracy_score(
            labels[valid_index],
            row_probabilities[valid_index].argmax(axis=1),
        )
        print(f"fold={fold} valid={score:.4f}", flush=True)

    score = balanced_accuracy_score(
        labels,
        row_probabilities.argmax(axis=1),
    )
    print(f"candidate_reranker OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, row_probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
