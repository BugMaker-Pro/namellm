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

sys.path.insert(0, str(ROOT / "scripts"))
from apply_cluster_balance import balance_probabilities  # noqa: E402


def center_log_probabilities(
    probabilities: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(normalized, 1e-12, None))
    clusters = blind_prompt_clusters(texts, 0.90)
    centered = log_probabilities.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= 0.5 * log_probabilities[mask].mean(
            axis=0,
            keepdims=True,
        )
    return centered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="features/summary_funnel_power_blend.npy",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "log", "centered", "raw_log", "rich"],
        required=True,
    )
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument("--c-values", default="")
    parser.add_argument("--balance-alpha", type=float, default=-1.0)
    parser.add_argument("--balance-strength", type=float, default=0.625)
    parser.add_argument("--balance-iterations", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    c_values = (
        [float(value) for value in args.c_values.split(",")]
        if args.c_values
        else [args.c]
    )

    paths = [
        Path(args.base),
        ROOT / "features" / "qwen3_embedding_4b_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    ]
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    branches = [
        np.load(path) / np.load(path).sum(axis=1, keepdims=True)
        for path in paths
    ]
    log_branches = [
        np.log(np.clip(branch, 1e-12, None))
        for branch in branches
    ]
    centered_branches = [np.zeros_like(branch) for branch in branches]
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    for _, valid_index in folds:
        texts = frame["text"].iloc[valid_index].tolist()
        for index, branch in enumerate(branches):
            centered_branches[index][valid_index] = center_log_probabilities(
                branch[valid_index],
                texts,
            )

    raw_features = np.hstack(branches)
    log_features = np.hstack(log_branches)
    centered_features = np.hstack(centered_branches)
    if args.mode == "raw":
        features = raw_features
    elif args.mode == "log":
        features = log_features
    elif args.mode == "centered":
        features = centered_features
    elif args.mode == "raw_log":
        features = np.hstack([raw_features, log_features])
    else:
        top_one_hot = []
        confidence_features = []
        group_features = []
        groups = sorted(set(group(label) for label in LABELS))
        for branch in branches:
            top = branch.argmax(axis=1)
            one_hot = np.zeros((len(frame), len(LABELS)), dtype=np.float32)
            one_hot[np.arange(len(frame)), top] = 1
            top_one_hot.append(one_hot)
            confidence_features.append(branch.max(axis=1, keepdims=True))
            confidence_features.append(
                -(branch * np.log(np.clip(branch, 1e-12, None))).sum(
                    axis=1,
                    keepdims=True,
                )
            )
            group_features.append(
                np.stack(
                    [
                        branch[
                            :,
                            [
                                index
                                for index, label in enumerate(LABELS)
                                if group(label) == group_name
                            ],
                        ].sum(axis=1)
                        for group_name in groups
                    ],
                    axis=1,
                )
            )
        features = np.hstack(
            [
                raw_features,
                log_features,
                centered_features,
                *top_one_hot,
                *confidence_features,
                *group_features,
            ]
        )
    features = features.astype(np.float32)

    probabilities = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        if len(c_values) == 1:
            selected_c = c_values[0]
        else:
            inner_folds = list(
                GroupKFold(4).split(
                    frame.iloc[train_index],
                    labels[train_index],
                    frame["prompt_id"].iloc[train_index],
                )
            )
            best_inner: tuple[float, float] | None = None
            for candidate_c in c_values:
                inner_probabilities = np.zeros(
                    (len(train_index), len(LABELS))
                )
                for _, inner_valid in inner_folds:
                    inner_train_index = np.setdiff1d(
                        np.arange(len(train_index)),
                        inner_valid,
                    )
                    inner_scaler = StandardScaler().fit(
                        features[train_index][inner_train_index]
                    )
                    inner_model = LogisticRegression(
                        C=candidate_c,
                        max_iter=5000,
                    )
                    inner_model.fit(
                        inner_scaler.transform(
                            features[train_index][inner_train_index]
                        ),
                        labels[train_index][inner_train_index],
                    )
                    inner_probabilities[inner_valid] = (
                        inner_model.predict_proba(
                            inner_scaler.transform(
                                features[train_index][inner_valid]
                            )
                        )
                    )
                if args.balance_alpha >= 0:
                    for _, inner_valid in inner_folds:
                        global_valid = train_index[inner_valid]
                        inner_probabilities[inner_valid] = (
                            balance_probabilities(
                                inner_probabilities[inner_valid],
                                frame["text"].iloc[global_valid].tolist(),
                                args.balance_alpha,
                                args.balance_strength,
                                args.balance_iterations,
                                args.min_cluster_size,
                            )
                        )
                inner_score = balanced_accuracy_score(
                    labels[train_index],
                    inner_probabilities.argmax(axis=1),
                )
                if best_inner is None or inner_score > best_inner[0]:
                    best_inner = (float(inner_score), candidate_c)
            if best_inner is None:
                raise RuntimeError("Nested stack search produced no candidate")
            selected_c = best_inner[1]
            print(
                f"fold={fold} selected_c={selected_c} "
                f"inner={best_inner[0]:.4f}",
                flush=True,
            )

        scaler = StandardScaler().fit(features[train_index])
        model = LogisticRegression(C=selected_c, max_iter=5000)
        model.fit(
            scaler.transform(features[train_index]),
            labels[train_index],
        )
        probabilities[valid_index] = model.predict_proba(
            scaler.transform(features[valid_index])
        )
        if args.balance_alpha >= 0:
            probabilities[valid_index] = balance_probabilities(
                probabilities[valid_index],
                frame["text"].iloc[valid_index].tolist(),
                args.balance_alpha,
                args.balance_strength,
                args.balance_iterations,
                args.min_cluster_size,
            )
        score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        print(f"fold={fold} valid={score:.4f}", flush=True)

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(
        f"logistic_stack mode={args.mode} c={args.c} OOF={score:.4f}",
        flush=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
