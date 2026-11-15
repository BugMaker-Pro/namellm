from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apply_cluster_balance import balance_probabilities  # noqa: E402
from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import (  # noqa: E402
    LABELS,
    TEST_CSV,
    TRAIN_CSV,
    group,
)


def normalize(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def center_by_texts(
    probabilities: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    normalized = normalize(probabilities)
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


def rich_features(
    branches: list[np.ndarray],
    centered: list[np.ndarray],
) -> np.ndarray:
    normalized = [normalize(branch) for branch in branches]
    top_one_hot = []
    confidence_features = []
    group_features = []
    group_names = sorted(set(group(label) for label in LABELS))
    for branch in normalized:
        top = branch.argmax(axis=1)
        one_hot = np.zeros((len(branch), len(LABELS)), dtype=np.float32)
        one_hot[np.arange(len(branch)), top] = 1
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
                    for group_name in group_names
                ],
                axis=1,
            )
        )
    return np.hstack(
        [
            *normalized,
            *[np.log(np.clip(branch, 1e-12, None)) for branch in normalized],
            *centered,
            *top_one_hot,
            *confidence_features,
            *group_features,
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--funnel-test",
        default="D:/tmp/namellm_funnel_test_prob.npy",
    )
    parser.add_argument(
        "--embedding-test",
        default="D:/tmp/namellm_embedding_test_probabilities.npz",
    )
    parser.add_argument("--c", type=float, default=0.003)
    parser.add_argument(
        "--output",
        default="submissions/submission.parquet",
    )
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in train["model"]])
    train_branches = [
        np.load(ROOT / "features" / "funnel_oof.npy"),
        np.load(ROOT / "features" / "qwen3_embedding_4b_oof.npy"),
        np.load(
            ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy"
        ),
        np.load(ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy"),
        np.load(
            ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy"
        ),
    ]
    embedding_test = np.load(args.embedding_test)
    test_branches = [
        np.load(args.funnel_test),
        embedding_test["e512"],
        embedding_test["e256"],
        embedding_test["layer4"],
        embedding_test["gru7"],
    ]

    centered_train = [np.zeros_like(branch) for branch in train_branches]
    folds = list(GroupKFold(5).split(train, labels, train["prompt_id"]))
    for _, valid_index in folds:
        texts = train["text"].iloc[valid_index].tolist()
        for index, branch in enumerate(train_branches):
            centered_train[index][valid_index] = center_by_texts(
                branch[valid_index],
                texts,
            )
    centered_test = [
        center_by_texts(branch, test["text"].tolist())
        for branch in test_branches
    ]

    train_features = rich_features(train_branches, centered_train)
    test_features = rich_features(test_branches, centered_test)
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(C=args.c, max_iter=5000)
    model.fit(scaler.transform(train_features), labels)
    probabilities = model.predict_proba(
        scaler.transform(test_features)
    )
    probabilities = balance_probabilities(
        probabilities,
        test["text"].tolist(),
        center_alpha=0.5,
        strength=0.625,
        iterations=5,
        min_cluster_size=8,
    )

    sample = pd.read_parquet(ROOT / "data" / "raw" / "sample_submission.parquet")
    if set(sample["id"]) != set(test["id"]):
        raise ValueError("Test and sample-submission IDs differ")
    raw_predictions = pd.DataFrame(
        {
            "id": test["id"],
            "model": np.array(LABELS)[probabilities.argmax(axis=1)],
            "confidence": probabilities.max(axis=1),
        }
    )
    detail = sample[["id"]].merge(raw_predictions, on="id", how="left")
    predictions = detail[["id", "model"]]
    if predictions["model"].isna().any():
        raise ValueError("Submission contains missing predictions")
    if not set(predictions["model"]).issubset(LABELS):
        raise ValueError("Submission contains invalid labels")

    print(detail.to_string(index=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    print(f"saved {output} rows={len(predictions)}", flush=True)


if __name__ == "__main__":
    main()
