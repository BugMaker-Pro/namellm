from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_owned_kaggle_inference import (  # noqa: E402
    LayerSequenceModel,
    balance_probabilities,
    center_by_texts,
    extra_features,
    group_relative,
    logistic_probabilities,
    rich_features,
    split_layers,
)


def main() -> None:
    artifact_dir = Path("D:/tmp/namellm_owned_artifacts")
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    from namellm.clustering import blind_prompt_clusters

    test_clusters = blind_prompt_clusters(test["text"].tolist(), 0.90)
    lm_frame = pd.read_parquet(
        ROOT / "features" / "e002_qwen3-4b-base-4bit.parquet"
    )
    test_lm = (
        lm_frame.query("split == 'test'")
        .set_index("id")
        .loc[test["id"]]
        .drop(columns="split")
        .to_numpy(dtype=np.float32)
    )
    test_extra = extra_features(test, test_clusters, test_lm)

    compact = joblib.load(artifact_dir / "compact_tfidf.joblib")
    compact_matrix = hstack(
        [
            compact["char"].transform(test["text"]),
            compact["word"].transform(test["text"]),
        ]
    ).tocsr()
    branch_probabilities = [
        logistic_probabilities(compact["model"], compact_matrix)
    ]

    feature_data = np.load(
        ROOT / "features" / "qwen3_embedding_4b_test_layers7.npz",
        allow_pickle=True,
    )
    test_features = feature_data["X"].astype(np.float32)
    layers = [int(value) for value in feature_data["layers"]]
    linear_artifacts = joblib.load(
        artifact_dir / "embedding_linear.joblib"
    )
    for name in ["e512", "e256", "layer4"]:
        artifact = linear_artifacts[name]
        selected = split_layers(
            test_features,
            layers,
            artifact["wanted_layers"],
        )
        pca_test = artifact["pca"].transform(
            artifact["scaler"].transform(selected)
        )
        combined = np.hstack(
            [
                pca_test,
                group_relative(pca_test, test_clusters),
                test_extra,
            ]
        )
        branch_probabilities.append(
            logistic_probabilities(
                artifact["classifier"],
                artifact["combined_scaler"].transform(combined),
            )
        )

    hidden_size = test_features.shape[1] // (len(layers) * 4)
    sequence_features = test_features.reshape(
        len(test),
        len(layers),
        4 * hidden_size,
    )
    gru_transforms = joblib.load(artifact_dir / "gru7_transforms.joblib")
    sequence_arrays = []
    for index, transform in enumerate(gru_transforms["layer_transforms"]):
        pca_test = transform["pca"].transform(
            transform["scaler"].transform(sequence_features[:, index])
        )
        sequence_arrays.append(
            np.hstack(
                [
                    pca_test,
                    group_relative(pca_test, test_clusters),
                ]
            ).astype(np.float32)
        )
    gru_sequence = np.stack(sequence_arrays, axis=1)
    gru_extra = gru_transforms["extra_scaler"].transform(test_extra).astype(
        np.float32
    )
    checkpoint = torch.load(
        artifact_dir / "gru7_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    gru_model = LayerSequenceModel(
        input_size=gru_sequence.shape[-1],
        extra_size=test_extra.shape[1],
        hidden_size=128,
        layer_count=len(layers),
    )
    gru_model.load_state_dict(checkpoint["state_dict"])
    gru_model.eval()
    with torch.inference_mode():
        logits = gru_model(
            torch.from_numpy(gru_sequence),
            torch.from_numpy(gru_extra),
        )
        branch_probabilities.append(
            torch.softmax(logits.float(), dim=-1).numpy()
        )

    centered = [
        center_by_texts(branch, test["text"].tolist())
        for branch in branch_probabilities
    ]
    stack_features = rich_features(branch_probabilities, centered)
    stack = joblib.load(artifact_dir / "rich_stack.joblib")
    probabilities = logistic_probabilities(
        stack["model"],
        stack["scaler"].transform(stack_features),
    )
    probabilities = balance_probabilities(
        probabilities,
        test["text"].tolist(),
    )
    predictions = test[["id"]].copy()
    from namellm.constants import LABELS

    predictions["model"] = np.array(LABELS)[
        probabilities.argmax(axis=1)
    ]
    predictions["confidence"] = probabilities.max(axis=1)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
