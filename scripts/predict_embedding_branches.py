from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TEST_CSV, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402
from train_layer_sequence import LayerSequenceModel  # noqa: E402


def split_layers(
    features: np.ndarray,
    layers: list[int],
    wanted: list[int],
) -> np.ndarray:
    hidden_size = features.shape[1] // (len(layers) * 4)
    block_size = hidden_size * 4
    blocks = {
        layer: features[
            :,
            index * block_size : (index + 1) * block_size,
        ]
        for index, layer in enumerate(layers)
    }
    return np.concatenate([blocks[layer] for layer in wanted], axis=1)


def extra_features(
    frame: pd.DataFrame,
    clusters: np.ndarray,
    lm_features: pd.DataFrame,
) -> np.ndarray:
    style = style_frame(frame["text"]).to_numpy(dtype=np.float32)
    lm = (
        lm_features.set_index("id")
        .loc[frame["id"]]
        .drop(columns="split")
        .to_numpy(dtype=np.float32)
    )
    return np.hstack(
        [
            style,
            group_relative(style, clusters),
            lm,
            group_relative(lm, clusters),
        ]
    )


def fit_linear_branch(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_extra: np.ndarray,
    test_extra: np.ndarray,
    train_labels: np.ndarray,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    pca_components: int,
    c: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_features)
    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=42,
    ).fit(scaler.transform(train_features))
    train_pca = pca.transform(scaler.transform(train_features))
    test_pca = pca.transform(scaler.transform(test_features))
    train_all = np.hstack(
        [
            train_pca,
            group_relative(train_pca, train_clusters),
            train_extra,
        ]
    )
    test_all = np.hstack(
        [
            test_pca,
            group_relative(test_pca, test_clusters),
            test_extra,
        ]
    )
    combined_scaler = StandardScaler().fit(train_all)
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(combined_scaler.transform(train_all), train_labels)
    return model.predict_proba(combined_scaler.transform(test_all))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-features",
        default="features/qwen3_embedding_4b_layers7.npz",
    )
    parser.add_argument(
        "--test-features",
        default="features/qwen3_embedding_4b_test_layers7.npz",
    )
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--output",
        default="D:/tmp/namellm_embedding_test_probabilities.npz",
    )
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    train_labels = np.array(
        [label_to_index[value] for value in train["model"]]
    )
    train_data = np.load(args.train_features, allow_pickle=True)
    test_data = np.load(args.test_features, allow_pickle=True)
    layers = [int(value) for value in train_data["layers"]]
    if layers != [int(value) for value in test_data["layers"]]:
        raise ValueError("Train and test layer layouts differ")
    train_features = train_data["X"].astype(np.float32)
    test_features = test_data["X"].astype(np.float32)
    lm_frame = pd.read_parquet(args.lm_features)
    train_lm = lm_frame.query("split == 'train'")
    test_lm = lm_frame.query("split == 'test'")
    train_clusters = blind_prompt_clusters(train["text"].tolist(), 0.90)
    test_clusters = blind_prompt_clusters(test["text"].tolist(), 0.90)
    train_extra = extra_features(train, train_clusters, train_lm)
    test_extra = extra_features(test, test_clusters, test_lm)

    probabilities = {}
    for name, wanted_layers, components in [
        ("e512", [24, 36], 512),
        ("e256", [24, 36], 256),
        ("layer4", [16, 24, 32, 36], 512),
    ]:
        probabilities[name] = fit_linear_branch(
            split_layers(train_features, layers, wanted_layers),
            split_layers(test_features, layers, wanted_layers),
            train_extra,
            test_extra,
            train_labels,
            train_clusters,
            test_clusters,
            components,
            0.03,
        )
        print(f"{name} predicted", flush=True)

    hidden_size = train_features.shape[1] // (len(layers) * 4)
    train_sequence = train_features.reshape(
        len(train),
        len(layers),
        4 * hidden_size,
    )
    test_sequence = test_features.reshape(
        len(test),
        len(layers),
        4 * hidden_size,
    )
    layer_arrays = []
    test_layer_arrays = []
    for layer_index in range(len(layers)):
        scaler = StandardScaler().fit(train_sequence[:, layer_index])
        pca = PCA(
            n_components=128,
            svd_solver="randomized",
            random_state=42,
        ).fit(scaler.transform(train_sequence[:, layer_index]))
        train_pca = pca.transform(
            scaler.transform(train_sequence[:, layer_index])
        )
        test_pca = pca.transform(
            scaler.transform(test_sequence[:, layer_index])
        )
        layer_arrays.append(
            np.hstack(
                [
                    train_pca,
                    group_relative(train_pca, train_clusters),
                ]
            ).astype(np.float32)
        )
        test_layer_arrays.append(
            np.hstack(
                [
                    test_pca,
                    group_relative(test_pca, test_clusters),
                ]
            ).astype(np.float32)
        )
    sequence = np.stack(layer_arrays, axis=1)
    test_sequence_model = np.stack(test_layer_arrays, axis=1)
    extra_scaler = StandardScaler().fit(train_extra)
    train_extra_scaled = extra_scaler.transform(train_extra).astype(
        np.float32
    )
    test_extra_scaled = extra_scaler.transform(test_extra).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    model = LayerSequenceModel(
        input_size=sequence.shape[-1],
        extra_size=train_extra.shape[1],
        hidden_size=128,
        layer_count=sequence.shape[1],
        cell_type="gru",
        bidirectional=True,
        residual=False,
        layer_delta=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(sequence),
            torch.from_numpy(train_extra_scaled),
            torch.from_numpy(train_labels).long(),
        ),
        batch_size=64,
        shuffle=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_sequence, batch_extra, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch_sequence.to(device),
                batch_extra.to(device),
            )
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0:
            print(
                f"gru7 epoch={epoch} loss={np.mean(losses):.4f}",
                flush=True,
            )
    model.eval()
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(test_sequence_model).to(device),
            torch.from_numpy(test_extra_scaled).to(device),
        )
        probabilities["gru7"] = torch.softmax(
            logits.float(),
            dim=-1,
        ).cpu().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **probabilities)
    for name, values in probabilities.items():
        print(
            f"{name}: "
            + ", ".join(
                f"{LABELS[index]}={values[0, index]:.4f}"
                for index in np.argsort(-values[0])[:3]
            )
        )
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
