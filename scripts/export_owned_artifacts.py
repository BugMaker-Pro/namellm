from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from predict_embedding_branches import (  # noqa: E402
    extra_features,
    split_layers,
)
from predict_submission import (  # noqa: E402
    center_by_texts,
    rich_features,
)
from train_layer_sequence import LayerSequenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="D:/tmp/namellm_owned_artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in train["model"]])

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
        dtype=np.float32,
    ).fit(train["text"])
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    ).fit(train["text"])
    compact_features = hstack(
        [char.transform(train["text"]), word.transform(train["text"])]
    ).tocsr()
    compact_model = LogisticRegression(C=3.0, max_iter=3000)
    compact_model.fit(compact_features, labels)
    joblib.dump(
        {
            "char": char,
            "word": word,
            "model": compact_model,
        },
        output_dir / "compact_tfidf.joblib",
        compress=3,
    )

    train_data = np.load(
        ROOT / "features" / "qwen3_embedding_4b_layers7.npz",
        allow_pickle=True,
    )
    layers = [int(value) for value in train_data["layers"]]
    train_features = train_data["X"].astype(np.float32)
    lm_frame = pd.read_parquet(
        ROOT / "features" / "e002_qwen3-4b-base-4bit.parquet"
    )
    lm_train = lm_frame.query("split == 'train'")
    train_clusters = blind_prompt_clusters(train["text"].tolist(), 0.90)
    train_extra = extra_features(train, train_clusters, lm_train)

    linear_artifacts = {}
    for name, wanted_layers, components in [
        ("e512", [24, 36], 512),
        ("e256", [24, 36], 256),
        ("layer4", [16, 24, 32, 36], 512),
    ]:
        selected = split_layers(train_features, layers, wanted_layers)
        scaler = StandardScaler().fit(selected)
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=42,
        ).fit(scaler.transform(selected))
        pca_train = pca.transform(scaler.transform(selected))
        combined_train = np.hstack(
            [
                pca_train,
                group_relative(pca_train, train_clusters),
                train_extra,
            ]
        )
        combined_scaler = StandardScaler().fit(combined_train)
        classifier = LogisticRegression(C=0.03, max_iter=5000)
        classifier.fit(
            combined_scaler.transform(combined_train),
            labels,
        )
        linear_artifacts[name] = {
            "wanted_layers": wanted_layers,
            "scaler": scaler,
            "pca": pca,
            "combined_scaler": combined_scaler,
            "classifier": classifier,
        }
        print(f"exported {name}", flush=True)
    joblib.dump(
        linear_artifacts,
        output_dir / "embedding_linear.joblib",
        compress=3,
    )

    hidden_size = train_features.shape[1] // (len(layers) * 4)
    sequence_features = train_features.reshape(
        len(train),
        len(layers),
        4 * hidden_size,
    )
    layer_transforms = []
    sequence_arrays = []
    for layer_index in range(len(layers)):
        scaler = StandardScaler().fit(sequence_features[:, layer_index])
        pca = PCA(
            n_components=128,
            svd_solver="randomized",
            random_state=42,
        ).fit(scaler.transform(sequence_features[:, layer_index]))
        pca_train = pca.transform(
            scaler.transform(sequence_features[:, layer_index])
        )
        layer_transforms.append({"scaler": scaler, "pca": pca})
        sequence_arrays.append(
            np.hstack(
                [
                    pca_train,
                    group_relative(pca_train, train_clusters),
                ]
            ).astype(np.float32)
        )
    sequence = np.stack(sequence_arrays, axis=1)
    extra_scaler = StandardScaler().fit(train_extra)
    extra_scaled = extra_scaler.transform(train_extra).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    sequence_model = LayerSequenceModel(
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
        sequence_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(sequence),
            torch.from_numpy(extra_scaled),
            torch.from_numpy(labels).long(),
        ),
        batch_size=64,
        shuffle=True,
    )
    for epoch in range(1, 61):
        sequence_model.train()
        losses = []
        for batch_sequence, batch_extra, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = sequence_model(
                batch_sequence.to(device),
                batch_extra.to(device),
            )
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sequence_model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0:
            print(
                f"gru7 epoch={epoch} loss={np.mean(losses):.4f}",
                flush=True,
            )
    torch.save(
        {
            "state_dict": sequence_model.state_dict(),
            "layers": layers,
            "cell_type": "gru",
            "bidirectional": True,
            "hidden_size": 128,
        },
        output_dir / "gru7_model.pt",
    )
    joblib.dump(
        {
            "layer_transforms": layer_transforms,
            "extra_scaler": extra_scaler,
        },
        output_dir / "gru7_transforms.joblib",
        compress=3,
    )
    print("exported gru7", flush=True)

    train_sequence_features = train_features.reshape(
        len(train),
        len(layers),
        4 * hidden_size,
    )
    gru_train_sequence = np.stack(
        [
            np.hstack(
                [
                    transform["pca"].transform(
                        transform["scaler"].transform(
                            train_sequence_features[:, index]
                        )
                    ),
                    group_relative(
                        transform["pca"].transform(
                            transform["scaler"].transform(
                                train_sequence_features[:, index]
                            )
                        ),
                        train_clusters,
                    ),
                ]
            )
            for index, transform in enumerate(layer_transforms)
        ],
        axis=1,
    ).astype(np.float32)
    sequence_model.eval()
    with torch.inference_mode():
        logits = sequence_model(
            torch.from_numpy(gru_train_sequence).to(device),
            torch.from_numpy(extra_scaled).to(device),
        )
        gru_oof_full = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    compact_oof = np.load(
        ROOT / "features" / "compact_baseline_oof.npy"
    )
    branch_oof = [
        compact_oof,
        np.load(ROOT / "features" / "qwen3_embedding_4b_oof.npy"),
        np.load(ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy"),
        np.load(ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy"),
        np.load(ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy"),
    ]
    centered_train = [np.zeros_like(branch) for branch in branch_oof]
    folds = list(GroupKFold(5).split(train, labels, train["prompt_id"]))
    for _, valid_index in folds:
        texts = train["text"].iloc[valid_index].tolist()
        for index, branch in enumerate(branch_oof):
            centered_train[index][valid_index] = center_by_texts(
                branch[valid_index],
                texts,
            )
    stack_features = rich_features(branch_oof, centered_train)
    stack_scaler = StandardScaler().fit(stack_features)
    stack_model = LogisticRegression(C=0.003, max_iter=5000)
    stack_model.fit(stack_scaler.transform(stack_features), labels)
    joblib.dump(
        {"scaler": stack_scaler, "model": stack_model},
        output_dir / "rich_stack.joblib",
        compress=3,
    )
    print("exported rich stack", flush=True)

    for code_file in [
        ROOT / "src" / "namellm" / "clustering.py",
        ROOT / "src" / "namellm" / "constants.py",
        ROOT / "src" / "namellm" / "style.py",
        ROOT / "src" / "namellm" / "lm_stats.py",
        ROOT / "scripts" / "run_owned_kaggle_inference.py",
    ]:
        if code_file.exists():
            shutil.copy2(code_file, output_dir / code_file.name)
    print(f"saved artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
