from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


class ClusterDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        clusters: np.ndarray,
        row_indices: np.ndarray,
    ):
        self.items = []
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            rows = row_indices[mask]
            self.items.append(
                (
                    torch.from_numpy(features[rows]),
                    torch.from_numpy(labels[rows]).long(),
                    torch.from_numpy(rows).long(),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def collate_clusters(items):
    features, labels, rows = zip(*items)
    lengths = torch.tensor([len(item) for item in features])
    mask = torch.arange(lengths.max())[None, :] < lengths[:, None]
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(labels, batch_first=True, padding_value=-1),
        pad_sequence(rows, batch_first=True, padding_value=-1),
        mask,
    )


class SetTransformerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, len(LABELS)),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        values = self.projection(features)
        values = self.encoder(values, src_key_padding_mask=~mask)
        return self.classifier(values)


def build_features(
    branches: list[np.ndarray],
    balanced: np.ndarray,
) -> np.ndarray:
    normalized = [
        branch / branch.sum(axis=1, keepdims=True)
        for branch in branches
    ]
    blocks = [*normalized]
    blocks.extend(
        np.log(np.clip(branch, 1e-12, None))
        for branch in normalized
    )
    blocks.append(balanced / balanced.sum(axis=1, keepdims=True))
    blocks.append(
        np.stack(
            [branch.max(axis=1) for branch in normalized],
            axis=1,
        )
    )
    blocks.append(
        np.stack(
            [
                -(branch * np.log(np.clip(branch, 1e-12, None))).sum(
                    axis=1
                )
                for branch in normalized
            ],
            axis=1,
        )
    )
    return np.hstack(blocks).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default="D:/tmp/namellm_set_transformer_oof.npy",
    )
    args = parser.parse_args()

    paths = [
        ROOT / "features" / "summary_funnel_power_blend.npy",
        ROOT / "features" / "qwen3_embedding_4b_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    ]
    balanced_path = (
        ROOT / "features" / "qwen3_embedding_cluster_balanced_fusion_oof.npy"
    )
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    branches = [np.load(path) for path in paths]
    features = build_features(branches, np.load(balanced_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    probabilities = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        scaler = StandardScaler().fit(features[train_index])
        scaled = scaler.transform(features).astype(np.float32)
        train_clusters = blind_prompt_clusters(
            frame["text"].iloc[train_index].tolist(),
            0.90,
        )
        valid_clusters = blind_prompt_clusters(
            frame["text"].iloc[valid_index].tolist(),
            0.90,
        )
        train_dataset = ClusterDataset(
            scaled,
            labels,
            train_clusters,
            train_index,
        )
        valid_dataset = ClusterDataset(
            scaled,
            labels,
            valid_clusters,
            valid_index,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_clusters,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_clusters,
        )

        model = SetTransformerModel(features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch_features, batch_labels, _, mask in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features, mask)
                loss = criterion(logits[mask], batch_labels[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch % 20 == 0:
                print(
                    f"fold={fold} epoch={epoch} "
                    f"train_loss={np.mean(losses):.4f}",
                    flush=True,
                )

        model.eval()
        with torch.inference_mode():
            for batch_features, _, batch_rows, mask in valid_loader:
                logits = model(batch_features.to(device), mask.to(device))
                batch_probabilities = torch.softmax(
                    logits.float(),
                    dim=-1,
                ).cpu().numpy()
                for item_index in range(batch_rows.shape[0]):
                    valid_positions = batch_rows[item_index][
                        batch_rows[item_index] >= 0
                    ].numpy()
                    probabilities[valid_positions] = batch_probabilities[
                        item_index,
                        : len(valid_positions),
                    ]
        score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        print(f"fold={fold} valid={score:.4f}", flush=True)

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"set_transformer OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
