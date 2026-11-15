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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def center_log_probabilities(
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


class FusionMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class FusionGate(nn.Module):
    def __init__(self, input_size: int, branch_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, branch_count),
        )

    def forward(
        self,
        features: torch.Tensor,
        branch_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.softmax(self.layers(features), dim=-1)
        return torch.einsum("bn,bnc->bc", weights, branch_probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlp", "gate"], required=True)
    parser.add_argument("--center-alpha", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [
        ROOT / "features" / "summary_funnel_power_blend.npy",
        ROOT / "features" / "qwen3_embedding_4b_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    ]
    current_path = (
        ROOT / "features" / "qwen3_embedding_prompt_centered_fusion_oof.npy"
    )
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    branches = [
        np.load(path) / np.load(path).sum(axis=1, keepdims=True)
        for path in paths
    ]
    centered = [np.zeros_like(branch) for branch in branches]
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    for _, valid_index in folds:
        texts = frame["text"].iloc[valid_index].tolist()
        for index, branch in enumerate(branches):
            centered[index][valid_index] = center_log_probabilities(
                branch[valid_index],
                texts,
                args.center_alpha,
            )

    feature_blocks = [
        *branches,
        *centered,
        np.stack(
            [
                branch.max(axis=1)
                for branch in branches
            ],
            axis=1,
        ),
        np.stack(
            [
                -(
                    branch
                    * np.log(np.clip(branch, 1e-12, None))
                ).sum(axis=1)
                for branch in branches
            ],
            axis=1,
        ),
    ]
    if current_path.exists():
        feature_blocks.append(np.load(current_path))
    features = np.hstack(feature_blocks).astype(np.float32)

    gate_branches = np.stack(branches, axis=1)
    if current_path.exists():
        current = np.load(current_path)
        current /= current.sum(axis=1, keepdims=True)
        gate_branches = np.concatenate(
            [gate_branches, current[:, None, :]],
            axis=1,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        scaler = StandardScaler().fit(features[train_index])
        train_features = scaler.transform(features[train_index]).astype(
            np.float32
        )
        valid_features = scaler.transform(features[valid_index]).astype(
            np.float32
        )
        train_branches = gate_branches[train_index].astype(np.float32)
        valid_branches = gate_branches[valid_index].astype(np.float32)
        train_dataset = TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_branches),
            torch.from_numpy(labels[train_index]).long(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        if args.mode == "mlp":
            model = FusionMLP(features.shape[1]).to(device)
        else:
            model = FusionGate(
                features.shape[1],
                gate_branches.shape[1],
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for _ in range(args.epochs):
            model.train()
            for batch_features, batch_branches, batch_labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if args.mode == "mlp":
                    logits = model(batch_features.to(device))
                else:
                    logits = model(
                        batch_features.to(device),
                        batch_branches.to(device),
                    )
                loss = criterion(logits, batch_labels.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.inference_mode():
            batch_features = torch.from_numpy(valid_features).to(device)
            batch_branches = torch.from_numpy(valid_branches).to(device)
            if args.mode == "mlp":
                logits = model(batch_features)
            else:
                logits = model(batch_features, batch_branches)
            fold_probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probabilities[valid_index] = fold_probabilities
        score = balanced_accuracy_score(
            labels[valid_index],
            fold_probabilities.argmax(axis=1),
        )
        print(f"fold={fold} valid={score:.4f}", flush=True)

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"{args.mode}_fusion OOF={score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
