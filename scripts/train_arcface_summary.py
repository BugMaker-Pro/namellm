from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402


class ArcFaceHead(nn.Module):
    def __init__(self, input_size: int, embedding_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, embedding_size),
        )
        self.weight = nn.Parameter(torch.randn(len(LABELS), embedding_size))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.nn.functional.normalize(self.encoder(values), dim=-1)
        weights = torch.nn.functional.normalize(self.weight, dim=-1)
        cosine = embedding @ weights.T
        target = cosine[torch.arange(len(cosine)), self.targets]
        cosine[torch.arange(len(cosine)), self.targets] = (
            torch.cos(torch.acos(target.clamp(-1 + 1e-7, 1 - 1e-7)) + 0.2)
        )
        return cosine * 20.0, embedding

    def set_targets(self, targets: torch.Tensor) -> None:
        self.targets = targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-features",
        default="features/qwen3_token_features_fold0.npz",
    )
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    data = np.load(args.summary_features, allow_pickle=True)
    summary = data["X"]
    labels = data["y"]
    train_mask = data["train_mask"]
    valid_mask = data["valid_mask"]
    row_indices = data["row_indices"]

    frame = pd.read_csv(TRAIN_CSV)
    style = style_frame(frame["text"]).to_numpy(dtype=np.float32)
    lm = (
        pd.read_parquet(args.lm_features)
        .query("split == 'train'")
        .set_index("id")
        .loc[frame["id"]]
        .drop(columns="split")
        .to_numpy(dtype=np.float32)
    )
    clusters = blind_prompt_clusters(frame["text"].tolist(), 0.90)
    extra = np.hstack(
        [
            style,
            group_relative(style, clusters),
            lm,
            group_relative(lm, clusters),
        ]
    )[row_indices]

    scaler = StandardScaler().fit(summary[train_mask])
    train_summary = scaler.transform(summary[train_mask])
    valid_summary = scaler.transform(summary[valid_mask])
    pca = PCA(n_components=512, svd_solver="randomized", random_state=args.seed)
    pca.fit(train_summary)
    train_features = np.hstack([pca.transform(train_summary), extra[train_mask]])
    valid_features = np.hstack([pca.transform(valid_summary), extra[valid_mask]])
    feature_scaler = StandardScaler().fit(train_features)
    train_features = feature_scaler.transform(train_features).astype(np.float32)
    valid_features = feature_scaler.transform(valid_features).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(labels[train_mask]).long(),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(valid_features),
        torch.from_numpy(labels[valid_mask]).long(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    model = ArcFaceHead(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score = -1.0
    best_probabilities = None
    best_targets = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for values, targets in train_loader:
            values = values.to(device)
            targets = targets.to(device)
            model.set_targets(targets)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        probabilities = []
        targets_list = []
        with torch.inference_mode():
            for values, targets in valid_loader:
                values = values.to(device)
                targets = targets.to(device)
                model.set_targets(targets)
                logits, _ = model(values)
                probabilities.append(torch.softmax(logits, dim=-1).cpu())
                targets_list.append(targets.cpu())
        probability_array = torch.cat(probabilities).numpy()
        target_array = torch.cat(targets_list).numpy()
        recalls = [
            (
                probability_array[target_array == index].argmax(axis=1)
                == index
            ).mean()
            for index in range(len(LABELS))
        ]
        score = float(np.mean(recalls))
        if score > best_score:
            best_score = score
            best_probabilities = probability_array
            best_targets = target_array
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch={epoch} loss={np.mean(losses):.4f} "
                f"balanced_accuracy={score:.4f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, best_probabilities)
    print(f"best_balanced_accuracy={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()

