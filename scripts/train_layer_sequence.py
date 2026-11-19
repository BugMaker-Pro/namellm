from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_sequence_downstream import GatedDeltaBlock  # noqa: E402


class LayerSequenceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        extra_size: int,
        hidden_size: int,
        layer_count: int,
        cell_type: str,
        bidirectional: bool,
        residual: bool,
        layer_delta: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.residual = residual
        self.layer_delta = layer_delta
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.position = nn.Parameter(torch.zeros(1, layer_count, hidden_size))
        if cell_type == "deltanet":
            if bidirectional:
                raise ValueError("DeltaNet layer sequence must be causal")
            self.encoder = None
            self.delta_block = GatedDeltaBlock(hidden_size, heads=4)
            self.output_size = hidden_size
        else:
            recurrent_class = nn.GRU if cell_type == "gru" else nn.LSTM
            self.encoder = recurrent_class(
                hidden_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.delta_block = None
            self.output_size = hidden_size * (2 if bidirectional else 1)
        self.residual_projection = (
            nn.Linear(hidden_size, self.output_size)
            if residual
            else None
        )
        classifier_input = (
            hidden_size * 6 + extra_size
            if bidirectional
            else hidden_size * 3 + extra_size
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(0.3),
            nn.Linear(classifier_input, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(LABELS)),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.input_projection(sequence) + self.position
        if self.layer_delta:
            delta = torch.zeros_like(encoded)
            delta[:, 1:] = encoded[:, 1:] - encoded[:, :-1]
            encoded = encoded + delta
        if self.delta_block is not None:
            sequence_output = self.delta_block(
                encoded,
                torch.ones(
                    encoded.shape[:2],
                    dtype=torch.bool,
                    device=encoded.device,
                ),
            )
        else:
            sequence_output, _ = self.encoder(encoded)
            if self.residual_projection is not None:
                sequence_output = (
                    sequence_output
                    + self.residual_projection(encoded)
                )
        if self.bidirectional:
            forward_last = sequence_output[:, -1, : self.hidden_size]
            backward_last = sequence_output[
                :,
                0,
                self.hidden_size :,
            ]
            pooled = torch.cat(
                [
                    forward_last,
                    backward_last,
                    sequence_output.mean(dim=1),
                    sequence_output.max(dim=1).values,
                ],
                dim=-1,
            )
        else:
            pooled = torch.cat(
                [
                    sequence_output[:, -1],
                    sequence_output.mean(dim=1),
                    sequence_output.max(dim=1).values,
                ],
                dim=-1,
            )
        return self.classifier(torch.cat([pooled, extra], dim=-1))


class LayerTransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        extra_size: int,
        hidden_size: int,
        layer_count: int,
        transformer_layers: int,
        heads: int,
        causal: bool,
        layer_delta: bool,
    ):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.causal = causal
        self.layer_delta = layer_delta
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.layer_embedding = nn.Parameter(
            torch.randn(1, layer_count, hidden_size) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        classifier_input = hidden_size * 3 + extra_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(0.3),
            nn.Linear(classifier_input, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(LABELS)),
        )

    def forward(self, sequence: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(sequence) + self.layer_embedding
        if self.layer_delta:
            delta = torch.zeros_like(encoded)
            delta[:, 1:] = encoded[:, 1:] - encoded[:, :-1]
            encoded = encoded + delta
        mask = None
        if self.causal:
            mask = torch.triu(
                torch.ones(
                    encoded.shape[1],
                    encoded.shape[1],
                    dtype=torch.bool,
                    device=encoded.device,
                ),
                diagonal=1,
            )
        output = self.encoder(encoded, mask=mask)
        pooled = torch.cat(
            [
                output.mean(dim=1),
                output.max(dim=1).values,
                output[:, -1],
            ],
            dim=-1,
        )
        return self.classifier(torch.cat([pooled, extra], dim=-1))


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for sequence, extra, labels in loader:
            logits = model(sequence.to(device), extra.to(device)).float()
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.tolist())
    return (
        np.asarray(targets),
        np.asarray(predictions),
        np.concatenate(probabilities),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--pca-per-layer", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--cell",
        choices=["gru", "lstm", "deltanet", "transformer"],
        default="gru",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--direction",
        choices=["bidirectional", "causal"],
        default="bidirectional",
    )
    parser.add_argument("--no-group-relative", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--layer-delta", action="store_true")
    parser.add_argument(
        "--output",
        default="D:/tmp/namellm_layer_sequence_oof.npy",
    )
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    data = np.load(args.features, allow_pickle=True)
    layers = [int(value) for value in data["layers"]]
    raw_features = data["X"].astype(np.float32)
    hidden_size = raw_features.shape[1] // (len(layers) * 4)
    sequence_features = raw_features.reshape(
        len(frame),
        len(layers),
        4 * hidden_size,
    )

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
    extra_parts = [style, lm]
    if not args.no_group_relative:
        extra_parts.extend(
            [
                group_relative(style, clusters),
                group_relative(lm, clusters),
            ]
        )
    extra = np.hstack(extra_parts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    probabilities = np.zeros((len(frame), len(LABELS)))
    for fold, (train_index, valid_index) in enumerate(folds):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        layer_arrays: list[np.ndarray] = []
        for layer_index in range(len(layers)):
            layer_values = sequence_features[:, layer_index]
            scaler = StandardScaler().fit(layer_values[train_index])
            train_scaled = scaler.transform(layer_values[train_index])
            pca = PCA(
                n_components=args.pca_per_layer,
                svd_solver="randomized",
                random_state=42,
            ).fit(train_scaled)
            pca_all = pca.transform(scaler.transform(layer_values))
            if args.no_group_relative:
                layer_arrays.append(pca_all.astype(np.float32))
            else:
                relative = group_relative(pca_all, clusters)
                layer_arrays.append(
                    np.hstack([pca_all, relative]).astype(np.float32)
                )
        sequence = np.stack(layer_arrays, axis=1)
        extra_scaler = StandardScaler().fit(extra[train_index])
        extra_scaled = extra_scaler.transform(extra).astype(np.float32)

        train_dataset = TensorDataset(
            torch.from_numpy(sequence[train_index]),
            torch.from_numpy(extra_scaled[train_index]),
            torch.from_numpy(labels[train_index]).long(),
        )
        valid_dataset = TensorDataset(
            torch.from_numpy(sequence[valid_index]),
            torch.from_numpy(extra_scaled[valid_index]),
            torch.from_numpy(labels[valid_index]).long(),
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

        if args.cell == "transformer":
            model = LayerTransformerModel(
                input_size=sequence.shape[-1],
                extra_size=extra.shape[1],
                hidden_size=args.hidden_size,
                layer_count=sequence.shape[1],
                transformer_layers=args.transformer_layers,
                heads=args.heads,
                causal=args.direction == "causal",
                layer_delta=args.layer_delta,
            ).to(device)
        else:
            model = LayerSequenceModel(
                input_size=sequence.shape[-1],
                extra_size=extra.shape[1],
                hidden_size=args.hidden_size,
                layer_count=sequence.shape[1],
                cell_type=args.cell,
                bidirectional=args.direction == "bidirectional",
                residual=args.residual,
                layer_delta=args.layer_delta,
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            for batch_sequence, batch_extra, batch_labels in train_loader:
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
                    f"fold={fold} epoch={epoch} "
                    f"train_loss={np.mean(losses):.4f}",
                    flush=True,
                )

        _, _, fold_probabilities = predict(model, valid_loader, device)
        probabilities[valid_index] = fold_probabilities
        score = balanced_accuracy_score(
            labels[valid_index],
            fold_probabilities.argmax(axis=1),
        )
        print(f"fold={fold} valid={score:.4f}", flush=True)

    score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(
        f"layer_sequence OOF={score:.4f} "
        f"cell={args.cell} direction={args.direction} "
        f"residual={args.residual} layer_delta={args.layer_delta}",
        flush=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
