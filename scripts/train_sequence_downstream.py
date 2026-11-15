from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, family  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, feature_dir: Path):
        self.storage = np.load(feature_dir / "sequences.npy", mmap_mode="r")
        metadata = np.load(feature_dir / "metadata.npz", allow_pickle=True)
        self.labels = metadata["y"]
        self.offsets = metadata["offsets"]
        self.lengths = metadata["lengths"]
        self.train_mask = metadata["train_mask"]
        self.valid_mask = metadata["valid_mask"]
        self.row_indices = metadata["row_indices"]
        self.train_indices = np.where(self.train_mask)[0]
        self.valid_indices = np.where(self.valid_mask)[0]

    def sequence(self, index: int) -> np.ndarray:
        return np.asarray(
            self.storage[
                self.offsets[index] : self.offsets[index + 1]
            ],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)


class TokenDataset(Dataset):
    def __init__(
        self,
        base: SequenceDataset,
        indices: np.ndarray,
        summaries: np.ndarray | None = None,
    ):
        self.base = base
        self.indices = indices
        self.summaries = summaries

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        summary = (
            self.summaries[index]
            if self.summaries is not None
            else np.zeros(0, dtype=np.float32)
        )
        return (
            self.base.sequence(index),
            int(self.base.labels[index]),
            index,
            summary,
        )


def collate(items):
    sequences, labels, indices, summaries = zip(*items)
    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[1]
    batch = torch.zeros((len(sequences), max_length, hidden_size), dtype=torch.float32)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        batch[row, :length] = torch.from_numpy(sequence)
        mask[row, :length] = True
    summary_batch = torch.from_numpy(np.stack(summaries))
    return batch, mask, torch.tensor(labels), torch.tensor(indices), summary_batch


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(values).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, values)


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        summary_size: int = 0,
        pooled_multiplier: int = 4,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, d_model)
        self.pooling = AttentionPooling(d_model)
        self.summary_branch = None
        combined_size = d_model * pooled_multiplier
        if summary_size:
            self.summary_branch = nn.Sequential(
                nn.LayerNorm(summary_size),
                nn.Linear(summary_size, 128),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            combined_size += 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Dropout(0.2),
            nn.Linear(combined_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(LABELS)),
        )

    def classify(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention = self.pooling(values, mask)
        masked = values.masked_fill(~mask.unsqueeze(-1), 0)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = masked.sum(dim=1) / counts
        max_value = values.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        lengths = mask.sum(dim=1) - 1
        last = values[torch.arange(values.shape[0], device=values.device), lengths]
        pooled = torch.cat([attention, mean, max_value, last], dim=-1)
        if self.summary_branch is not None:
            if summary is None:
                raise ValueError("summary features are required by this model")
            pooled = torch.cat([pooled, self.summary_branch(summary)], dim=-1)
        return self.classifier(pooled)


class BiGRUModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(input_size, d_model=256, summary_size=summary_size)
        self.rnn = nn.GRU(
            256,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        values, _ = self.rnn(values)
        return self.classify(values, mask, summary)


class CNNBiGRUModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(input_size, d_model=256, summary_size=summary_size)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(256)
        self.rnn = nn.GRU(
            256,
            128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        conv_input = values.transpose(1, 2)
        conv = torch.nn.functional.gelu(self.conv3(conv_input) + self.conv5(conv_input))
        values = self.conv_norm(values + conv.transpose(1, 2))
        values, _ = self.rnn(values)
        return self.classify(values, mask, summary)


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm1(values)
        values = self.conv1(values.transpose(1, 2))
        values = torch.nn.functional.gelu(values).transpose(1, 2)
        values = self.dropout(values)
        values = self.norm2(values)
        values = self.conv2(values.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(values)


class ResNetBiGRUModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(input_size, d_model=256, summary_size=summary_size)
        self.residual = ResidualConvBlock1D(256)
        self.rnn = nn.GRU(
            256,
            128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        values = self.residual(values)
        values = values.masked_fill(~mask.unsqueeze(-1), 0)
        values, _ = self.rnn(values)
        return self.classify(values, mask, summary)


class AttentionMILModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(
            input_size,
            d_model=128,
            summary_size=summary_size,
            pooled_multiplier=2,
        )
        self.attention_hidden = nn.Linear(128, 128)
        self.attention_gate = nn.Linear(128, 128)
        self.attention_output = nn.Linear(128, 1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        hidden = torch.tanh(self.attention_hidden(values))
        gate = torch.sigmoid(self.attention_gate(values))
        scores = self.attention_output(hidden * gate).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        bag = torch.einsum("bt,btd->bd", weights, values)
        mean = (
            values.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp_min(1)
        )
        pooled = torch.cat([bag, mean], dim=-1)
        if self.summary_branch is not None:
            if summary is None:
                raise ValueError("summary features are required by this model")
            pooled = torch.cat([pooled, self.summary_branch(summary)], dim=-1)
        return self.classifier(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_length: int = 512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]]


class TransformerModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(input_size, d_model=256, summary_size=summary_size)
        self.position = PositionalEncoding(256)
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.position(self.projection(self.norm(values)))
        values = self.encoder(values, src_key_padding_mask=~mask)
        return self.classify(values, mask, summary)


class GatedDeltaBlock(nn.Module):
    """Small gated delta-rule memory block for short style sequences."""

    def __init__(self, hidden_size: int, heads: int = 4):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.beta = nn.Linear(hidden_size, heads)
        self.gate = nn.Linear(hidden_size, heads)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = values.shape
        q = self.q(values).view(batch, length, self.heads, self.head_dim)
        k = self.k(values).view(batch, length, self.heads, self.head_dim)
        v = self.v(values).view(batch, length, self.heads, self.head_dim)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta(values))
        gate = torch.sigmoid(self.gate(values))
        state = values.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        outputs = []
        for position in range(length):
            query = q[:, position]
            key = k[:, position]
            value = v[:, position]
            prediction = torch.einsum("bhkv,bhk->bhv", state, key)
            delta = beta[:, position].unsqueeze(-1) * (value - prediction)
            updated = (
                state * gate[:, position].unsqueeze(-1).unsqueeze(-1)
                + delta.unsqueeze(-1) * key.unsqueeze(-2)
            )
            active = mask[:, position].view(batch, 1, 1, 1)
            state = torch.where(active, updated, state)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, query))
        output = torch.stack(outputs, dim=1).reshape(batch, length, hidden)
        return self.norm(values + self.output(output))


class DeltaNetModel(SequenceClassifier):
    def __init__(self, input_size: int, summary_size: int = 0):
        super().__init__(input_size, d_model=128, summary_size=summary_size)
        self.delta = GatedDeltaBlock(128, heads=4)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        values = self.delta(values, mask)
        return self.classify(values, mask, summary)


MODELS = {
    "bigru": BiGRUModel,
    "cnn_bigru": CNNBiGRUModel,
    "resnet_bigru": ResNetBiGRUModel,
    "attention_mil": AttentionMILModel,
    "transformer": TransformerModel,
    "deltanet": DeltaNetModel,
}


def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for values, mask, labels, _, summary in loader:
            logits = model(
                values.to(device),
                mask.to(device),
                summary.to(device),
            )
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
            targets.append(labels)
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [
        (predictions[targets == index] == index).mean()
        for index in range(len(LABELS))
        if (targets == index).any()
    ]
    return float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features/qwen3_seq_fold0")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--summary-features", default="")
    parser.add_argument("--summary-components", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    base = SequenceDataset(Path(args.features_dir))
    summaries = None
    if args.summary_features:
        summary_data = np.load(args.summary_features, allow_pickle=True)
        summary_features = summary_data["X"]
        summary_rows = summary_data["row_indices"]
        if not np.array_equal(summary_rows, base.row_indices):
            order = {int(row): index for index, row in enumerate(summary_rows)}
            summary_features = summary_features[
                [order[int(row)] for row in base.row_indices]
            ]
        scaler = StandardScaler().fit(summary_features[base.train_mask])
        scaled_summary = scaler.transform(summary_features)
        pca = PCA(
            n_components=args.summary_components,
            svd_solver="randomized",
            random_state=args.seed,
        ).fit(scaled_summary[base.train_mask])
        summaries = pca.transform(scaled_summary).astype(np.float32)
        print(
            f"summary PCA explained={pca.explained_variance_ratio_.sum():.4f}",
            flush=True,
        )
    train_loader = DataLoader(
        TokenDataset(base, base.train_indices, summaries),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TokenDataset(base, base.valid_indices, summaries),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    summary_size = args.summary_components if summaries is not None else 0
    model = MODELS[args.model](
        base.storage.shape[1],
        summary_size=summary_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score = -1.0
    best_state = None
    best_probabilities = None
    best_targets = None
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for values, mask, labels, _, summary in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    values.to(device),
                    mask.to(device),
                    summary.to(device),
                )
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities, targets = evaluate(model, valid_loader, device)
        score = balanced_accuracy(targets, probabilities.argmax(axis=1))
        family_score = np.mean(
            [
                family(LABELS[predicted]) == family(LABELS[target])
                for predicted, target in zip(
                    probabilities.argmax(axis=1),
                    targets,
                )
            ]
        )
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"balanced_accuracy={score:.4f} family={family_score:.4f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            best_probabilities = probabilities
            best_targets = targets
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, best_probabilities)
    result = {
        "model": args.model,
        "best_balanced_accuracy": best_score,
        "targets": best_targets.tolist(),
        "probabilities": str(output),
        "elapsed": time.time() - started,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"best_balanced_accuracy={best_score:.4f} saved={output}", flush=True)


if __name__ == "__main__":
    main()
