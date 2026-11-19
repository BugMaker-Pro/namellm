from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV, family  # noqa: E402
from namellm.style import style_frame  # noqa: E402


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
        self.prompt_ids = metadata["prompt_ids"].astype(str)
        self.prompt_codes = np.unique(
            self.prompt_ids,
            return_inverse=True,
        )[1]
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


class QueryStyleDataset(TokenDataset):
    def __init__(
        self,
        base: SequenceDataset,
        indices: np.ndarray,
        summaries: np.ndarray | None,
        styles: np.ndarray | None,
    ):
        super().__init__(base, indices, summaries)
        self.styles = styles

    def __getitem__(self, item: int):
        values, label, index, summary = super().__getitem__(item)
        style = (
            self.styles[index]
            if self.styles is not None
            else np.zeros(0, dtype=np.float32)
        )
        return (
            values,
            label,
            index,
            summary,
            style,
            int(self.base.prompt_codes[index]),
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


def collate_query_style(items):
    sequences, labels, indices, summaries, styles, prompts = zip(*items)
    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[1]
    batch = torch.zeros((len(sequences), max_length, hidden_size), dtype=torch.float32)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        batch[row, :length] = torch.from_numpy(sequence)
        mask[row, :length] = True
    return (
        batch,
        mask,
        torch.tensor(labels),
        torch.tensor(indices),
        torch.from_numpy(np.stack(summaries)),
        torch.from_numpy(np.stack(styles)),
        torch.tensor(prompts),
    )


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


class QueryFlowModel(nn.Module):
    """Read a passage through a small bank of task-specific query tokens."""

    def __init__(
        self,
        input_size: int,
        summary_size: int = 0,
        d_model: int = 128,
        query_count: int = 16,
        query_layers: int = 2,
        query_heads: int = 4,
        query_dropout: float = 0.2,
        query_causal: bool = True,
        style_size: int = 0,
        prompt_size: int = 0,
    ):
        super().__init__()
        if d_model % query_heads:
            raise ValueError("d_model must be divisible by query_heads")
        self.query_count = query_count
        self.query_causal = query_causal
        self.norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, d_model)
        self.query_dropout = nn.Dropout(query_dropout)
        self.query_bank = nn.Parameter(
            torch.randn(query_count, d_model) * 0.02
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=query_heads,
            dim_feedforward=d_model * 4,
            dropout=query_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=query_layers,
            norm=nn.LayerNorm(d_model),
        )
        combined_size = d_model * 3
        self.summary_branch = None
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
        self.style_head = None
        if style_size:
            self.style_head = nn.Sequential(
                nn.LayerNorm(combined_size),
                nn.Linear(combined_size, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, style_size),
            )
        self.prompt_head = None
        if prompt_size:
            self.prompt_head = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, prompt_size),
            )

    def forward_features(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.projection(self.norm(values))
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0)
        queries = self.query_bank.unsqueeze(0).expand(
            values.shape[0],
            -1,
            -1,
        )
        queries = self.query_dropout(queries)
        query_mask = None
        if self.query_causal:
            query_mask = torch.triu(
                torch.ones(
                    self.query_count,
                    self.query_count,
                    dtype=torch.bool,
                    device=values.device,
                ),
                diagonal=1,
            )
        decoded = self.decoder(
            queries,
            memory,
            tgt_mask=query_mask,
            memory_key_padding_mask=~mask,
        )
        query_features = torch.cat(
            [
                decoded.mean(dim=1),
                decoded.max(dim=1).values,
                decoded[:, -1],
            ],
            dim=-1,
        )
        pooled = query_features
        if self.summary_branch is not None:
            if summary is None:
                raise ValueError("summary features are required by this model")
            pooled = torch.cat([pooled, self.summary_branch(summary)], dim=-1)
        return pooled, query_features

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features, _ = self.forward_features(values, mask, summary)
        return self.classifier(features)

    def forward_aux(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        features, query_features = self.forward_features(
            values,
            mask,
            summary,
        )
        style = (
            self.style_head(features)
            if self.style_head is not None
            else None
        )
        return self.classifier(features), style, query_features


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(context, values: torch.Tensor, scale: float) -> torch.Tensor:
        context.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(context, gradient: torch.Tensor):
        return -context.scale * gradient, None


class PatchLateInteractionModel(SequenceClassifier):
    def __init__(
        self,
        input_size: int,
        summary_size: int = 0,
        patch_size: int = 8,
        prototype_count: int = 8,
    ):
        super().__init__(
            input_size,
            d_model=192,
            summary_size=summary_size,
            pooled_multiplier=4,
        )
        self.patch_size = patch_size
        self.prototype_count = prototype_count
        self.conv = nn.Conv1d(
            192,
            192,
            kernel_size=3,
            padding=1,
        )
        self.patch_norm = nn.LayerNorm(192)
        self.prototypes = nn.Parameter(
            torch.randn(
                len(LABELS),
                prototype_count,
                192,
            )
            * 0.02
        )
        self.scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.late_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.projection(self.norm(values))
        residual = values
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        values = self.patch_norm(residual + torch.nn.functional.gelu(values))
        if values.shape[1] < self.patch_size:
            pad = self.patch_size - values.shape[1]
            values = torch.nn.functional.pad(values, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        patches = values.unfold(
            1,
            self.patch_size,
            self.patch_size,
        ).mean(dim=-1)
        patch_mask = mask.unfold(
            1,
            self.patch_size,
            self.patch_size,
        ).any(dim=-1)
        global_logits = self.classify(patches, patch_mask, summary)
        patch_vectors = torch.nn.functional.normalize(patches, dim=-1)
        prototypes = torch.nn.functional.normalize(
            self.prototypes,
            dim=-1,
        )
        similarity = torch.einsum(
            "bpd,ckd->bpck",
            patch_vectors,
            prototypes,
        )
        similarity = similarity * self.scale.clamp(max=100.0).exp()
        similarity = similarity.masked_fill(
            ~patch_mask[:, :, None, None],
            -1e4,
        )
        late_scores = similarity.max(dim=1).values.mean(dim=-1)
        return global_logits + self.late_weight * late_scores


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
    "query_flow": QueryFlowModel,
    "patch_late": PatchLateInteractionModel,
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
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--prototype-count", type=int, default=8)
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--query-layers", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--query-dropout", type=float, default=0.2)
    parser.add_argument("--style-weight", type=float, default=0.0)
    parser.add_argument("--prompt-adv-weight", type=float, default=0.0)
    parser.add_argument("--prompt-adv-scale", type=float, default=1.0)
    parser.add_argument(
        "--query-causal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.style_weight > 0 or args.prompt_adv_weight > 0
    ) and args.model != "query_flow":
        parser.error(
            "--style-weight and --prompt-adv-weight are only supported "
            "by query_flow"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    base = SequenceDataset(Path(args.features_dir))
    summaries = None
    query_styles = None
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
    if args.style_weight > 0:
        frame = pd.read_csv(TRAIN_CSV)
        raw_styles = style_frame(
            frame["text"].fillna("").astype(str)
        ).to_numpy(np.float32)
        raw_styles = raw_styles[base.row_indices]
        style_scaler = StandardScaler().fit(raw_styles[base.train_mask])
        query_styles = style_scaler.transform(raw_styles).astype(np.float32)
    use_query_aux = (
        args.style_weight > 0 or args.prompt_adv_weight > 0
    )
    train_loader = DataLoader(
        (
            QueryStyleDataset(
                base,
                base.train_indices,
                summaries,
                query_styles,
            )
            if use_query_aux
            else TokenDataset(base, base.train_indices, summaries)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=(
            collate_query_style
            if use_query_aux
            else collate
        ),
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
    if args.model == "patch_late":
        model = PatchLateInteractionModel(
            base.storage.shape[1],
            summary_size=summary_size,
            patch_size=args.patch_size,
            prototype_count=args.prototype_count,
        ).to(device)
    elif args.model == "query_flow":
        model = QueryFlowModel(
            base.storage.shape[1],
            summary_size=summary_size,
            query_count=args.query_count,
            query_layers=args.query_layers,
            query_heads=args.query_heads,
            query_dropout=args.query_dropout,
            query_causal=args.query_causal,
            style_size=(
                query_styles.shape[1]
                if query_styles is not None
                else 0
            ),
            prompt_size=(
                int(base.prompt_codes.max()) + 1
                if args.prompt_adv_weight > 0
                else 0
            ),
        ).to(device)
    else:
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
        for batch in train_loader:
            values, mask, labels, _, summary = batch[:5]
            style = batch[5] if len(batch) > 5 else None
            prompts = batch[6] if len(batch) > 6 else None
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if style is not None:
                    logits, style_prediction, query_features = model.forward_aux(
                        values.to(device),
                        mask.to(device),
                        summary.to(device),
                    )
                else:
                    logits = model(
                        values.to(device),
                        mask.to(device),
                        summary.to(device),
                    )
                    style_prediction = None
                    query_features = None
                loss = criterion(logits, labels)
                if style_prediction is not None:
                    loss = loss + args.style_weight * torch.nn.functional.mse_loss(
                        style_prediction.float(),
                        style.to(device).float(),
                    )
                if (
                    query_features is not None
                    and prompts is not None
                    and model.prompt_head is not None
                    and args.prompt_adv_weight > 0
                ):
                    reversed_features = GradientReverse.apply(
                        query_features,
                        args.prompt_adv_scale,
                    )
                    prompt_logits = model.prompt_head(reversed_features)
                    loss = loss + args.prompt_adv_weight * criterion(
                        prompt_logits,
                        prompts.to(device),
                    )
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
        "model_config": {
            "query_count": args.query_count,
            "query_layers": args.query_layers,
            "query_heads": args.query_heads,
            "query_dropout": args.query_dropout,
            "query_causal": args.query_causal,
            "style_weight": args.style_weight,
            "prompt_adv_weight": args.prompt_adv_weight,
            "prompt_adv_scale": args.prompt_adv_scale,
        },
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
