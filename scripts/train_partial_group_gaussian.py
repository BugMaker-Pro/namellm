from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def softmax_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = scores / temperature
    values -= values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=1, keepdims=True)


class DiagonalGaussian:
    def __init__(self, shrinkage: float, variance_floor: float):
        self.shrinkage = shrinkage
        self.variance_floor = variance_floor
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.log_norm_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DiagonalGaussian":
        means = []
        variances = []
        for label in range(len(LABELS)):
            block = features[labels == label]
            if len(block) < 2:
                raise ValueError(f"Class {label} has fewer than two rows")
            means.append(block.mean(axis=0))
            variances.append(block.var(axis=0, ddof=1))
        means_array = np.stack(means)
        variances_array = np.stack(variances)
        pooled = variances_array.mean(axis=0, keepdims=True)
        blended = (
            (1.0 - self.shrinkage) * variances_array
            + self.shrinkage * pooled
        )
        blended = np.maximum(blended, self.variance_floor)
        self.means_ = means_array
        self.variances_ = blended
        self.log_norm_ = -0.5 * np.log(blended).sum(axis=1)
        return self

    def scores(self, features: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.variances_ is None:
            raise RuntimeError("Gaussian model is not fitted")
        x = features[:, None, :]
        diff = x - self.means_[None, :, :]
        quadratic = (diff.square() / self.variances_[None, :, :]).sum(axis=2)
        return self.log_norm_[None, :] - 0.5 * quadratic


def transform_features(
    train_features: np.ndarray,
    all_features: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    all_scaled = scaler.transform(all_features)
    max_components = min(train_scaled.shape)
    count = (
        max_components
        if components <= 0
        else min(components, max_components)
    )
    pca = PCA(
        n_components=count,
        svd_solver="randomized",
        random_state=seed,
    ).fit(train_scaled)
    output = StandardScaler().fit(pca.transform(train_scaled))
    return (
        output.transform(pca.transform(train_scaled)).astype(np.float32),
        output.transform(pca.transform(all_scaled)).astype(np.float32),
        {"scaler": scaler, "pca": pca, "output_scaler": output},
    )


def group_relative_rows(
    features: np.ndarray,
    clusters: np.ndarray,
    minimum: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = features.copy()
    eligible = np.zeros(len(features), dtype=bool)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        if minimum <= int(mask.sum()) <= maximum:
            result[mask] -= features[mask].mean(axis=0, keepdims=True)
            eligible[mask] = True
    return result, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--shrinkage", type=float, default=0.2)
    parser.add_argument("--variance-floor", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["model"].map(label_to_index).to_numpy(dtype=np.int64)
    groups = frame["prompt_id"].astype(str).to_numpy()
    texts = frame["text"].fillna("").astype(str).tolist()
    data = np.load(args.features, allow_pickle=True)
    features = data["X"].astype(np.float32)
    if "row_indices" in data.files and not np.array_equal(
        data["row_indices"].astype(np.int64),
        np.arange(len(frame)),
    ):
        raise ValueError("Feature rows do not align with the input CSV")

    probabilities = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    fold_metrics = []
    folds = list(GroupKFold(5).split(frame, labels, groups))
    for fold, (train_index, valid_index) in enumerate(folds):
        train_dense, all_dense, _ = transform_features(
            features[train_index],
            features,
            args.pca_components,
            args.seed,
        )
        gaussian = DiagonalGaussian(
            args.shrinkage,
            args.variance_floor,
        ).fit(train_dense, labels[train_index])
        valid_clusters = blind_prompt_clusters(
            [texts[index] for index in valid_index],
            0.90,
        )
        valid_relative, eligible = group_relative_rows(
            all_dense[valid_index],
            valid_clusters,
            args.min_group_size,
            args.max_group_size,
        )
        scores = gaussian.scores(all_dense[valid_index])
        relative_scores = gaussian.scores(valid_relative)
        scores[eligible] = relative_scores[eligible]
        probabilities[valid_index] = softmax_scores(
            scores,
            args.temperature,
        )
        score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        fold_metrics.append(
            {
                "fold": fold,
                "valid_score": float(score),
                "eligible_rows": int(eligible.sum()),
                "clusters": int(len(np.unique(valid_clusters))),
            }
        )
        print(
            f"fold={fold} valid={score:.4f} "
            f"group_rows={int(eligible.sum())} "
            f"clusters={len(np.unique(valid_clusters))}",
            flush=True,
        )
    oof = balanced_accuracy_score(
        labels,
        probabilities.argmax(axis=1),
    )
    print(f"partial_group_gaussian OOF={oof:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    metrics = {
        "features": args.features,
        "pca_components": args.pca_components,
        "shrinkage": args.shrinkage,
        "temperature": args.temperature,
        "group_sizes": [args.min_group_size, args.max_group_size],
        "folds": fold_metrics,
        "oof_balanced_accuracy": float(oof),
    }
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else output.with_suffix(output.suffix + ".json")
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
