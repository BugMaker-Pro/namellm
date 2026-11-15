from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402
from train_style_branches import (  # noqa: E402
    build_design,
    fit_transform_model,
)


def align_embedding(
    data: np.ndarray,
    frame: pd.DataFrame,
    path: Path,
) -> np.ndarray:
    embedding = data["X"].astype(np.float32)
    if "ids" in data.files:
        saved_ids = data["ids"].astype(str)
        expected_ids = frame["id"].astype(str).to_numpy()
        if not np.array_equal(saved_ids, expected_ids):
            raise ValueError(f"Embedding ids do not align for {path}")
    elif "row_indices" in data.files:
        row_indices = data["row_indices"].astype(np.int64)
        if not np.array_equal(row_indices, np.arange(len(frame))):
            raise ValueError(f"Embedding rows do not align for {path}")
    if len(embedding) != len(frame):
        raise ValueError(
            f"Embedding rows {len(embedding)} != frame rows {len(frame)}"
        )
    return embedding


def select_c_external(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    outer_train_official: np.ndarray,
    pool_indices: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[float, float]:
    inner_folds = list(
        GroupKFold(4).split(
            outer_train_official,
            labels[outer_train_official],
            groups[outer_train_official],
        )
    )
    best: tuple[float, float] | None = None
    for candidate in c_values:
        scores = []
        for inner_train, inner_valid in inner_folds:
            inner_official_train = outer_train_official[inner_train]
            inner_valid_indices = outer_train_official[inner_valid]
            fit_indices = np.concatenate(
                [inner_official_train, pool_indices]
            )
            model = LogisticRegression(
                C=candidate,
                max_iter=5000,
                random_state=seed,
            )
            model.fit(
                features[fit_indices],
                labels[fit_indices],
                sample_weight=weights[fit_indices],
            )
            scores.append(
                balanced_accuracy_score(
                    labels[inner_valid_indices],
                    model.predict(features[inner_valid_indices]),
                )
            )
        score = float(np.mean(scores))
        if best is None or score > best[1]:
            best = (candidate, score)
    if best is None:
        raise RuntimeError("C selection produced no candidate")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--pool-features", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--pool-weight", type=float, default=0.35)
    parser.add_argument("--c-values", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--use-style", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()
    print("loading train, pool, and embeddings", flush=True)

    train = pd.read_csv(args.input_csv)
    train["text"] = train["text"].fillna("").astype(str)
    pool = pd.read_parquet(args.pool)
    pool["text"] = pool["text"].fillna("").astype(str)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    train_labels = train["model"].map(label_to_index).to_numpy(dtype=np.int64)
    pool_labels = pool["model"].map(label_to_index).to_numpy(dtype=np.int64)
    if np.isnan(train_labels).any() or np.isnan(pool_labels).any():
        raise ValueError("Unknown labels in train or external pool")
    train_groups = train["prompt_id"].astype(str).to_numpy()
    train_data = np.load(args.train_features, allow_pickle=True)
    pool_data = np.load(args.pool_features, allow_pickle=True)
    train_embedding = align_embedding(
        train_data,
        train,
        Path(args.train_features),
    )
    pool_embedding = align_embedding(
        pool_data,
        pool,
        Path(args.pool_features),
    )
    train_clusters = np.zeros(len(train), dtype=np.int32)
    pool_clusters = np.zeros(len(pool), dtype=np.int32)
    train_style = (
        style_frame(train["text"]).to_numpy(dtype=np.float32)
        if args.use_style
        else np.empty((len(train), 0), dtype=np.float32)
    )
    pool_style = (
        style_frame(pool["text"]).to_numpy(dtype=np.float32)
        if args.use_style
        else np.empty((len(pool), 0), dtype=np.float32)
    )
    train_design, train_extra = build_design(
        train_embedding,
        train_style,
        train_clusters,
        args.use_style,
        False,
    )
    pool_design, pool_extra = build_design(
        pool_embedding,
        pool_style,
        pool_clusters,
        args.use_style,
        False,
    )
    embedding = np.vstack([train_design, pool_design])
    extra = np.vstack([train_extra, pool_extra])
    labels = np.concatenate([train_labels, pool_labels])
    weights = np.concatenate(
        [
            np.ones(len(train), dtype=np.float32),
            np.full(len(pool), args.pool_weight, dtype=np.float32),
        ]
    )
    groups = np.concatenate(
        [
            train_groups,
            pool["prompt_id"].astype(str).to_numpy(),
        ]
    )
    official_count = len(train)
    pool_indices = np.arange(official_count, len(labels), dtype=np.int64)
    print(
        f"train={len(train)} pool={len(pool)} "
        f"embedding={embedding.shape} extra={extra.shape}",
        flush=True,
    )

    folds = list(
        GroupKFold(5).split(train, train_labels, train_groups)
    )
    probabilities = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    fold_metrics = []
    c_values = [
        float(value) for value in args.c_values.split(",") if value
    ]
    for fold, (outer_train, outer_valid) in enumerate(folds):
        fit_indices = np.concatenate([outer_train, pool_indices])
        train_features, all_features, transform = fit_transform_model(
            embedding[fit_indices],
            embedding,
            extra[fit_indices],
            extra,
            args.pca_components,
            args.seed,
        )
        selected_c, inner_score = select_c_external(
            all_features,
            labels,
            weights,
            groups,
            outer_train,
            pool_indices,
            c_values,
            args.seed,
        )
        model = LogisticRegression(
            C=selected_c,
            max_iter=5000,
            random_state=args.seed,
        )
        model.fit(
            train_features,
            labels[fit_indices],
            sample_weight=weights[fit_indices],
        )
        probabilities[outer_valid] = model.predict_proba(
            all_features[outer_valid]
        )
        valid_score = balanced_accuracy_score(
            train_labels[outer_valid],
            probabilities[outer_valid].argmax(axis=1),
        )
        fold_metrics.append(
            {
                "fold": fold,
                "selected_c": selected_c,
                "inner_score": inner_score,
                "valid_score": float(valid_score),
            }
        )
        print(
            f"fold={fold} c={selected_c:g} inner={inner_score:.4f} "
            f"valid={valid_score:.4f}",
            flush=True,
        )

    oof_score = balanced_accuracy_score(
        train_labels,
        probabilities.argmax(axis=1),
    )
    print(f"style_external OOF={oof_score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else output.with_suffix(output.suffix + ".json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "train_features": args.train_features,
                "pool_features": args.pool_features,
                "pool": args.pool,
                "pool_rows": len(pool),
                "pool_weight": args.pool_weight,
                "use_style": args.use_style,
                "pca_components": args.pca_components,
                "folds": fold_metrics,
                "oof_balanced_accuracy": float(oof_score),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {metrics_path}", flush=True)

    if args.artifact:
        all_indices = np.arange(len(labels), dtype=np.int64)
        fit_features, _, transform = fit_transform_model(
            embedding,
            embedding,
            extra,
            extra,
            args.pca_components,
            args.seed,
        )
        final_c = float(
            np.median([row["selected_c"] for row in fold_metrics])
        )
        final_model = LogisticRegression(
            C=final_c,
            max_iter=5000,
            random_state=args.seed,
        ).fit(
            fit_features,
            labels,
            sample_weight=weights,
        )
        artifact_path = Path(args.artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transform": transform,
                "model": final_model,
                "labels": LABELS,
                "use_style": args.use_style,
                "pool_rows": len(pool),
                "pool_weight": args.pool_weight,
            },
            artifact_path,
            compress=3,
        )
        print(f"saved {artifact_path}", flush=True)


if __name__ == "__main__":
    main()
