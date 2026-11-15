from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402


def build_design(
    embedding: np.ndarray,
    style: np.ndarray,
    clusters: np.ndarray,
    use_style: bool,
    use_group_relative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(embedding, dtype=np.float32)
    extra_parts = []
    if use_style:
        extra_parts.append(np.asarray(style, dtype=np.float32))
    if use_group_relative:
        extra_parts.append(group_relative(matrix, clusters))
        if use_style:
            extra_parts.append(group_relative(style, clusters))
    if not extra_parts:
        return matrix, np.empty((len(matrix), 0), dtype=np.float32)
    return matrix, np.hstack(extra_parts).astype(np.float32)


def fit_transform_model(
    embedding_train: np.ndarray,
    embedding_all: np.ndarray,
    extra_train: np.ndarray,
    extra_all: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = StandardScaler().fit(embedding_train)
    train_scaled = scaler.transform(embedding_train)
    all_scaled = scaler.transform(embedding_all)
    max_components = min(train_scaled.shape)
    components = (
        max_components
        if pca_components <= 0
        else min(pca_components, max_components)
    )
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=seed,
    ).fit(train_scaled)
    train_dense = pca.transform(train_scaled)
    all_dense = pca.transform(all_scaled)
    if extra_train.shape[1]:
        extra_scaler = StandardScaler().fit(extra_train)
        train_dense = np.hstack(
            [train_dense, extra_scaler.transform(extra_train)]
        )
        all_dense = np.hstack(
            [all_dense, extra_scaler.transform(extra_all)]
        )
    else:
        extra_scaler = None
    combined_scaler = StandardScaler().fit(train_dense)
    return (
        combined_scaler.transform(train_dense).astype(np.float32),
        combined_scaler.transform(all_dense).astype(np.float32),
        {
            "embedding_scaler": scaler,
            "embedding_pca": pca,
            "extra_scaler": extra_scaler,
            "combined_scaler": combined_scaler,
            "pca_components": components,
        },
    )


def select_c(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[float, float]:
    inner_folds = list(
        GroupKFold(4).split(
            features[train_index],
            labels[train_index],
            groups[train_index],
        )
    )
    best: tuple[float, float] | None = None
    global_train_index = train_index
    for candidate in c_values:
        inner_scores = []
        for inner_train, inner_valid in inner_folds:
            model = LogisticRegression(
                C=candidate,
                max_iter=5000,
                random_state=seed,
            )
            model.fit(
                features[global_train_index][inner_train],
                labels[global_train_index][inner_train],
            )
            predictions = model.predict(
                features[global_train_index][inner_valid]
            )
            inner_scores.append(
                balanced_accuracy_score(
                    labels[global_train_index][inner_valid],
                    predictions,
                )
            )
        score = float(np.mean(inner_scores))
        if best is None or score > best[1]:
            best = (candidate, score)
    if best is None:
        raise RuntimeError("C selection produced no candidate")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--c-values", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--use-style", action="store_true")
    parser.add_argument("--group-relative", action="store_true")
    parser.add_argument("--no-nested-c", action="store_true")
    parser.add_argument("--fixed-c", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["model"].map(label_to_index).to_numpy(dtype=np.int64)
    if np.isnan(labels).any():
        raise ValueError("Input CSV contains unknown model labels")
    groups = frame["prompt_id"].astype(str).to_numpy()
    texts = frame["text"].fillna("").astype(str).tolist()
    data = np.load(args.features, allow_pickle=True)
    embedding = data["X"].astype(np.float32)
    row_indices = data["row_indices"].astype(np.int64)
    if not np.array_equal(row_indices, np.arange(len(frame))):
        raise ValueError("Embedding rows do not align with the input CSV")
    if "y" in data.files:
        saved_labels = data["y"].astype(np.int64)
        if not np.array_equal(saved_labels, labels):
            raise ValueError("Embedding labels do not match the input CSV")
    style = style_frame(texts).to_numpy(dtype=np.float32)
    clusters = blind_prompt_clusters(texts, 0.90)
    embedding_design, extra_design = build_design(
        embedding,
        style,
        clusters,
        args.use_style,
        args.group_relative,
    )
    print(
        f"embedding={embedding_design.shape} extra={extra_design.shape} "
        f"clusters={len(np.unique(clusters))}",
        flush=True,
    )

    c_values = [
        float(value) for value in args.c_values.split(",") if value
    ]
    folds = list(GroupKFold(5).split(frame, labels, groups))
    probabilities = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(folds):
        fold_features, all_features, transform = fit_transform_model(
            embedding_design[train_index],
            embedding_design,
            extra_design[train_index],
            extra_design,
            args.pca_components,
            args.seed,
        )
        if args.no_nested_c:
            selected_c = args.fixed_c
            inner_score = float("nan")
        else:
            selected_c, inner_score = select_c(
                all_features,
                labels,
                groups,
                train_index,
                c_values,
                args.seed,
            )
        model = LogisticRegression(
            C=selected_c,
            max_iter=5000,
            random_state=args.seed,
        )
        model.fit(fold_features, labels[train_index])
        probabilities[valid_index] = model.predict_proba(
            all_features[valid_index]
        )
        valid_score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        fold_metrics.append(
            {
                "fold": fold,
                "selected_c": selected_c,
                "inner_score": inner_score,
                "valid_score": float(valid_score),
                "pca_components": transform["pca_components"],
            }
        )
        print(
            f"fold={fold} c={selected_c:g} inner={inner_score:.4f} "
            f"valid={valid_score:.4f}",
            flush=True,
        )

    oof_score = balanced_accuracy_score(
        labels,
        probabilities.argmax(axis=1),
    )
    print(f"style_branch OOF={oof_score:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    print(f"saved {output}", flush=True)

    metrics = {
        "features": str(Path(args.features)),
        "use_style": args.use_style,
        "group_relative": args.group_relative,
        "pca_components": args.pca_components,
        "c_values": c_values,
        "fixed_c": None if not args.no_nested_c else args.fixed_c,
        "folds": fold_metrics,
        "oof_balanced_accuracy": float(oof_score),
    }
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else output.with_suffix(output.suffix + ".json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"saved {metrics_path}", flush=True)

    if args.artifact:
        all_features, _, transform = fit_transform_model(
            embedding_design,
            embedding_design,
            extra_design,
            extra_design,
            args.pca_components,
            args.seed,
        )
        final_model = LogisticRegression(
            C=args.fixed_c if args.no_nested_c else float(
                np.median([row["selected_c"] for row in fold_metrics])
            ),
            max_iter=5000,
            random_state=args.seed,
        ).fit(all_features, labels)
        artifact_path = Path(args.artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transform": transform,
                "model": final_model,
                "labels": LABELS,
                "use_style": args.use_style,
                "group_relative": args.group_relative,
            },
            artifact_path,
            compress=3,
        )
        print(f"saved {artifact_path}", flush=True)


if __name__ == "__main__":
    main()
