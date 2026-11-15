from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument(
        "--blend-baseline",
        default="features/summary_funnel_power_blend.npy",
    )
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument("--no-extra", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-embedding", default="")
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    data = np.load(args.features, allow_pickle=True)
    features = data["X"].astype(np.float32)
    row_indices = data["row_indices"]
    if not np.array_equal(row_indices, np.arange(len(frame))):
        raise ValueError("Embedding features must cover training rows in order")
    if not np.array_equal(data["y"], labels):
        raise ValueError("Embedding feature labels do not match train.csv")

    clusters = blind_prompt_clusters(frame["text"].tolist(), 0.90)
    if args.no_extra:
        extra = np.empty((len(frame), 0), dtype=np.float32)
    else:
        style = style_frame(frame["text"]).to_numpy(dtype=np.float32)
        lm = (
            pd.read_parquet(args.lm_features)
            .query("split == 'train'")
            .set_index("id")
            .loc[frame["id"]]
            .drop(columns="split")
            .to_numpy(dtype=np.float32)
        )
        extra = np.hstack(
            [
                style,
                group_relative(style, clusters),
                lm,
                group_relative(lm, clusters),
            ]
        )
    print(f"embedding_features={features.shape} extra_features={extra.shape}")

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    embedding_probabilities = np.zeros((len(frame), len(LABELS)))
    combined_probabilities = np.zeros_like(embedding_probabilities)
    for fold, (train_index, valid_index) in enumerate(folds):
        scaler = StandardScaler().fit(features[train_index])
        train_scaled = scaler.transform(features[train_index])
        pca = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=42,
        ).fit(train_scaled)
        pca_all = pca.transform(scaler.transform(features))
        relative_all = group_relative(pca_all, clusters)
        train_pca = np.hstack([pca_all[train_index], relative_all[train_index]])
        valid_pca = np.hstack([pca_all[valid_index], relative_all[valid_index]])

        embedding_model = LogisticRegression(C=args.c, max_iter=5000)
        embedding_model.fit(train_pca, labels[train_index])
        embedding_probabilities[valid_index] = embedding_model.predict_proba(
            valid_pca
        )
        embedding_score = balanced_accuracy_score(
            labels[valid_index],
            embedding_probabilities[valid_index].argmax(axis=1),
        )

        combined_train = np.hstack([train_pca, extra[train_index]])
        combined_valid = np.hstack([valid_pca, extra[valid_index]])
        combined_scaler = StandardScaler().fit(combined_train)
        combined_model = LogisticRegression(C=args.c, max_iter=5000)
        combined_model.fit(
            combined_scaler.transform(combined_train),
            labels[train_index],
        )
        combined_probabilities[valid_index] = combined_model.predict_proba(
            combined_scaler.transform(combined_valid)
        )
        combined_score = balanced_accuracy_score(
            labels[valid_index],
            combined_probabilities[valid_index].argmax(axis=1),
        )
        print(
            f"fold={fold} embedding={embedding_score:.4f} "
            f"combined={combined_score:.4f}",
            flush=True,
        )

    embedding_oof = balanced_accuracy_score(
        labels,
        embedding_probabilities.argmax(axis=1),
    )
    combined_oof = balanced_accuracy_score(
        labels,
        combined_probabilities.argmax(axis=1),
    )
    print(f"embedding OOF={embedding_oof:.4f}", flush=True)
    print(f"combined OOF={combined_oof:.4f}", flush=True)

    baseline_path = Path(args.blend_baseline)
    if baseline_path.exists():
        baseline = np.load(baseline_path)
        for weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
            blended = (
                weight * combined_probabilities
                + (1.0 - weight) * baseline
            )
            score = balanced_accuracy_score(labels, blended.argmax(axis=1))
            print(
                f"blend embedding_weight={weight:.1f} "
                f"balanced_accuracy={score:.4f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, combined_probabilities)
    print(f"saved {output}", flush=True)
    if args.output_embedding:
        embedding_output = Path(args.output_embedding)
        embedding_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_output, embedding_probabilities)
        print(f"saved {embedding_output}", flush=True)


if __name__ == "__main__":
    main()
