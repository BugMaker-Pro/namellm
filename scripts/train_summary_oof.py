from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features/token_folds")
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument("--funnel", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
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
    )

    probabilities = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    covered = np.zeros(len(frame), dtype=bool)
    for fold in range(5):
        path = Path(args.features_dir) / f"qwen3_token_features_fold{fold}.npz"
        data = np.load(path, allow_pickle=True)
        summary = data["X"]
        rows = data["row_indices"]
        train_mask = data["train_mask"]
        valid_mask = data["valid_mask"]
        fold_labels = data["y"]

        summary_scaler = StandardScaler().fit(summary[train_mask])
        train_summary = summary_scaler.transform(summary[train_mask])
        valid_summary = summary_scaler.transform(summary[valid_mask])
        pca = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=42,
        ).fit(train_summary)
        train_features = np.hstack(
            [pca.transform(train_summary), extra[rows[train_mask]]]
        )
        valid_features = np.hstack(
            [pca.transform(valid_summary), extra[rows[valid_mask]]]
        )
        feature_scaler = StandardScaler().fit(train_features)
        train_features = feature_scaler.transform(train_features)
        valid_features = feature_scaler.transform(valid_features)
        model = LogisticRegression(C=args.c, max_iter=5000)
        model.fit(train_features, fold_labels[train_mask])
        fold_probabilities = model.predict_proba(valid_features)
        probabilities[rows[valid_mask]] = fold_probabilities
        covered[rows[valid_mask]] = True
        score = balanced_accuracy_score(
            fold_labels[valid_mask],
            fold_probabilities.argmax(axis=1),
        )
        print(f"fold={fold} balanced_accuracy={score:.4f}", flush=True)

    if not covered.all():
        raise ValueError("Summary OOF does not cover every training row")
    oof_score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))
    print(f"summary OOF balanced_accuracy={oof_score:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    if args.funnel:
        funnel = np.load(args.funnel)
        for weight in [0.2, 0.3, 0.4, 0.5]:
            blended = weight * probabilities + (1 - weight) * funnel
            score = balanced_accuracy_score(
                labels,
                blended.argmax(axis=1),
            )
            print(
                f"blend summary_weight={weight:.1f} "
                f"balanced_accuracy={score:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
