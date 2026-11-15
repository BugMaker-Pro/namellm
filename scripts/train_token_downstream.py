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

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        default="features/token_folds",
        help="Directory with qwen3_token_features_fold{0..4}.npz",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--funnel", default="")
    parser.add_argument("--token-weight", type=float, default=0.2)
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    y_all = np.array([label_to_index[value] for value in frame["model"]])
    probabilities = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    covered = np.zeros(len(frame), dtype=bool)

    for fold in range(5):
        path = Path(args.features_dir) / f"qwen3_token_features_fold{fold}.npz"
        data = np.load(path, allow_pickle=True)
        features = data["X"]
        train_mask = data["train_mask"]
        valid_mask = data["valid_mask"]
        row_indices = data["row_indices"]
        labels = data["y"]

        scaler = StandardScaler().fit(features[train_mask])
        train_scaled = scaler.transform(features[train_mask])
        valid_scaled = scaler.transform(features[valid_mask])
        pca = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=42,
        ).fit(train_scaled)
        train_features = pca.transform(train_scaled)
        valid_features = pca.transform(valid_scaled)
        model = LogisticRegression(C=args.c, max_iter=4000)
        model.fit(train_features, labels[train_mask])
        fold_probabilities = model.predict_proba(valid_features)
        probabilities[row_indices[valid_mask]] = fold_probabilities
        covered[row_indices[valid_mask]] = True
        score = balanced_accuracy_score(
            labels[valid_mask],
            fold_probabilities.argmax(axis=1),
        )
        print(f"fold={fold} balanced_accuracy={score:.4f}", flush=True)

    if not covered.all():
        raise ValueError("Token downstream OOF does not cover every training row")
    score = balanced_accuracy_score(y_all, probabilities.argmax(axis=1))
    print(f"token OOF balanced_accuracy={score:.4f}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, probabilities)

    if args.funnel:
        funnel = np.load(args.funnel)
        weight = args.token_weight
        blended = weight * probabilities + (1 - weight) * funnel
        blended_score = balanced_accuracy_score(y_all, blended.argmax(axis=1))
        print(
            f"blend token_weight={weight:.2f} "
            f"balanced_accuracy={blended_score:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
