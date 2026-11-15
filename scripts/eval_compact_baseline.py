from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apply_cluster_balance import balance_probabilities  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    probabilities = np.zeros((len(frame), len(LABELS)))
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))

    for fold, (train_index, valid_index) in enumerate(folds):
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            sublinear_tf=True,
            min_df=2,
            lowercase=False,
            dtype=np.float32,
        ).fit(frame["text"].iloc[train_index])
        word = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        ).fit(frame["text"].iloc[train_index])
        train_features = hstack(
            [
                char.transform(frame["text"].iloc[train_index]),
                word.transform(frame["text"].iloc[train_index]),
            ]
        ).tocsr()
        valid_features = hstack(
            [
                char.transform(frame["text"].iloc[valid_index]),
                word.transform(frame["text"].iloc[valid_index]),
            ]
        ).tocsr()
        model = LogisticRegression(C=3.0, max_iter=3000)
        model.fit(train_features, labels[train_index])
        probabilities[valid_index] = model.predict_proba(valid_features)
        score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        print(f"fold={fold} raw={score:.4f}", flush=True)

    raw_score = balanced_accuracy_score(
        labels,
        probabilities.argmax(axis=1),
    )
    balanced = np.zeros_like(probabilities)
    for _, valid_index in folds:
        balanced[valid_index] = balance_probabilities(
            probabilities[valid_index],
            frame["text"].iloc[valid_index].tolist(),
            center_alpha=0.5,
            strength=0.625,
            iterations=5,
            min_cluster_size=8,
        )
    balanced_score = balanced_accuracy_score(
        labels,
        balanced.argmax(axis=1),
    )
    print(f"compact raw OOF={raw_score:.4f}")
    print(f"compact balanced OOF={balanced_score:.4f}")
    if args.output:
        np.save(args.output, probabilities)


if __name__ == "__main__":
    main()
