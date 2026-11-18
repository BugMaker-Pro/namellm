from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pool_sibling_specialists import (  # noqa: E402
    load_pool,
    normalize_text,
    text_key,
)


FAMILIES = ("claude", "gpt", "gemini", "glm", "qwen", "kimi")


def family_name(label: str) -> str:
    value = label.lower()
    if value.startswith("qwen3.8"):
        return "qwen"
    return value.split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(ROOT / "data/raw/train.csv"))
    parser.add_argument(
        "--pool",
        default=str(
            ROOT
            / "artifacts"
            / "open_pool_filtered"
            / "ntl_open_pool_filtered.parquet"
        ),
    )
    parser.add_argument("--max-per-model", type=int, default=12000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument(
        "--output",
        default=str(ROOT / "validation/family_router_oof.npy"),
    )
    parser.add_argument(
        "--metrics",
        default=str(ROOT / "validation/family_router_metrics.json"),
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    train["text"] = train["text"].fillna("").astype(str).map(normalize_text)
    labels = np.asarray([family_name(value) for value in train["model"]])
    family_to_index = {
        family: index for index, family in enumerate(FAMILIES)
    }
    targets = np.asarray([family_to_index[value] for value in labels])
    pool = load_pool(Path(args.pool), args.seed, args.max_per_model)
    official_keys = set(train["text"].map(text_key))
    pool = pool[~pool["_text_key"].isin(official_keys)].reset_index(drop=True)
    pool["text"] = pool["text"].map(normalize_text)
    pool_targets = np.asarray(
        [family_to_index[family_name(value)] for value in pool["model"]]
    )
    folds = list(
        GroupKFold(5).split(train, targets, train["prompt_id"])
    )
    probabilities = np.zeros((len(train), len(FAMILIES)), dtype=np.float32)
    fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(folds):
        fit_texts = pool["text"].tolist() + train.iloc[train_index]["text"].tolist()
        fit_targets = np.concatenate(
            [pool_targets, targets[train_index]]
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            sublinear_tf=True,
            min_df=3,
            max_features=600000,
            lowercase=False,
            dtype=np.float32,
        )
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_features=300000,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            lowercase=False,
            dtype=np.float32,
        )
        matrix = sparse.hstack(
            [
                char.fit_transform(fit_texts),
                word.fit_transform(fit_texts),
            ],
            format="csr",
        )
        query = sparse.hstack(
            [
                char.transform(train.iloc[valid_index]["text"]),
                word.transform(train.iloc[valid_index]["text"]),
            ],
            format="csr",
        )
        model = SGDClassifier(
            loss="log_loss",
            alpha=args.alpha,
            max_iter=20,
            tol=None,
            class_weight="balanced",
            random_state=args.seed + fold,
        ).fit(matrix, fit_targets)
        classes = np.asarray(model.classes_, dtype=int)
        local = model.predict_proba(query)
        probabilities[np.ix_(valid_index, classes)] = local
        valid_score = balanced_accuracy_score(
            targets[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": int(len(fit_texts)),
                "valid_rows": int(len(valid_index)),
                "feature_shape": list(matrix.shape),
                "balanced_accuracy": float(valid_score),
            }
        )
        print(json.dumps(fold_metrics[-1]), flush=True)

    normalised = probabilities / probabilities.sum(axis=1, keepdims=True)
    overall = balanced_accuracy_score(
        targets,
        normalised.argmax(axis=1),
    )
    result = {
        "families": list(FAMILIES),
        "rows": int(len(train)),
        "overall_balanced_accuracy": float(overall),
        "folds": fold_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, normalised)
    metrics = Path(args.metrics)
    metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
