from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ridgeft import RidgeFTModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402
from namellm.style import style_frame  # noqa: E402
from train_style_branches import fit_transform_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--input-csv", default=str(TRAIN_CSV))
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--total-dim", type=int, default=2048)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--use-style", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    labels = frame["model"].map(
        {label: index for index, label in enumerate(LABELS)}
    ).to_numpy(dtype=np.int64)
    groups = frame["prompt_id"].astype(str).to_numpy()
    texts = frame["text"].fillna("").astype(str).tolist()
    data = np.load(args.features, allow_pickle=True)
    embedding = data["X"].astype(np.float32)
    if "row_indices" in data.files and not np.array_equal(
        data["row_indices"].astype(np.int64),
        np.arange(len(frame)),
    ):
        raise ValueError("Feature rows do not align with the input CSV")
    style = (
        style_frame(texts).to_numpy(dtype=np.float32)
        if args.use_style
        else np.empty((len(frame), 0), dtype=np.float32)
    )
    zeros = np.zeros(len(frame), dtype=np.int32)
    probabilities = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    folds = list(GroupKFold(5).split(frame, labels, groups))
    fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(folds):
        train_features, all_features, _ = fit_transform_model(
            embedding[train_index],
            embedding,
            style[train_index],
            style,
            args.pca_components,
            args.seed,
        )
        model = RidgeFTModel.fit_base(
            train_features,
            labels[train_index],
            delta=args.delta,
            shrinkage=args.shrinkage,
            total_dim=args.total_dim,
            beta=args.beta,
            ridge_lam=args.ridge_lambda,
            seed=args.seed,
        )
        scores = model.scores(all_features[valid_index])
        scores -= scores.max(axis=1, keepdims=True)
        probabilities[valid_index] = np.exp(scores)
        probabilities[valid_index] /= probabilities[valid_index].sum(
            axis=1,
            keepdims=True,
        )
        score = balanced_accuracy_score(
            labels[valid_index],
            probabilities[valid_index].argmax(axis=1),
        )
        fold_metrics.append({"fold": fold, "valid_score": float(score)})
        print(f"fold={fold} valid={score:.4f}", flush=True)
    oof = balanced_accuracy_score(
        labels,
        probabilities.argmax(axis=1),
    )
    print(f"ridgeft_style OOF={oof:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    metrics = {
        "features": args.features,
        "pca_components": args.pca_components,
        "total_dim": args.total_dim,
        "delta": args.delta,
        "shrinkage": args.shrinkage,
        "beta": args.beta,
        "ridge_lambda": args.ridge_lambda,
        "use_style": args.use_style,
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
