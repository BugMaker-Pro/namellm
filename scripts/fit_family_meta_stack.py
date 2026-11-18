from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument(
        "--base-oof",
        default=str(
            ROOT / "features" / "qwen3_embedding_stack_blend_oof.npy"
        ),
    )
    parser.add_argument(
        "--specialist",
        default=str(
            ROOT / "validation" / "official_family_specialists.npz"
        ),
    )
    parser.add_argument("--c", type=float, default=0.3)
    parser.add_argument(
        "--output",
        default=str(ROOT / "configs" / "family_meta_stack_c03.json"),
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    targets = train["model"].map(
        {label: index for index, label in enumerate(LABELS)}
    ).to_numpy(dtype=np.int64)
    base = np.load(args.base_oof).astype(np.float64)
    base /= base.sum(axis=1, keepdims=True)
    with np.load(args.specialist, allow_pickle=False) as payload:
        specialist = payload["log_probabilities"].astype(np.float64)
    families = {}
    for family in ("claude", "gpt", "gemini"):
        labels = [
            label
            for label in LABELS
            if label.startswith(family + "-")
        ]
        columns = [LABELS.index(label) for label in labels]
        row_mask = train["model"].isin(labels).to_numpy()
        features = np.hstack(
            [
                np.log(np.clip(base[:, columns], 1e-12, None)),
                specialist[:, columns],
            ]
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.c,
                max_iter=2000,
                class_weight="balanced",
                random_state=20261118,
            ),
        )
        model.fit(features[row_mask], targets[row_mask])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        families[family] = {
            "labels": labels,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist(),
            "train_rows": int(row_mask.sum()),
        }
        print(
            f"{family} rows={row_mask.sum()} "
            f"features={features.shape[1]} classes={len(labels)}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format": "namellm-family-meta-stack-v1",
                "C": args.c,
                "families": families,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
