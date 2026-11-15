from __future__ import annotations

import argparse
import json
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


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--pca-components", default="128,256,512,768")
    parser.add_argument("--c-values", default="0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    pca_values = parse_ints(args.pca_components)
    c_values = parse_floats(args.c_values)
    max_components = max(pca_values)

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    features = np.load(args.features, allow_pickle=True)["X"].astype(np.float32)

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

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    candidates = [
        (components, c)
        for components in pca_values
        for c in c_values
    ]
    probabilities = {
        candidate: np.zeros((len(frame), len(LABELS)))
        for candidate in candidates
    }

    for fold, (train_index, valid_index) in enumerate(folds):
        scaler = StandardScaler().fit(features[train_index])
        pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            random_state=42,
        ).fit(scaler.transform(features[train_index]))
        pca_all = pca.transform(scaler.transform(features))
        fold_scores: list[tuple[float, int, float]] = []

        for components in pca_values:
            selected = pca_all[:, :components]
            relative = group_relative(selected, clusters)
            combined = np.hstack([selected, relative, extra])
            combined_scaler = StandardScaler().fit(combined[train_index])
            train_x = combined_scaler.transform(combined[train_index])
            valid_x = combined_scaler.transform(combined[valid_index])
            for c in c_values:
                model = LogisticRegression(C=c, max_iter=5000)
                model.fit(train_x, labels[train_index])
                candidate_probabilities = model.predict_proba(valid_x)
                probabilities[(components, c)][valid_index] = (
                    candidate_probabilities
                )
                score = balanced_accuracy_score(
                    labels[valid_index],
                    candidate_probabilities.argmax(axis=1),
                )
                fold_scores.append((score, components, c))

        best_score, best_components, best_c = max(fold_scores)
        print(
            f"fold={fold} best={best_score:.4f} "
            f"pca={best_components} c={best_c}",
            flush=True,
        )

    results = []
    for (components, c), candidate_probabilities in probabilities.items():
        score = balanced_accuracy_score(
            labels,
            candidate_probabilities.argmax(axis=1),
        )
        results.append(
            {
                "pca_components": components,
                "c": c,
                "balanced_accuracy": float(score),
            }
        )
    results.sort(key=lambda item: item["balanced_accuracy"], reverse=True)

    for result in results[:10]:
        print(
            f"pca={result['pca_components']} c={result['c']} "
            f"OOF={result['balanced_accuracy']:.4f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = results[0]
    best_key = (best["pca_components"], best["c"])
    np.save(output_dir / "best_oof.npy", probabilities[best_key])
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
