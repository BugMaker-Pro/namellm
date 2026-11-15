from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import LABELS, TRAIN_CSV, family, group  # noqa: E402
from namellm.style import style_frame  # noqa: E402


GROUPS = ["ClaudeGLM", "GPTKimiQwen", "Gemini"]
COMPONENTS = ["tfidf", "tfidf_rel", "dense", "hgb"]


def fit_predict(component: str, x_train, y_train, x_test) -> np.ndarray:
    if component == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=0,
        )
    else:
        c = {"tfidf": 10.0, "tfidf_rel": 3.0, "dense": 0.3}[component]
        if component == "dense":
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        model = LogisticRegression(C=c, max_iter=3000)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)


def grouped_probabilities(probabilities: np.ndarray, members: dict[str, list[int]]) -> np.ndarray:
    return np.stack(
        [probabilities[:, members[name]].sum(axis=1) for name in GROUPS],
        axis=1,
    )


def meta_predict(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    class_count: int,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = np.zeros((len(labels), class_count), dtype=np.float64)
    for train_idx, valid_idx in folds:
        fit_idx = train_idx
        if train_mask is not None:
            fit_idx = train_idx[train_mask[train_idx]]
        scaler = StandardScaler().fit(features[fit_idx])
        model = LogisticRegression(C=0.3, max_iter=3000)
        model.fit(scaler.transform(features[fit_idx]), labels[fit_idx])
        output[valid_idx] = model.predict_proba(scaler.transform(features[valid_idx]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument("--output-dir", default="D:/tmp/namellm_funnel")
    parser.add_argument("--max-folds", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    frame = pd.read_csv(TRAIN_CSV)
    n_rows = len(frame)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    y18 = np.array([label_to_index[value] for value in frame["model"]])
    y_group = np.array([group(value) for value in frame["model"]])
    folds = list(GroupKFold(5).split(frame, y_group, frame["prompt_id"]))[
        : args.max_folds
    ]

    clusters = blind_prompt_clusters(frame["text"].tolist(), 0.90)
    print(f"blind clusters: {len(np.unique(clusters))}", flush=True)

    style = style_frame(frame["text"]).to_numpy(dtype=np.float32)
    lm = (
        frame[["id"]]
        .merge(
            pd.read_parquet(args.lm_features)
            .query("split == 'train'")
            .drop(columns="split"),
            on="id",
            how="left",
        )
        .drop(columns="id")
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(lm).all():
        raise ValueError("LM features contain missing or non-finite values")
    dense = np.hstack(
        [
            style,
            group_relative(style, clusters),
            lm,
            group_relative(lm, clusters),
        ]
    )
    print(f"dense features: {dense.shape}", flush=True)

    members = {
        name: [LABELS.index(label) for label in LABELS if group(label) == name]
        for name in GROUPS
    }
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    flat_oof = {name: np.zeros((n_rows, len(LABELS))) for name in COMPONENTS}

    for fold_index, (train_idx, valid_idx) in enumerate(folds):
        cache_path = cache_dir / f"fold{fold_index}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            for name in COMPONENTS:
                flat_oof[name][valid_idx] = cached["flat"][name]
            print(f"fold {fold_index}: loaded cache", flush=True)
            continue

        fold_started = time.time()
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            sublinear_tf=True,
            min_df=2,
            lowercase=False,
            dtype=np.float32,
        ).fit(frame["text"].iloc[train_idx])
        word = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        ).fit(frame["text"].iloc[train_idx])
        tfidf = hstack(
            [char.transform(frame["text"]), word.transform(frame["text"])]
        ).tocsr()

        char_relative = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            sublinear_tf=True,
            min_df=3,
            lowercase=False,
            max_features=6000,
            dtype=np.float32,
        ).fit(frame["text"].iloc[train_idx])
        word_relative = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_features=2500,
        ).fit(frame["text"].iloc[train_idx])
        tfidf_relative = group_relative(
            np.hstack(
                [
                    char_relative.transform(frame["text"]).toarray(),
                    word_relative.transform(frame["text"]).toarray(),
                ]
            ),
            clusters,
        ).astype(np.float32)

        matrices = {
            "tfidf": tfidf,
            "tfidf_rel": tfidf_relative,
            "dense": dense,
            "hgb": dense,
        }
        flat_fold = {}
        group_fold: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in GROUPS
        }
        for component, matrix in matrices.items():
            flat_fold[component] = fit_predict(
                component,
                matrix[train_idx],
                y18[train_idx],
                matrix[valid_idx],
            )
            for group_name in GROUPS:
                selected = train_idx[y_group[train_idx] == group_name]
                group_fold[group_name][component] = fit_predict(
                    component,
                    matrix[selected],
                    y18[selected],
                    matrix[valid_idx],
                )

        with cache_path.open("wb") as handle:
            pickle.dump({"flat": flat_fold, "group": group_fold}, handle)
        for component in COMPONENTS:
            flat_oof[component][valid_idx] = flat_fold[component]
        print(
            f"fold {fold_index}: {time.time() - fold_started:.1f}s",
            flush=True,
        )

    flat_features = np.hstack(
        [np.log(flat_oof[name] + 1e-9) for name in COMPONENTS]
    )
    flat_control = meta_predict(flat_features, y18, folds, len(LABELS))

    group_components = {
        name: grouped_probabilities(flat_oof[name], members) for name in COMPONENTS
    }
    stage1_features = np.hstack(
        [np.log(group_components[name] + 1e-9) for name in COMPONENTS]
    )
    group_to_index = {name: index for index, name in enumerate(GROUPS)}
    y_group_index = np.array([group_to_index[name] for name in y_group])
    stage1 = meta_predict(stage1_features, y_group_index, folds, len(GROUPS))

    stage2: dict[str, np.ndarray] = {}
    for group_name in GROUPS:
        component_values = {
            name: np.zeros((n_rows, len(members[group_name])))
            for name in COMPONENTS
        }
        for fold_index, (train_idx, valid_idx) in enumerate(folds):
            with (cache_dir / f"fold{fold_index}.pkl").open("rb") as handle:
                cached = pickle.load(handle)["group"][group_name]
            for name in COMPONENTS:
                component_values[name][valid_idx] = cached[name]
        stage2_features = np.hstack(
            [np.log(component_values[name] + 1e-9) for name in COMPONENTS]
        )
        stage2[group_name] = meta_predict(
            stage2_features,
            y18,
            folds,
            len(members[group_name]),
            train_mask=y_group == group_name,
        )

    funnel = np.zeros((n_rows, len(LABELS)))
    for group_index, group_name in enumerate(GROUPS):
        class_indices = members[group_name]
        funnel[:, class_indices] = (
            stage1[:, [group_index]] * stage2[group_name]
        )

    def report(name: str, probabilities: np.ndarray) -> None:
        predictions = probabilities.argmax(axis=1)
        score = balanced_accuracy_score(y18, predictions)
        family_score = np.mean(
            [family(LABELS[pred]) == family(actual) for pred, actual in zip(predictions, frame["model"])]
        )
        print(f"{name:<18} balanced={score:.4f} family={family_score:.4f}", flush=True)
        np.save(output_dir / f"{name}.npy", probabilities)

    print(f"\ntotal feature build: {time.time() - started:.1f}s", flush=True)
    stage1_predictions = np.array(GROUPS)[stage1.argmax(axis=1)]
    print(
        f"stage1 accuracy={np.mean(stage1_predictions == y_group):.4f}",
        flush=True,
    )
    report("flat_control", flat_control)
    report("funnel", funnel)


if __name__ == "__main__":
    main()
