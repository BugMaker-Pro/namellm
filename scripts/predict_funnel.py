from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_funnel import COMPONENTS, GROUPS, fit_predict  # noqa: E402
from namellm.clustering import blind_prompt_clusters, group_relative  # noqa: E402
from namellm.constants import (  # noqa: E402
    LABELS,
    TEST_CSV,
    TRAIN_CSV,
    group,
)
from namellm.style import style_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--test-csv", default=str(TEST_CSV))
    parser.add_argument(
        "--lm-features",
        default="features/e002_qwen3-4b-base-4bit.parquet",
    )
    parser.add_argument(
        "--cache-dir",
        default="D:/tmp/namellm_funnel/cache",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    y_train = np.array([label_to_index[value] for value in train["model"]])
    y_group = np.array([group(value) for value in train["model"]])

    lm_frame = pd.read_parquet(args.lm_features)
    train_lm = (
        lm_frame.query("split == 'train'")
        .set_index("id")
        .loc[train["id"]]
        .drop(columns="split")
        .to_numpy(dtype=np.float32)
    )
    test_lm = (
        lm_frame.query("split == 'test'")
        .set_index("id")
        .loc[test["id"]]
        .drop(columns="split")
        .to_numpy(dtype=np.float32)
    )

    train_clusters = blind_prompt_clusters(train["text"].tolist(), 0.90)
    test_clusters = blind_prompt_clusters(test["text"].tolist(), 0.90)
    train_style = style_frame(train["text"]).to_numpy(dtype=np.float32)
    test_style = style_frame(test["text"]).to_numpy(dtype=np.float32)
    train_dense = np.hstack(
        [
            train_style,
            group_relative(train_style, train_clusters),
            train_lm,
            group_relative(train_lm, train_clusters),
        ]
    )
    test_dense = np.hstack(
        [
            test_style,
            group_relative(test_style, test_clusters),
            test_lm,
            group_relative(test_lm, test_clusters),
        ]
    )

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
        dtype=np.float32,
    ).fit(train["text"])
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    ).fit(train["text"])
    train_tfidf = hstack(
        [char.transform(train["text"]), word.transform(train["text"])]
    ).tocsr()
    test_tfidf = hstack(
        [char.transform(test["text"]), word.transform(test["text"])]
    ).tocsr()

    char_relative = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        sublinear_tf=True,
        min_df=3,
        lowercase=False,
        max_features=6000,
        dtype=np.float32,
    ).fit(train["text"])
    word_relative = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_features=2500,
    ).fit(train["text"])
    train_tfidf_relative = group_relative(
        np.hstack(
            [
                char_relative.transform(train["text"]).toarray(),
                word_relative.transform(train["text"]).toarray(),
            ]
        ),
        train_clusters,
    ).astype(np.float32)
    test_tfidf_relative = group_relative(
        np.hstack(
            [
                char_relative.transform(test["text"]).toarray(),
                word_relative.transform(test["text"]).toarray(),
            ]
        ),
        test_clusters,
    ).astype(np.float32)

    train_matrices = {
        "tfidf": train_tfidf,
        "tfidf_rel": train_tfidf_relative,
        "dense": train_dense,
        "hgb": train_dense,
    }
    test_matrices = {
        "tfidf": test_tfidf,
        "tfidf_rel": test_tfidf_relative,
        "dense": test_dense,
        "hgb": test_dense,
    }
    flat_test = {}
    group_test: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in GROUPS
    }
    for component in COMPONENTS:
        train_matrix = train_matrices[component]
        test_matrix = test_matrices[component]
        flat_test[component] = fit_predict(
            component,
            train_matrix,
            y_train,
            test_matrix,
        )
        for group_name in GROUPS:
            selected = y_group == group_name
            group_test[group_name][component] = fit_predict(
                component,
                train_matrix[selected],
                y_train[selected],
                test_matrix,
            )

    cache_dir = Path(args.cache_dir)
    members = {
        name: np.array(
            [
                index
                for index, label in enumerate(LABELS)
                if group(label) == name
            ]
        )
        for name in GROUPS
    }
    flat_oof = {
        component: np.zeros((len(train), len(LABELS)))
        for component in COMPONENTS
    }
    group_oof: dict[str, dict[str, np.ndarray]] = {
        group_name: {
            component: np.zeros(
                (len(train), len(members[group_name]))
            )
            for component in COMPONENTS
        }
        for group_name in GROUPS
    }
    folds = []
    from sklearn.model_selection import GroupKFold

    folds = list(
        GroupKFold(5).split(train, y_group, train["prompt_id"])
    )
    for fold, (_, valid_index) in enumerate(folds):
        with (cache_dir / f"fold{fold}.pkl").open("rb") as handle:
            cached = pickle.load(handle)
        for component in COMPONENTS:
            flat_oof[component][valid_index] = cached["flat"][component]
            for group_name in GROUPS:
                group_oof[group_name][component][valid_index] = (
                    cached["group"][group_name][component]
                )

    def grouped(probabilities: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                probabilities[:, members[name]].sum(axis=1)
                for name in GROUPS
            ],
            axis=1,
        )

    stage1_train = np.hstack(
        [
            np.log(grouped(flat_oof[component]) + 1e-9)
            for component in COMPONENTS
        ]
    )
    stage1_test = np.hstack(
        [
            np.log(grouped(flat_test[component]) + 1e-9)
            for component in COMPONENTS
        ]
    )
    group_to_index = {
        group_name: index for index, group_name in enumerate(GROUPS)
    }
    y_group_index = np.array(
        [group_to_index[value] for value in y_group]
    )
    stage1_scaler = StandardScaler().fit(stage1_train)
    stage1_model = LogisticRegression(C=0.3, max_iter=3000)
    stage1_model.fit(
        stage1_scaler.transform(stage1_train),
        y_group_index,
    )
    stage1_test_probabilities = stage1_model.predict_proba(
        stage1_scaler.transform(stage1_test)
    )

    stage2_test = {}
    for group_name in GROUPS:
        class_indices = members[group_name]
        train_features = np.hstack(
            [
                np.log(group_oof[group_name][component] + 1e-9)
                for component in COMPONENTS
            ]
        )
        test_features = np.hstack(
            [
                np.log(group_test[group_name][component] + 1e-9)
                for component in COMPONENTS
            ]
        )
        train_mask = y_group == group_name
        local_labels = np.searchsorted(
            class_indices,
            y_train[train_mask],
        )
        scaler = StandardScaler().fit(train_features[train_mask])
        model = LogisticRegression(C=0.3, max_iter=3000)
        model.fit(
            scaler.transform(train_features[train_mask]),
            local_labels,
        )
        stage2_test[group_name] = model.predict_proba(
            scaler.transform(test_features)
        )

    probabilities = np.zeros((len(test), len(LABELS)))
    for group_index, group_name in enumerate(GROUPS):
        probabilities[:, members[group_name]] = (
            stage1_test_probabilities[:, [group_index]]
            * stage2_test[group_name]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probabilities)
    predictions = pd.DataFrame(
        {
            "id": test["id"],
            "model": np.array(LABELS)[probabilities.argmax(axis=1)],
            "confidence": probabilities.max(axis=1),
        }
    )
    predictions.to_csv(
        output.with_suffix(".csv"),
        index=False,
    )
    print(predictions.to_string(index=False))
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
