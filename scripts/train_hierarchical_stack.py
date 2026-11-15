from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV, group  # noqa: E402


GROUPS = ["ClaudeGLM", "GPTKimiQwen", "Gemini"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-c", type=float, default=0.1)
    parser.add_argument("--stage2-c", type=float, default=0.03)
    parser.add_argument(
        "--output",
        default="D:/tmp/namellm_hierarchical_stack_oof.npy",
    )
    args = parser.parse_args()

    paths = [
        ROOT / "features" / "summary_funnel_power_blend.npy",
        ROOT / "features" / "qwen3_embedding_4b_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_pca256_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layers4_oof.npy",
        ROOT / "features" / "qwen3_embedding_4b_layer_sequence7_oof.npy",
    ]
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    y_group = np.array([group(value) for value in frame["model"]])
    group_to_index = {name: index for index, name in enumerate(GROUPS)}
    y_group_index = np.array([group_to_index[value] for value in y_group])
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
    components = [np.load(path) for path in paths]

    stage1_features = np.hstack(
        [
            np.log(
                np.stack(
                    [
                        values[:, members[name]].sum(axis=1)
                        for name in GROUPS
                    ],
                    axis=1,
                )
                + 1e-9
            )
            for values in components
        ]
    )
    stage2_features = {}
    for name in GROUPS:
        class_indices = members[name]
        blocks = []
        for values in components:
            group_probabilities = values[:, class_indices]
            group_probabilities /= group_probabilities.sum(
                axis=1,
                keepdims=True,
            )
            blocks.append(np.log(group_probabilities + 1e-9))
        stage2_features[name] = np.hstack(blocks)

    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    stage1_probabilities = np.zeros((len(frame), len(GROUPS)))
    stage2_probabilities = {
        name: np.zeros((len(frame), len(members[name])))
        for name in GROUPS
    }
    for fold, (train_index, valid_index) in enumerate(folds):
        scaler = StandardScaler().fit(stage1_features[train_index])
        model = LogisticRegression(C=args.stage1_c, max_iter=3000)
        model.fit(
            scaler.transform(stage1_features[train_index]),
            y_group_index[train_index],
        )
        stage1_probabilities[valid_index] = model.predict_proba(
            scaler.transform(stage1_features[valid_index])
        )

        for name in GROUPS:
            group_train = train_index[y_group[train_index] == name]
            values = stage2_features[name]
            scaler = StandardScaler().fit(values[group_train])
            model = LogisticRegression(C=args.stage2_c, max_iter=3000)
            model.fit(
                scaler.transform(values[group_train]),
                labels[group_train],
            )
            stage2_probabilities[name][valid_index] = model.predict_proba(
                scaler.transform(values[valid_index])
            )
        print(f"fold={fold}", flush=True)

    stacked = np.zeros((len(frame), len(LABELS)))
    for group_index, name in enumerate(GROUPS):
        stacked[:, members[name]] = (
            stage1_probabilities[:, [group_index]]
            * stage2_probabilities[name]
        )

    score = balanced_accuracy_score(labels, stacked.argmax(axis=1))
    group_score = np.mean(
        np.array(GROUPS)[stage1_probabilities.argmax(axis=1)] == y_group
    )
    print(f"hierarchical_stack OOF={score:.4f} group={group_score:.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, stacked)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
