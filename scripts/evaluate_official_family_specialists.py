from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pool_sibling_specialists import (  # noqa: E402
    FAMILIES,
    LABELS,
    build_specialist,
    load_pool,
    normalize_text,
    text_key,
)


def score_subset(targets: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> float:
    local_y = targets[mask]
    predictions = probabilities[mask].argmax(axis=1)
    recalls = [
        float(np.mean(predictions[local_y == label] == label))
        for label in np.unique(local_y)
    ]
    return float(np.mean(recalls))


def apply_family(
    logits: np.ndarray,
    log_probabilities: np.ndarray,
    classes: np.ndarray,
    weight: float,
) -> np.ndarray:
    result = logits.copy()
    columns = [LABELS.index(label) for label in classes]
    mass = np.logaddexp.reduce(
        result[:, columns],
        axis=1,
        keepdims=True,
    )
    result[:, columns] = mass + log_softmax(
        result[:, columns] + weight * log_probabilities[:, columns],
        axis=1,
    )
    return result


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
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--alpha", type=float, default=2e-6)
    parser.add_argument(
        "--families",
        default="claude,gpt,gemini",
        help="Comma-separated family prefixes.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "validation/official_family_specialists.json"),
    )
    parser.add_argument(
        "--array",
        default=str(ROOT / "validation/official_family_specialists.npz"),
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    train["text"] = train["text"].fillna("").astype(str).map(normalize_text)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    targets = train["model"].map(label_to_index).to_numpy(dtype=np.int64)
    pool = load_pool(Path(args.pool), args.seed, args.max_per_model)
    official_keys = set(train["text"].map(text_key))
    pool = pool[~pool["_text_key"].isin(official_keys)].reset_index(drop=True)
    folds = list(
        GroupKFold(5).split(
            train,
            targets,
            train["prompt_id"],
        )
    )
    families = [
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    ]
    if not families:
        raise ValueError("--families cannot be empty")
    specialist_logp = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    specialist_meta: dict[str, object] = {}
    fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(folds):
        print(
            f"fold={fold} train={len(train_index)} valid={len(valid_index)}",
            flush=True,
        )
        for family in families:
            labels = [
                label
                for label in LABELS
                if label.startswith(family + "-")
            ]
            family_mask = train["model"].isin(labels).to_numpy()
            fit_mask = family_mask.copy()
            fit_mask[valid_index] = False
            family_train = train.loc[
                np.flatnonzero(fit_mask),
                ["text", "model"],
            ]
            family_valid = train.loc[
                np.flatnonzero(family_mask & np.isin(
                    np.arange(len(train)),
                    valid_index,
                )),
                ["text", "model"],
            ]
            if family_valid.empty:
                continue
            external = pool[pool["model"].isin(labels)]
            fit_texts = (
                external["text"].tolist()
                + family_train["text"].tolist()
            )
            fit_targets = np.concatenate(
                [
                    external["model"].to_numpy(),
                    family_train["model"].to_numpy(),
                ]
            )
            logp, classes, shape = build_specialist(
                fit_texts,
                fit_targets,
                family_valid["text"].tolist(),
                args.seed + fold,
                alpha=args.alpha,
            )
            columns = [LABELS.index(label) for label in classes]
            specialist_logp[
                np.ix_(family_valid.index.to_numpy(), columns)
            ] = logp
            specialist_meta[family] = {
                "classes": classes.tolist(),
                "feature_shape": shape.tolist(),
            }
            fold_metrics.append(
                {
                    "fold": fold,
                    "family": family,
                    "external_rows": len(external),
                    "official_rows": len(family_train),
                    "validation_rows": len(family_valid),
                    "feature_shape": shape.tolist(),
                }
            )

    logits = np.log(
        np.clip(
            np.load(
                ROOT / "features/qwen3_embedding_stack_blend_oof.npy"
            ),
            1e-12,
            None,
        )
    )
    base_probabilities = softmax(logits, axis=1)
    base_score = float(
        balanced_accuracy_score(targets, base_probabilities.argmax(axis=1))
    )
    family_rows = {
        family: train["model"].str.startswith(family + "-").to_numpy()
        for family in families
    }
    results: dict[str, object] = {
        "base_score": base_score,
        "family_base": {
            family: score_subset(
                targets,
                base_probabilities,
                family_rows[family],
            )
            for family in families
        },
        "weights": {},
        "fold_metrics": fold_metrics,
        "specialist": specialist_meta,
    }
    for weight in (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5):
        candidate_logits = logits.copy()
        for family in families:
            classes = np.asarray(specialist_meta[family]["classes"])
            candidate_logits = apply_family(
                candidate_logits,
                specialist_logp,
                classes,
                weight,
            )
        probabilities = softmax(candidate_logits, axis=1)
        results["weights"][str(weight)] = {
            "overall": float(
                balanced_accuracy_score(
                    targets,
                    probabilities.argmax(axis=1),
                )
            ),
            "family": {
                family: score_subset(
                    targets,
                    probabilities,
                    family_rows[family],
                )
                for family in families
            },
        }

    np.savez(
        args.array,
        log_probabilities=specialist_logp,
        classes=np.asarray(
            [
                specialist_meta[family]["classes"][0]
                for family in families
            ]
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved {output}")
    print(f"saved {args.array}")


if __name__ == "__main__":
    main()
