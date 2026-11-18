from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


FAMILIES = ("claude", "gpt", "gemini")


def normalize_text(value: object) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip()


def text_key(value: object) -> str:
    return hashlib.md5(normalize_text(value).encode("utf-8")).hexdigest()


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(f"Unexpected probability shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Base probabilities contain non-finite values")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Base probability rows contain zero mass")
    return values / totals


def load_pool(path: Path, seed: int, max_per_model: int) -> pd.DataFrame:
    pool = pd.read_parquet(path)
    required = {"text", "model"}
    if not required.issubset(pool.columns):
        raise ValueError(
            f"Pool is missing required columns: {sorted(required - set(pool.columns))}"
        )
    pool = pool.copy()
    pool["text"] = pool["text"].fillna("").astype(str).map(normalize_text)
    pool = pool[pool["text"].ne("") & pool["model"].isin(LABELS)]
    for column in (
        "flag_train_prompt_regen",
        "flag_near_train_prompt",
        "flag_gemini37_routing_risk",
        "flag_kb_proxy_after_ruling",
    ):
        if column in pool.columns:
            pool = pool[~pool[column].fillna(False).astype(bool)]
    if "gen_served_model" in pool.columns:
        served = pool["gen_served_model"].fillna("").astype(str)
        mismatch = served.ne("") & served.ne(pool["model"].astype(str))
        pool = pool[~mismatch]
    pool["_text_key"] = pool["text"].map(text_key)
    pool = pool.drop_duplicates("_text_key")
    if max_per_model > 0:
        pool = (
            pool.sample(frac=1.0, random_state=seed)
            .groupby("model", group_keys=False, sort=False)
            .head(max_per_model)
            .sort_index()
        )
    return pool.reset_index(drop=True)


def build_specialist(
    texts: list[str],
    target: np.ndarray,
    fit_texts: list[str],
    seed: int,
    alpha: float = 2e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        sublinear_tf=True,
        min_df=3,
        max_features=600_000,
        lowercase=False,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_features=300_000,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        lowercase=False,
        dtype=np.float32,
    )
    train_matrix = sparse.hstack(
        [
            char.fit_transform(texts),
            word.fit_transform(texts),
        ],
        format="csr",
    )
    query_matrix = sparse.hstack(
        [
            char.transform(fit_texts),
            word.transform(fit_texts),
        ],
        format="csr",
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=20,
        tol=None,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_matrix, target)
    classes = np.asarray(model.classes_, dtype=str)
    probabilities = model.predict_proba(query_matrix)
    log_probabilities = np.log(np.clip(probabilities, 1e-9, None))
    log_probabilities -= log_probabilities.mean(axis=1, keepdims=True)
    return log_probabilities, classes, np.asarray(train_matrix.shape)


def apply_specialist(
    base: np.ndarray,
    specialist: np.ndarray,
    classes: np.ndarray,
    weight: float,
) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    indices = [LABELS.index(value) for value in classes]
    lab_mass = np.logaddexp.reduce(result[:, indices], axis=1, keepdims=True)
    result[:, indices] = lab_mass + log_softmax(
        result[:, indices] + weight * specialist,
        axis=1,
    )
    return result


def score_subset(
    targets: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = probabilities[mask]
    local_labels = targets[mask]
    predictions = selected.argmax(axis=1)
    recalls = [
        float(np.mean(predictions[local_labels == label] == label))
        for label in np.unique(local_labels)
    ]
    return float(np.mean(recalls))


def evaluate(
    train: pd.DataFrame,
    base: np.ndarray,
    specialists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: list[float],
) -> dict[str, object]:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    targets = train["model"].map(label_to_index).to_numpy(dtype=np.int64)
    base_score = float(balanced_accuracy_score(targets, base.argmax(axis=1)))
    folds = list(
        GroupKFold(5).split(
            train,
            targets,
            train["prompt_id"],
        )
    )
    result: dict[str, object] = {
        "base": base_score,
        "folds": [],
        "weights": {},
    }
    base_logits = np.log(np.clip(base, 1e-12, None))
    for family, (specialist, classes, _) in specialists.items():
        row_mask = train["model"].str.startswith(family + "-").to_numpy()
        class_indices = [LABELS.index(value) for value in classes]
        standalone_predictions = np.asarray(classes)[
            specialist[row_mask].argmax(axis=1)
        ]
        standalone_accuracy = float(
            np.mean(
                standalone_predictions
                == train.loc[row_mask, "model"].to_numpy()
            )
        )
        result[family] = {
            "labels": classes.tolist(),
            "base": score_subset(targets, base, row_mask),
            "base_rows": int(row_mask.sum()),
            "standalone_accuracy": standalone_accuracy,
            "standalone_classes": class_indices,
            "families": {
                "all": {},
                "folds": {},
            },
        }
        for weight in weights:
            candidate = apply_specialist(
                base_logits,
                specialist,
                classes,
                weight,
            )
            family_score = score_subset(targets, candidate, row_mask)
            overall = float(balanced_accuracy_score(targets, candidate.argmax(axis=1)))
            result["weights"].setdefault(str(weight), {})[family] = {
                "family": family_score,
                "overall": overall,
            }
            result[family]["families"]["all"][str(weight)] = family_score
            for fold, (_, valid) in enumerate(folds):
                fold_mask = np.zeros(len(train), dtype=bool)
                fold_mask[valid] = True
                base_family = score_subset(
                    targets,
                    base,
                    fold_mask & row_mask,
                )
                result["folds"].append(
                    {
                        "fold": fold,
                        "family": family,
                        "weight": weight,
                        "base": base_family,
                        "family_candidate": score_subset(
                            targets,
                            candidate,
                            fold_mask & row_mask,
                        ),
                    }
                )

    result["combined"] = {}
    for weight in weights:
        candidate = base_logits.copy()
        for family in FAMILIES:
            specialist, classes, _ = specialists[family]
            candidate = apply_specialist(
                candidate,
                specialist,
                classes,
                weight,
            )
        family_scores = {}
        for family in FAMILIES:
            row_mask = train["model"].str.startswith(family + "-").to_numpy()
            family_scores[family] = score_subset(targets, candidate, row_mask)
        result["combined"][str(weight)] = {
            "overall": float(
                balanced_accuracy_score(targets, candidate.argmax(axis=1))
            ),
            "families": family_scores,
            "folds": [
                {
                    "fold": fold,
                    "score": score_subset(
                        targets,
                        candidate,
                        np.isin(np.arange(len(train)), valid),
                    ),
                }
                for fold, (_, valid) in enumerate(folds)
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument(
        "--pool",
        default=str(ROOT / "data" / "external" / "ntl_encoder_pool_122k.parquet"),
    )
    parser.add_argument(
        "--base",
        action="append",
        required=True,
        help="OOF probability file; repeat for multiple parents.",
    )
    parser.add_argument(
        "--weights",
        default="0,0.05,0.10,0.15,0.20,0.25,0.35,0.50",
    )
    parser.add_argument("--max-per-model", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument(
        "--specialist-cache",
        default=str(ROOT / "validation" / "pool_sibling_specialists.npz"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "validation" / "pool_sibling_specialists.json"),
    )
    parser.add_argument(
        "--save-candidate-dir",
        default=str(ROOT / "validation" / "pool_sibling_candidates"),
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    train["text"] = train["text"].fillna("").astype(str).map(normalize_text)
    train["_text_key"] = train["text"].map(text_key)
    pool = load_pool(Path(args.pool), args.seed, args.max_per_model)
    pool = pool[~pool["_text_key"].isin(set(train["_text_key"]))]
    print(
        f"train={len(train)} pool={len(pool)} "
        f"pool_counts={pool['model'].value_counts().sort_index().to_dict()}",
        flush=True,
    )

    specialists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cache_path = Path(args.specialist_cache)
    cache_valid = False
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_valid = (
                str(cache["pool"].item()) == str(Path(args.pool).resolve())
                and int(cache["max_per_model"].item()) == args.max_per_model
                and int(cache["seed"].item()) == args.seed
            )
            if cache_valid:
                for family in FAMILIES:
                    specialists[family] = (
                        cache[f"{family}_logp"],
                        cache[f"{family}_classes"],
                        cache[f"{family}_feature_shape"],
                    )
        if cache_valid:
            print(f"loaded specialist cache {cache_path}", flush=True)
        else:
            print(f"ignoring stale specialist cache {cache_path}", flush=True)

    if not cache_valid:
        for family in FAMILIES:
            labels = [
                label
                for label in LABELS
                if label.startswith(family + "-")
            ]
            local = pool[pool["model"].isin(labels)].copy()
            if local.empty:
                raise ValueError(f"No pool rows found for family {family}")
            log_probabilities, classes, shape = build_specialist(
                local["text"].tolist(),
                local["model"].to_numpy(),
                train["text"].tolist(),
                args.seed,
            )
            specialists[family] = (log_probabilities, classes, shape)
            print(
                f"specialist={family} rows={len(local)} "
                f"feature_shape={shape} classes={classes.tolist()}",
                flush=True,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            pool=np.asarray(str(Path(args.pool).resolve())),
            max_per_model=np.asarray(args.max_per_model),
            seed=np.asarray(args.seed),
            **{
                f"{family}_{name}": value
                for family, (logp, classes, shape) in specialists.items()
                for name, value in (
                    ("logp", logp),
                    ("classes", classes),
                    ("feature_shape", shape),
                )
            },
        )
        print(f"saved specialist cache {cache_path}", flush=True)

    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    reports: dict[str, object] = {}
    candidate_dir = Path(args.save_candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for base_path in args.base:
        base = normalize_probabilities(np.load(base_path))
        if len(base) != len(train):
            raise ValueError(f"Base row count differs: {base_path}")
        reports[base_path] = evaluate(train, base, specialists, weights)
        base_logits = np.log(np.clip(base, 1e-12, None))
        for weight in weights:
            candidate = base_logits.copy()
            for family in FAMILIES:
                specialist, classes, _ = specialists[family]
                candidate = apply_specialist(
                    candidate,
                    specialist,
                    classes,
                    weight,
                )
            output_path = (
                candidate_dir
                / f"{Path(base_path).stem}_siblings_w{weight:g}.npy"
            )
            np.save(output_path, softmax(candidate, axis=1))
        print(
            json.dumps(
                {
                    "base": base_path,
                    "score": reports[base_path]["base"],
                    "combined": reports[base_path]["combined"],
                },
                indent=2,
            ),
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "pool": args.pool,
                "pool_rows": len(pool),
                "max_per_model": args.max_per_model,
                "weights": weights,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
