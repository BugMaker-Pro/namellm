from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


DEFAULT_BASES = (
    "compact=features/compact_baseline_oof.npy",
    "style_external=features/styledistance_external_style_oof.npy",
    "embedding=features/qwen3_embedding_fusion7_oof.npy",
    "cluster_balanced=features/qwen3_embedding_cluster_balanced_fusion_oof.npy",
    "stack_blend=features/qwen3_embedding_stack_blend_oof.npy",
)


def parse_pairs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        result.append((name.strip(), path))
    return result


def probability_entropy(log_probabilities: np.ndarray) -> float:
    return float(
        -(np.exp(log_probabilities) * log_probabilities).sum(axis=1).mean()
    )


def match_entropy_temperature(
    log_probabilities: np.ndarray,
    target_entropy: float,
) -> float:
    low = 1e-3
    high = 1e3
    if (
        probability_entropy(log_softmax(log_probabilities * high, axis=1))
        > target_entropy
        or probability_entropy(log_softmax(log_probabilities * low, axis=1))
        < target_entropy
    ):
        raise ValueError("Reference entropy is outside the reachable range")
    for _ in range(100):
        middle = np.sqrt(low * high)
        current = probability_entropy(
            log_softmax(log_probabilities * middle, axis=1)
        )
        if current > target_entropy:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def load_features(
    train_csv: Path,
    flash_csv: Path,
    glm53_csv: Path,
    model_json: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict]:
    train = pd.read_csv(train_csv, dtype={"id": str})
    flash = pd.read_csv(flash_csv, dtype={"id": str}).set_index("id")
    glm53 = pd.read_csv(glm53_csv, dtype={"id": str}).set_index("id")
    model = json.loads(model_json.read_text(encoding="utf-8"))

    train = train.set_index("id")
    frame = (
        flash.add_prefix("flash:")
        .join(glm53.add_prefix("glm53:"), how="inner")
        .join(train[["prompt_id", "model"]], how="inner")
    )
    missing = train.index.difference(frame.index)
    extra = frame.index.difference(train.index)
    if len(missing) or len(extra):
        raise ValueError(
            "Feature/train ID mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    frame = frame.loc[train.index]
    for column in model["diff_features"]:
        frame[f"d:{column}"] = (
            frame[f"flash:{column}"] - frame[f"glm53:{column}"]
        )

    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array(
        [label_to_index[value] for value in frame["model"]],
        dtype=np.int64,
    )
    groups = frame["prompt_id"].astype(str).to_numpy()
    return frame, labels, groups, model["features"], model


def cross_score_features(
    flash_scores_path: Path,
    glm53_scores_path: Path,
    ids: list[str],
) -> pd.DataFrame:
    with (
        np.load(flash_scores_path, allow_pickle=True) as flash,
        np.load(glm53_scores_path, allow_pickle=True) as glm53,
    ):
        if not np.array_equal(flash["ids"], glm53["ids"]):
            raise ValueError("Flash and GLM score IDs differ")
        if not np.array_equal(flash["offsets"], glm53["offsets"]):
            raise ValueError("Flash and GLM token offsets differ")
        source_ids = flash["ids"].astype(str).tolist()
        positions = {value: index for index, value in enumerate(source_ids)}
        offsets = flash["offsets"]
        flash_scores = flash["scores"]
        glm_scores = glm53["scores"]

    rows = []
    quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    entropy_quantiles = (0.10, 0.50, 0.90)
    for passage_id in ids:
        index = positions[str(passage_id)]
        left = flash_scores[offsets[index] : offsets[index + 1]]
        right = glm_scores[offsets[index] : offsets[index + 1]]
        delta_lp = left[:, 0] - right[:, 0]
        delta_entropy = left[:, 1] - right[:, 1]
        delta_logrank = np.log1p(left[:, 3]) - np.log1p(right[:, 3])
        length = len(delta_lp)
        half = length // 2
        q = np.quantile(delta_lp, quantiles)
        q_entropy = np.quantile(delta_entropy, entropy_quantiles)
        positive = delta_lp > 0
        transition = (
            float(np.mean(positive[1:] != positive[:-1]))
            if length > 1
            else 0.0
        )
        positive_run = 0
        negative_run = 0
        max_positive_run = 0
        max_negative_run = 0
        for value in positive:
            if value:
                positive_run += 1
                negative_run = 0
            else:
                negative_run += 1
                positive_run = 0
            max_positive_run = max(max_positive_run, positive_run)
            max_negative_run = max(max_negative_run, negative_run)
        first = delta_lp[: max(half, 1)].mean()
        second = (
            delta_lp[half:].mean()
            if length > half
            else delta_lp.mean()
        )
        rows.append(
            {
                "id": str(passage_id),
                "cd_mean": delta_lp.mean(),
                "cd_std": delta_lp.std(),
                "cd_min": delta_lp.min(),
                "cd_max": delta_lp.max(),
                "cd_q05": q[0],
                "cd_q10": q[1],
                "cd_q25": q[2],
                "cd_q50": q[3],
                "cd_q75": q[4],
                "cd_q90": q[5],
                "cd_q95": q[6],
                "cd_pos": positive.mean(),
                "cd_gt1": float((delta_lp > 1).mean()),
                "cd_ltm1": float((delta_lp < -1).mean()),
                "cd_absmean": float(np.abs(delta_lp).mean()),
                "cd_absq90": float(np.quantile(np.abs(delta_lp), 0.90)),
                "cd_first": first,
                "cd_second": second,
                "cd_delta": second - first,
                "cd_trans": transition,
                "cd_maxpos": max_positive_run,
                "cd_maxneg": max_negative_run,
                "ce_mean": delta_entropy.mean(),
                "ce_std": delta_entropy.std(),
                "ce_q10": q_entropy[0],
                "ce_q50": q_entropy[1],
                "ce_q90": q_entropy[2],
                "ce_pos": float((delta_entropy > 0).mean()),
                "cr_mean": delta_logrank.mean(),
                "cr_std": delta_logrank.std(),
                "cr_q10": float(np.quantile(delta_logrank, 0.10)),
                "cr_q50": float(np.quantile(delta_logrank, 0.50)),
                "cr_q90": float(np.quantile(delta_logrank, 0.90)),
                "corr_lp": (
                    float(np.corrcoef(left[:, 0], right[:, 0])[0, 1])
                    if length > 1
                    else 0.0
                ),
                "corr_ent": (
                    float(np.corrcoef(left[:, 1], right[:, 1])[0, 1])
                    if length > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_metric_bundle(
    probability: np.ndarray,
    y: np.ndarray,
    glm_indices: list[int],
) -> dict:
    rows = np.flatnonzero(np.isin(y, glm_indices))
    binary = (y[rows] == glm_indices[0]).astype(np.int64)
    predictions = (probability[rows] >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(binary, predictions)
        ),
        "accuracy": float((predictions == binary).mean()),
        "roc_auc": float(roc_auc_score(binary, probability[rows])),
    }


def metric_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = balanced_accuracy_score(y_true, predictions)
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(float((predictions[mask] == label).mean()))
    return {
        "balanced_accuracy": float(score),
        "accuracy": float((predictions == y_true).mean()),
        "mean_class_recall": float(np.mean(recalls)),
        "min_class_recall": float(np.min(recalls)),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=list(range(len(LABELS))),
            )
        ),
    }


def inner_split(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )
    for fit_pos, valid_pos in splitter.split(indices, y[indices], groups[indices]):
        fit = indices[fit_pos]
        valid = indices[valid_pos]
        if len(np.unique(y[fit])) >= 2 and len(np.unique(y[valid])) >= 2:
            return fit, valid
    return None


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    c_value: float,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x[indices])
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
    )
    model.fit(scaler.transform(x[indices]), y[indices])
    return scaler, model


def predict_classes(
    x: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
    wanted_labels: list[int],
) -> np.ndarray:
    probabilities = model.predict_proba(scaler.transform(x[indices]))
    output = np.zeros((len(indices), len(wanted_labels)), dtype=np.float64)
    position = {label: index for index, label in enumerate(wanted_labels)}
    for source, label in enumerate(model.classes_):
        if int(label) in position:
            output[:, position[int(label)]] = probabilities[:, source]
    return output


def nested_multiclass_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_values: list[float],
    outer_splits: int,
) -> tuple[np.ndarray, list[dict]]:
    probabilities = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    choices: list[dict] = []
    outer = list(GroupKFold(outer_splits).split(x, y, groups))
    for outer_index, (train_index, valid_index) in enumerate(outer):
        scored = []
        for order, c_value in enumerate(c_values):
            actual = np.array([], dtype=np.int64)
            predicted = np.array([], dtype=np.int64)
            for inner_index, seed in enumerate((1701, 2701, 3701)):
                split = inner_split(
                    train_index,
                    y,
                    groups,
                    20261116 + 100 * outer_index + seed,
                )
                if split is None:
                    continue
                fit_index, valid_part = split
                scaler, model = fit_classifier(
                    x,
                    y,
                    fit_index,
                    c_value,
                )
                local = predict_classes(
                    x,
                    valid_part,
                    scaler,
                    model,
                    list(range(len(LABELS))),
                )
                actual = np.concatenate([actual, y[valid_part]])
                predicted = np.concatenate(
                    [predicted, local.argmax(axis=1)]
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = balanced_accuracy_score(actual, predicted)
            scored.append((score, -order, c_value))
        selected_c = max(scored)[2]
        scaler, model = fit_classifier(x, y, train_index, selected_c)
        probabilities[valid_index] = predict_classes(
            x,
            valid_index,
            scaler,
            model,
            list(range(len(LABELS))),
        )
        choices.append(
            {
                "fold": outer_index,
                "selected_c": selected_c,
                "inner_scores": [
                    {"c": c_value, "balanced_accuracy": score}
                    for score, _, c_value in scored
                ],
            }
        )
    return probabilities, choices


def nested_pairwise_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    glm_indices: list[int],
    c_values: list[float],
    outer_splits: int,
) -> tuple[np.ndarray, list[dict]]:
    glm_indices = sorted(glm_indices)
    rows = np.flatnonzero(np.isin(y, glm_indices))
    pair_x = x[rows]
    pair_groups = groups[rows]
    positive = glm_indices[0]
    pair_y = (y[rows] == positive).astype(np.int64)
    pair_probability = np.full(len(y), np.nan, dtype=np.float64)
    choices: list[dict] = []
    outer = list(
        GroupKFold(outer_splits).split(
            pair_x,
            pair_y,
            pair_groups,
        )
    )
    for outer_index, (train_index, valid_index) in enumerate(outer):
        scored = []
        for order, c_value in enumerate(c_values):
            actual = []
            predicted = []
            for seed in (1701, 2701, 3701):
                split = inner_split(
                    train_index,
                    pair_y,
                    pair_groups,
                    20261116 + 100 * outer_index + seed,
                )
                if split is None:
                    continue
                fit_index, valid_part = split
                scaler, model = fit_classifier(
                    pair_x,
                    pair_y,
                    fit_index,
                    c_value,
                )
                local = model.predict_proba(
                    scaler.transform(pair_x[valid_part])
                )[:, 1]
                actual.extend(pair_y[valid_part].tolist())
                predicted.extend((local >= 0.5).astype(int).tolist())
            score = balanced_accuracy_score(actual, predicted)
            scored.append((score, -order, c_value))
        selected_c = max(scored)[2]
        scaler, model = fit_classifier(
            pair_x,
            pair_y,
            train_index,
            selected_c,
        )
        pair_probability[rows[valid_index]] = model.predict_proba(
            scaler.transform(pair_x[valid_index])
        )[:, 1]
        choices.append(
            {
                "fold": outer_index,
                "selected_c": selected_c,
                "inner_scores": [
                    {"c": c_value, "balanced_accuracy": score}
                    for score, _, c_value in scored
                ],
            }
        )
    return pair_probability, choices


def multiclass_log_evidence(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, 1e-12, None))


def pairwise_log_evidence(
    pair_probability: np.ndarray,
    glm_indices: list[int],
) -> np.ndarray:
    glm_indices = sorted(glm_indices)
    evidence = np.zeros((len(pair_probability), len(LABELS)), dtype=np.float64)
    logit = np.log(
        np.clip(pair_probability, 1e-12, 1 - 1e-12)
        / np.clip(1 - pair_probability, 1e-12, 1)
    )
    evidence[:, glm_indices[0]] = 0.5 * logit
    evidence[:, glm_indices[1]] = -0.5 * logit
    return evidence


def selective_blend(
    base_probabilities: np.ndarray,
    evidence: np.ndarray,
    y: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    weight: float,
) -> dict:
    base_log = np.log(np.clip(base_probabilities, 1e-12, None))
    base_predictions = base_log.argmax(axis=1)
    if coverage >= 1:
        order = np.arange(len(y))
    else:
        mass = base_probabilities[:, glm_indices].sum(axis=1)
        order = np.argsort(-mass, kind="stable")
        order = order[: max(1, int(round(coverage * len(y))))]
    combined = base_log.copy()
    combined[order] += weight * evidence[order]
    predictions = combined.argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_score = balanced_accuracy_score(y, base_predictions)
        score = balanced_accuracy_score(y, predictions)
    return {
        "coverage": coverage,
        "weight": weight,
        "selected": int(len(order)),
        "base_balanced_accuracy": float(base_score),
        "balanced_accuracy": float(score),
        "delta": float(score - base_score),
        "changed": float((predictions != base_predictions).mean()),
        "changed_selected": float(
            (predictions[order] != base_predictions[order]).mean()
        ),
        "selected_base_accuracy": float(
            (base_predictions[order] == y[order]).mean()
        ),
        "selected_accuracy": float((predictions[order] == y[order]).mean()),
    }


def sequential_pair_gate(
    base_probabilities: np.ndarray,
    multiclass_evidence: np.ndarray,
    pair_probability: np.ndarray,
    y: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    multiclass_weight: float,
) -> dict:
    base_log = np.log(np.clip(base_probabilities, 1e-12, None))
    base_predictions = base_log.argmax(axis=1)
    mass = base_probabilities[:, glm_indices].sum(axis=1)
    order = np.argsort(-mass, kind="stable")
    order = order[: max(1, int(round(coverage * len(y))))]
    combined = base_log.copy()
    combined[order] += multiclass_weight * multiclass_evidence[order]
    first_pass = combined.argmax(axis=1)
    gate = order[np.isin(first_pass[order], glm_indices)]
    predictions = first_pass.copy()
    predictions[gate] = np.where(
        pair_probability[gate] >= 0.5,
        glm_indices[0],
        glm_indices[1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_score = balanced_accuracy_score(y, base_predictions)
        score = balanced_accuracy_score(y, predictions)
    return {
        "coverage": coverage,
        "multiclass_weight": multiclass_weight,
        "selected": int(len(order)),
        "gated": int(len(gate)),
        "base_balanced_accuracy": float(base_score),
        "balanced_accuracy": float(score),
        "delta": float(score - base_score),
        "changed": float((predictions != base_predictions).mean()),
        "changed_selected": float(
            (predictions[order] != base_predictions[order]).mean()
        ),
        "gated_base_accuracy": float(
            (base_predictions[gate] == y[gate]).mean()
        ),
        "gated_accuracy": float((predictions[gate] == y[gate]).mean()),
    }


def sparse_two_row_probe(
    base_probabilities: np.ndarray,
    evidence: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    weight: float,
    seeds: int,
) -> dict:
    prompt_rows: dict[str, np.ndarray] = {}
    for group in np.unique(groups):
        prompt_rows[str(group)] = np.flatnonzero(groups == group)
    rng = np.random.default_rng(20261116)
    deltas = []
    base_scores = []
    blended_scores = []
    for _ in range(seeds):
        selected_rows = []
        for rows in prompt_rows.values():
            take = min(2, len(rows))
            selected_rows.extend(rng.choice(rows, size=take, replace=False))
        subset = np.sort(np.asarray(selected_rows, dtype=np.int64))
        base = selective_blend(
            base_probabilities[subset],
            evidence[subset],
            y[subset],
            glm_indices,
            coverage,
            weight,
        )
        base_scores.append(base["base_balanced_accuracy"])
        blended_scores.append(base["balanced_accuracy"])
        deltas.append(base["delta"])
    deltas_array = np.asarray(deltas)
    return {
        "seeds": seeds,
        "coverage": coverage,
        "weight": weight,
        "mean_base": float(np.mean(base_scores)),
        "mean_blended": float(np.mean(blended_scores)),
        "mean_delta": float(np.mean(deltas_array)),
        "std_delta": float(np.std(deltas_array)),
        "win_rate": float((deltas_array > 0).mean()),
    }


def sequential_sparse_probe(
    base_probabilities: np.ndarray,
    multiclass_evidence: np.ndarray,
    pair_probability: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    multiclass_weight: float,
    seeds: int,
) -> dict:
    prompt_rows: dict[str, np.ndarray] = {}
    for group in np.unique(groups):
        prompt_rows[str(group)] = np.flatnonzero(groups == group)
    rng = np.random.default_rng(20261116)
    deltas = []
    base_scores = []
    blended_scores = []
    for _ in range(seeds):
        selected_rows = []
        for rows in prompt_rows.values():
            take = min(2, len(rows))
            selected_rows.extend(rng.choice(rows, size=take, replace=False))
        subset = np.sort(np.asarray(selected_rows, dtype=np.int64))
        result = sequential_pair_gate(
            base_probabilities[subset],
            multiclass_evidence[subset],
            pair_probability[subset],
            y[subset],
            glm_indices,
            coverage,
            multiclass_weight,
        )
        base_scores.append(result["base_balanced_accuracy"])
        blended_scores.append(result["balanced_accuracy"])
        deltas.append(result["delta"])
    deltas_array = np.asarray(deltas)
    return {
        "seeds": seeds,
        "coverage": coverage,
        "multiclass_weight": multiclass_weight,
        "mean_base": float(np.mean(base_scores)),
        "mean_blended": float(np.mean(blended_scores)),
        "mean_delta": float(np.mean(deltas_array)),
        "std_delta": float(np.std(deltas_array)),
        "win_rate": float((deltas_array > 0).mean()),
    }


def per_fold_blend(
    base_probabilities: np.ndarray,
    evidence: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    weight: float,
    splits: int,
) -> dict:
    folds = []
    for index, (_, valid) in enumerate(
        GroupKFold(splits).split(base_probabilities, y, groups)
    ):
        result = selective_blend(
            base_probabilities[valid],
            evidence[valid],
            y[valid],
            glm_indices,
            coverage,
            weight,
        )
        result["fold"] = index
        folds.append(result)
    return {
        "mean_delta": float(np.mean([row["delta"] for row in folds])),
        "std_delta": float(np.std([row["delta"] for row in folds])),
        "folds_improved": int(sum(row["delta"] > 0 for row in folds)),
        "fold_count": len(folds),
        "folds": folds,
    }


def sequential_per_fold(
    base_probabilities: np.ndarray,
    multiclass_evidence: np.ndarray,
    pair_probability: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    glm_indices: list[int],
    coverage: float,
    multiclass_weight: float,
    splits: int,
) -> dict:
    folds = []
    for index, (_, valid) in enumerate(
        GroupKFold(splits).split(base_probabilities, y, groups)
    ):
        result = sequential_pair_gate(
            base_probabilities[valid],
            multiclass_evidence[valid],
            pair_probability[valid],
            y[valid],
            glm_indices,
            coverage,
            multiclass_weight,
        )
        result["fold"] = index
        folds.append(result)
    return {
        "mean_delta": float(np.mean([row["delta"] for row in folds])),
        "std_delta": float(np.std([row["delta"] for row in folds])),
        "folds_improved": int(sum(row["delta"] > 0 for row in folds)),
        "fold_count": len(folds),
        "folds": folds,
    }


def coverage_weight_grid(
    base_probabilities: np.ndarray,
    evidence: np.ndarray,
    y: np.ndarray,
    glm_indices: list[int],
) -> list[dict]:
    rows = []
    for coverage in (0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        for weight in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
            rows.append(
                selective_blend(
                    base_probabilities,
                    evidence,
                    y,
                    glm_indices,
                    coverage,
                    weight,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -abs(row["weight"]),
            row["coverage"],
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--flash-features", required=True)
    parser.add_argument("--glm53-features", required=True)
    parser.add_argument("--flash-scores")
    parser.add_argument("--glm53-scores")
    parser.add_argument(
        "--model",
        default=str(Path("D:/tmp/glm_likelihood_v1/ll_model.json")),
    )
    parser.add_argument(
        "--base",
        action="append",
        default=[],
        help="NAME=PATH probability matrix; may be repeated.",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--sparse-seeds", type=int, default=100)
    parser.add_argument(
        "--output",
        default="features/glm_likelihood_eval.json",
    )
    parser.add_argument(
        "--multiclass-oof",
        default="features/glm_likelihood_member_oof.npy",
    )
    parser.add_argument(
        "--pairwise-oof",
        default="features/glm_likelihood_pairwise_oof.npy",
    )
    parser.add_argument(
        "--pairwise-cross-oof",
        default="features/glm_likelihood_pairwise_cross_oof.npy",
    )
    parser.add_argument(
        "--pairwise-model-output",
        help="Optional JSON model for GLM class-internal reranking.",
    )
    parser.add_argument("--pairwise-c", type=float, default=0.30)
    parser.add_argument(
        "--multiclass-model-output",
        help="Optional V1-compatible multiclass likelihood model JSON.",
    )
    parser.add_argument("--multiclass-c", type=float, default=0.03)
    parser.add_argument("--multiclass-w", type=float, default=1.0)
    args = parser.parse_args()

    frame, y, groups, feature_columns, reference_model = load_features(
        Path(args.train_csv),
        Path(args.flash_features),
        Path(args.glm53_features),
        Path(args.model),
    )
    if list(LABELS) != list(reference_model["classes"]):
        raise ValueError("Model and project label orders differ")
    cross_columns: list[str] = []
    if bool(args.flash_scores) != bool(args.glm53_scores):
        raise ValueError(
            "--flash-scores and --glm53-scores must be supplied together"
        )
    if args.flash_scores:
        cross = cross_score_features(
            Path(args.flash_scores),
            Path(args.glm53_scores),
            frame.index.astype(str).tolist(),
        ).set_index("id")
        frame = frame.join(cross, how="left")
        if frame[cross.columns].isna().any().any():
            raise ValueError("Cross-score features have missing rows")
        cross_columns = cross.columns.tolist()
    glm_indices = [LABELS.index("glm-5.3"), LABELS.index("glm-5.3-flash")]
    glm_mask = np.isin(y, glm_indices)
    c_values = [0.01, 0.03, 0.05, 0.1, 0.3, 1.0]

    x = frame[feature_columns].to_numpy(dtype=np.float64)
    multiclass, multiclass_choices = nested_multiclass_oof(
        x,
        y,
        groups,
        c_values,
        args.outer_splits,
    )
    pair_base, pairwise_base_choices = nested_pairwise_oof(
        x,
        y,
        groups,
        glm_indices,
        c_values,
        args.outer_splits,
    )
    pair_cross = pair_base
    pairwise_cross_choices = pairwise_base_choices
    if cross_columns:
        pair_cross, pairwise_cross_choices = nested_pairwise_oof(
            frame[feature_columns + cross_columns].to_numpy(dtype=np.float64),
            y,
            groups,
            glm_indices,
            c_values,
            args.outer_splits,
        )
    if args.pairwise_model_output:
        pair_mask = np.isin(y, glm_indices)
        pair_x = frame.loc[pair_mask, feature_columns].to_numpy(
            dtype=np.float64
        )
        pair_y = (y[pair_mask] == glm_indices[0]).astype(np.int64)
        scaler = StandardScaler().fit(pair_x)
        classifier = LogisticRegression(
            C=args.pairwise_c,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
        )
        classifier.fit(scaler.transform(pair_x), pair_y)
        model_path = Path(args.pairwise_model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(
            json.dumps(
                {
                    "format": "ntl-glm-pairwise-1",
                    "features": feature_columns,
                    "classes": [
                        LABELS[glm_indices[0]],
                        LABELS[glm_indices[1]],
                    ],
                    "positive_label": LABELS[glm_indices[0]],
                    "C": args.pairwise_c,
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": classifier.coef_[0].tolist(),
                    "intercept": float(classifier.intercept_[0]),
                    "train_rows": int(pair_mask.sum()),
                    "source": {
                        "flash_features": str(args.flash_features),
                        "glm53_features": str(args.glm53_features),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {model_path}")
    if args.multiclass_model_output:
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(
            C=args.multiclass_c,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
        )
        classifier.fit(scaler.transform(x), y)
        model_path = Path(args.multiclass_model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tokens_per_char = float(
            (
                frame["flash:n_tok"]
                / frame.index.map(
                    pd.read_csv(
                        args.train_csv,
                        dtype={"id": str},
                    ).set_index("id")["text"].fillna("").astype(str).str.len()
                ).to_numpy()
            ).mean()
        )
        model_path.write_text(
            json.dumps(
                {
                    "format": "ntl-v1-ll-model-1",
                    "features": feature_columns,
                    "diff_features": reference_model["diff_features"],
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": classifier.coef_.tolist(),
                    "intercept": classifier.intercept_.tolist(),
                    "classes": list(LABELS),
                    "C": args.multiclass_c,
                    "W": args.multiclass_w,
                    "entropy_ref": float(reference_model["entropy_ref"]),
                    "max_text_tokens": 160,
                    "tok_per_char": tokens_per_char,
                    "source": {
                        "flash_features": str(args.flash_features),
                        "glm53_features": str(args.glm53_features),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {model_path}")

    multiclass_recalls = {}
    multiclass_predictions = multiclass.argmax(axis=1)
    for label in glm_indices:
        mask = y == label
        multiclass_recalls[LABELS[label]] = float(
            (multiclass_predictions[mask] == label).mean()
        )
    multiclass_metrics = metric_summary(y, multiclass)
    multiclass_metrics["glm_pair_recalls"] = multiclass_recalls
    multiclass_metrics["glm_pair_balanced_accuracy"] = float(
        balanced_accuracy_score(
            y[glm_mask],
            multiclass_predictions[glm_mask],
        )
    )

    evidence_variants = {
        "multiclass": multiclass_log_evidence(multiclass),
        "pairwise_base": pairwise_log_evidence(pair_base, glm_indices),
        "pairwise_cross": pairwise_log_evidence(pair_cross, glm_indices),
    }
    base_entries = parse_pairs(args.base or list(DEFAULT_BASES))
    base_results = {}
    for name, path in base_entries:
        base = np.load(path).astype(np.float64)
        if base.shape != (len(y), len(LABELS)):
            raise ValueError(f"{path} has shape {base.shape}")
        normalized_base = base / base.sum(axis=1, keepdims=True)
        base_log = np.log(np.clip(normalized_base, 1e-12, None))
        temperature = match_entropy_temperature(
            base_log,
            float(reference_model["entropy_ref"]),
        )
        calibrated_weight = float(reference_model["W"]) / temperature
        base_results[name] = {
            "path": str(path),
            "base": metric_summary(y, normalized_base),
            "entropy_calibration": {
                "temperature": temperature,
                "reference_weight": float(reference_model["W"]),
                "effective_multiclass_weight": calibrated_weight,
            },
            "variants": {},
            "sequential": {},
        }
        for variant_name, evidence in evidence_variants.items():
            grid = coverage_weight_grid(base, evidence, y, glm_indices)
            fixed = selective_blend(
                base,
                evidence,
                y,
                glm_indices,
                0.30,
                1.0,
            )
            sparse = sparse_two_row_probe(
                base,
                evidence,
                y,
                groups,
                glm_indices,
                0.30,
                1.0,
                args.sparse_seeds,
            )
            folds = per_fold_blend(
                base,
                evidence,
                y,
                groups,
                glm_indices,
                0.30,
                1.0,
                args.outer_splits,
            )
            base_results[name]["variants"][variant_name] = {
                "best_grid_row": grid[0],
                "fixed_30pct_weight1": fixed,
                "sparse_two_row": sparse,
                "per_fold_fixed": folds,
            }
        for multiclass_weight in (0.20, 0.35, 0.50, 0.75, 1.0):
            key = f"weight_{multiclass_weight:g}"
            base_results[name]["sequential"][key] = {
                "fixed_30pct": sequential_pair_gate(
                    base,
                    evidence_variants["multiclass"],
                    pair_base,
                    y,
                    glm_indices,
                    0.30,
                    multiclass_weight,
                )
            }
        base_results[name]["sequential"]["weight_0.35"].update(
            {
                "sparse_two_row": sequential_sparse_probe(
                    base,
                    evidence_variants["multiclass"],
                    pair_base,
                    y,
                    groups,
                    glm_indices,
                    0.30,
                    0.35,
                    args.sparse_seeds,
                ),
                "per_fold_fixed": sequential_per_fold(
                    base,
                    evidence_variants["multiclass"],
                    pair_base,
                    y,
                    groups,
                    glm_indices,
                    0.30,
                    0.35,
                    args.outer_splits,
                ),
            }
        )
        base_results[name]["sequential"]["entropy_calibrated"] = {
            "weight": calibrated_weight,
            "fixed_30pct": sequential_pair_gate(
                base,
                evidence_variants["multiclass"],
                pair_base,
                y,
                glm_indices,
                0.30,
                calibrated_weight,
            ),
            "sparse_two_row": sequential_sparse_probe(
                base,
                evidence_variants["multiclass"],
                pair_base,
                y,
                groups,
                glm_indices,
                0.30,
                calibrated_weight,
                args.sparse_seeds,
            ),
            "per_fold_fixed": sequential_per_fold(
                base,
                evidence_variants["multiclass"],
                pair_base,
                y,
                groups,
                glm_indices,
                0.30,
                calibrated_weight,
                args.outer_splits,
            ),
        }

    report = {
        "train_rows": int(len(y)),
        "prompt_rows": int(len(np.unique(groups))),
        "feature_count": int(len(feature_columns)),
        "class_counts": {
            LABELS[index]: int(count)
            for index, count in sorted(Counter(y.tolist()).items())
        },
        "multiclass": {
            "metrics": multiclass_metrics,
            "nested_c": multiclass_choices,
        },
        "pairwise": {
            "base": {
                "metrics": pairwise_metric_bundle(
                    pair_base,
                    y,
                    glm_indices,
                ),
                "nested_c": pairwise_base_choices,
            },
            "cross": {
                "metrics": pairwise_metric_bundle(
                    pair_cross,
                    y,
                    glm_indices,
                ),
                "nested_c": pairwise_cross_choices,
            },
            "positive_label": LABELS[glm_indices[0]],
            "negative_label": LABELS[glm_indices[1]],
        },
        "bases": base_results,
    }

    multiclass_path = Path(args.multiclass_oof)
    multiclass_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(multiclass_path, multiclass)
    pair_base_path = Path(args.pairwise_oof)
    pair_base_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pair_base_path, pair_base)
    pair_cross_path = Path(args.pairwise_cross_oof)
    pair_cross_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pair_cross_path, pair_cross)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "multiclass "
        f"balanced={multiclass_metrics['balanced_accuracy']:.4f} "
        f"glm_pair={multiclass_metrics['glm_pair_balanced_accuracy']:.4f}"
    )
    for variant, probability in (
        ("pairwise_base", pair_base),
        ("pairwise_cross", pair_cross),
    ):
        metrics = pairwise_metric_bundle(probability, y, glm_indices)
        print(
            f"{variant} "
            f"balanced={metrics['balanced_accuracy']:.4f} "
            f"auc={metrics['roc_auc']:.4f}"
        )
    for name, result in base_results.items():
        for variant, values in result["variants"].items():
            fixed = values["fixed_30pct_weight1"]
            sparse = values["sparse_two_row"]
            folds = values["per_fold_fixed"]
            print(
                f"{name} {variant}: "
                f"base={fixed['base_balanced_accuracy']:.4f} "
                f"fixed={fixed['balanced_accuracy']:.4f} "
                f"delta={fixed['delta']:+.4f} "
                f"sparse={sparse['mean_delta']:+.4f} "
                f"sparse_win={sparse['win_rate']:.2f} "
                f"folds={folds['folds_improved']}/{folds['fold_count']}"
            )
        sequential = result["sequential"]["weight_0.35"]
        fixed = sequential["fixed_30pct"]
        sparse = sequential["sparse_two_row"]
        folds = sequential["per_fold_fixed"]
        print(
            f"{name} multiclass_then_pairwise: "
            f"base={fixed['base_balanced_accuracy']:.4f} "
            f"fixed={fixed['balanced_accuracy']:.4f} "
            f"delta={fixed['delta']:+.4f} "
            f"sparse={sparse['mean_delta']:+.4f} "
            f"sparse_win={sparse['win_rate']:.2f} "
            f"folds={folds['folds_improved']}/{folds['fold_count']}"
        )
        calibrated = result["sequential"]["entropy_calibrated"]
        fixed = calibrated["fixed_30pct"]
        sparse = calibrated["sparse_two_row"]
        folds = calibrated["per_fold_fixed"]
        print(
            f"{name} entropy_calibrated: "
            f"temperature={result['entropy_calibration']['temperature']:.3f} "
            f"weight={calibrated['weight']:.3f} "
            f"fixed={fixed['balanced_accuracy']:.4f} "
            f"delta={fixed['delta']:+.4f} "
            f"sparse={sparse['mean_delta']:+.4f} "
            f"sparse_win={sparse['win_rate']:.2f} "
            f"folds={folds['folds_improved']}/{folds['fold_count']}"
        )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
