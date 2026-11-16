from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score


def entropy(log_probabilities: np.ndarray) -> float:
    return float(
        -(np.exp(log_probabilities) * log_probabilities).sum(axis=1).mean()
    )


def match_temperature(
    log_probabilities: np.ndarray,
    target_entropy: float,
) -> float:
    low = 1e-3
    high = 1e3
    if (
        entropy(log_softmax(log_probabilities * high, axis=1))
        > target_entropy
        or entropy(log_softmax(log_probabilities * low, axis=1))
        < target_entropy
    ):
        raise ValueError("Reference entropy is outside the reachable range")
    for _ in range(100):
        middle = np.sqrt(low * high)
        current = entropy(
            log_softmax(log_probabilities * middle, axis=1)
        )
        if current > target_entropy:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def member_log_probabilities(
    model: dict,
    flash_features: pd.DataFrame,
    glm53_features: pd.DataFrame,
    ids: list[str],
) -> np.ndarray:
    frame = (
        flash_features.set_index("id")
        .add_prefix("flash:")
        .join(
            glm53_features.set_index("id").add_prefix("glm53:"),
            how="inner",
        )
    )
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"Likelihood features missing {len(missing)} IDs")
    frame = frame.loc[ids]
    for column in model["diff_features"]:
        frame[f"d:{column}"] = (
            frame[f"flash:{column}"] - frame[f"glm53:{column}"]
        )
    values = frame[model["features"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Likelihood features contain non-finite values")
    scaled = (
        values - np.asarray(model["mean"], dtype=np.float64)
    ) / np.asarray(model["scale"], dtype=np.float64)
    logits = (
        scaled @ np.asarray(model["coef"], dtype=np.float64).T
        + np.asarray(model["intercept"], dtype=np.float64)
    )
    return log_softmax(logits, axis=1)


def pairwise_probabilities(
    model: dict,
    flash_features: pd.DataFrame,
    glm53_features: pd.DataFrame,
    ids: list[str],
) -> np.ndarray:
    frame = (
        flash_features.set_index("id")
        .add_prefix("flash:")
        .join(
            glm53_features.set_index("id").add_prefix("glm53:"),
            how="inner",
        )
    )
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"Pairwise features missing {len(missing)} IDs")
    frame = frame.loc[ids]
    for column in ("lp_mean", "lp_q50", "ent_mean", "top1", "logrank_mean", "fdg"):
        frame[f"d:{column}"] = (
            frame[f"flash:{column}"] - frame[f"glm53:{column}"]
        )
    values = frame[model["features"]].to_numpy(dtype=np.float64)
    scaled = (
        values - np.asarray(model["mean"], dtype=np.float64)
    ) / np.asarray(model["scale"], dtype=np.float64)
    logit = (
        scaled @ np.asarray(model["coef"], dtype=np.float64)
        + float(model["intercept"])
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-logits", required=True)
    parser.add_argument("--flash-features", required=True)
    parser.add_argument("--glm53-features", required=True)
    parser.add_argument("--ll-model", required=True)
    parser.add_argument("--pair-model", required=True)
    parser.add_argument("--coverage", type=float, default=0.30)
    parser.add_argument(
        "--multiclass-weight",
        type=float,
        help="Override reference weight before entropy temperature matching.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--labels-csv")
    args = parser.parse_args()

    ll_model = json.loads(Path(args.ll_model).read_text(encoding="utf-8"))
    pair_model = json.loads(
        Path(args.pair_model).read_text(encoding="utf-8")
    )
    if pair_model.get("format") != "ntl-glm-pairwise-1":
        raise ValueError("Unsupported pairwise model format")
    flash_features = pd.read_csv(
        args.flash_features,
        dtype={"id": str},
    )
    glm53_features = pd.read_csv(
        args.glm53_features,
        dtype={"id": str},
    )
    with np.load(args.final_logits, allow_pickle=False) as payload:
        ids = payload["ids"].astype(str).tolist()
        labels = payload["labels"].astype(str).tolist()
        base_logits = payload["logits"].astype(np.float64)
    if base_logits.shape != (len(ids), len(labels)):
        raise ValueError("Final logits do not match IDs and labels")

    common_ids = [
        value
        for value in ids
        if value in set(flash_features["id"]) & set(glm53_features["id"])
    ]
    if not common_ids:
        raise ValueError("No scored IDs have complete GLM features")
    flash = flash_features[flash_features["id"].isin(common_ids)].copy()
    glm53 = glm53_features[glm53_features["id"].isin(common_ids)].copy()

    base_log_probabilities = log_softmax(base_logits, axis=1)
    base_predictions = base_log_probabilities.argmax(axis=1)
    likelihood = member_log_probabilities(
        ll_model,
        flash,
        glm53,
        common_ids,
    )
    id_position = {value: index for index, value in enumerate(ids)}
    feature_rows = np.array(
        [id_position[value] for value in common_ids],
        dtype=np.int64,
    )
    likelihood_by_row = np.zeros_like(base_log_probabilities)
    likelihood_by_row[feature_rows] = likelihood[
        :,
        [
            ll_model["classes"].index(label)
            for label in labels
        ],
    ]

    glm_labels = {"glm-5.3", "glm-5.3-flash"}
    glm_indices = [
        index for index, label in enumerate(labels)
        if label in glm_labels
    ]
    if len(glm_indices) != 2:
        raise ValueError("Expected both GLM labels in final logits")
    mass = np.exp(base_log_probabilities[:, glm_indices]).sum(axis=1)
    order = np.argsort(-mass, kind="stable")
    selected_count = max(1, int(round(args.coverage * len(ids))))
    selected = order[:selected_count]
    selected_with_features = np.array(
        [index for index in selected if index in set(feature_rows)],
        dtype=np.int64,
    )

    temperature = match_temperature(
        base_log_probabilities,
        float(ll_model["entropy_ref"]),
    )
    reference_weight = (
        args.multiclass_weight
        if args.multiclass_weight is not None
        else float(ll_model["W"])
    )
    likelihood_weight = reference_weight / temperature
    first_pass = base_predictions.copy()
    first_pass[selected_with_features] = (
        base_log_probabilities[selected_with_features]
        + likelihood_weight * likelihood_by_row[selected_with_features]
    ).argmax(axis=1)

    gate = np.array(
        [
            index
            for index in selected_with_features
            if labels[first_pass[index]] in glm_labels
        ],
        dtype=np.int64,
    )
    predictions = first_pass.copy()
    gate_ids = [ids[index] for index in gate]
    pair_probability = pairwise_probabilities(
        pair_model,
        flash,
        glm53,
        gate_ids,
    )
    glm_53 = labels.index(pair_model["positive_label"])
    glm_flash = next(
        index for index, label in enumerate(labels)
        if label in glm_labels and label != pair_model["positive_label"]
    )
    predictions[gate] = np.where(
        pair_probability >= 0.5,
        glm_53,
        glm_flash,
    )

    output = pd.DataFrame(
        {
            "id": ids,
            "model": np.asarray(labels)[predictions],
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    report = {
        "rows": len(ids),
        "scored_rows": len(common_ids),
        "selected_rows": len(selected_with_features),
        "gated_rows": len(gate),
        "temperature": temperature,
        "reference_multiclass_weight": reference_weight,
        "model_W": float(ll_model["W"]),
        "likelihood_weight": likelihood_weight,
        "changed_rows": int((predictions != base_predictions).sum()),
    }
    if args.labels_csv:
        truth_frame = pd.read_csv(args.labels_csv, dtype={"id": str})
        truth = truth_frame.set_index("id")["model"]
        actual_labels = truth.loc[ids].astype(str).tolist()
        label_to_index = {label: index for index, label in enumerate(labels)}
        actual = np.array(
            [label_to_index[value] for value in actual_labels],
            dtype=np.int64,
        )
        report["base_balanced_accuracy"] = float(
            balanced_accuracy_score(actual, base_predictions)
        )
        report["balanced_accuracy"] = float(
            balanced_accuracy_score(actual, predictions)
        )
        report["delta"] = (
            report["balanced_accuracy"]
            - report["base_balanced_accuracy"]
        )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
