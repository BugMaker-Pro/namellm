from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_softmax, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS


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
    for _ in range(100):
        middle = np.sqrt(low * high)
        if (
            entropy(log_softmax(log_probabilities * middle, axis=1))
            > target_entropy
        ):
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--multiclass-oof", required=True)
    parser.add_argument("--pairwise-oof", required=True)
    parser.add_argument("--ll-model", required=True)
    parser.add_argument("--coverage", type=float, default=0.30)
    parser.add_argument("--reference-weight", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    base = np.load(args.base).astype(np.float64)
    multiclass = np.load(args.multiclass_oof).astype(np.float64)
    pairwise = np.load(args.pairwise_oof).astype(np.float64)
    model = json.loads(
        Path(args.ll_model).read_text(encoding="utf-8")
    )
    if base.shape != (len(multiclass), len(LABELS)):
        raise ValueError("Base probabilities have the wrong shape")
    if pairwise.shape != (len(base),):
        raise ValueError("Pairwise probabilities have the wrong shape")
    base /= base.sum(axis=1, keepdims=True)
    multiclass /= multiclass.sum(axis=1, keepdims=True)
    base_log = np.log(np.clip(base, 1e-12, None))
    multiclass_log = np.log(np.clip(multiclass, 1e-12, None))

    glm_indices = [
        LABELS.index("glm-5.3"),
        LABELS.index("glm-5.3-flash"),
    ]
    mass = base[:, glm_indices].sum(axis=1)
    selected_count = max(1, int(round(args.coverage * len(base))))
    selected = np.argsort(-mass, kind="stable")[:selected_count]
    temperature = match_temperature(
        base_log,
        float(model["entropy_ref"]),
    )
    reference_weight = (
        args.reference_weight
        if args.reference_weight is not None
        else float(model["W"])
    )
    likelihood_weight = reference_weight / temperature

    combined = base_log.copy()
    combined[selected] += likelihood_weight * multiclass_log[selected]
    predictions = combined.argmax(axis=1)
    gate = selected[
        np.isin(predictions[selected], glm_indices)
    ]
    pairwise = np.asarray(pairwise, dtype=np.float64)
    if np.isnan(pairwise[gate]).any():
        raise ValueError("Pairwise probabilities missing for gated rows")
    predictions[gate] = np.where(
        pairwise[gate] >= 0.5,
        glm_indices[0],
        glm_indices[1],
    )
    decision_probabilities = softmax(combined, axis=1)
    decision_probabilities[np.arange(len(base))] = 0.0
    decision_probabilities[
        np.arange(len(base)),
        predictions,
    ] = 1.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, decision_probabilities.astype(np.float32))
    report = {
        "rows": len(base),
        "coverage": args.coverage,
        "selected_rows": int(len(selected)),
        "gated_rows": int(len(gate)),
        "temperature": temperature,
        "reference_weight": reference_weight,
        "likelihood_weight": likelihood_weight,
        "changed_rows": int(
            (predictions != base.argmax(axis=1)).sum()
        ),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
