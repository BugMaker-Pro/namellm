from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text(text: object) -> str:
    return " ".join(str(text or "").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--per-label", type=int, default=1500)
    parser.add_argument("--min-words", type=int, default=36)
    parser.add_argument("--max-words", type=int, default=275)
    args = parser.parse_args()
    if args.per_label <= 0:
        raise ValueError("--per-label must be positive")

    pool = pd.read_parquet(args.pool)
    required = {"id", "text", "model"}
    if not required.issubset(pool.columns):
        raise ValueError(
            f"Pool is missing required columns: {sorted(required - set(pool.columns))}"
        )
    frame = pool[pool["model"].isin(LABELS)].copy()
    initial = len(frame)
    for column in [
        "flag_train_prompt_regen",
        "flag_near_train_prompt",
        "flag_gemini37_routing_risk",
        "flag_kb_proxy_after_ruling",
    ]:
        if column in frame.columns:
            frame = frame[~frame[column].fillna(False).astype(bool)]
    if "gen_served_model" in frame.columns:
        served = frame["gen_served_model"].fillna("").astype(str)
        mismatch = served.ne("") & served.ne(frame["model"].astype(str))
        frame = frame[~mismatch]
    frame["text"] = frame["text"].fillna("").astype(str).map(
        normalized_text
    )
    frame = frame[frame["text"].ne("")]
    word_count = frame["text"].str.split().str.len()
    frame = frame[word_count.between(args.min_words, args.max_words)]
    train = pd.read_csv(args.train_csv)
    train_text = set(
        train["text"].fillna("").astype(str).map(normalized_text)
    )
    frame = frame[~frame["text"].isin(train_text)]
    frame = frame.drop_duplicates(subset=["text"]).copy()
    frame["_hash"] = frame["text"].map(stable_hash)
    frame = (
        frame.sort_values(["model", "_hash"])
        .groupby("model", group_keys=False, sort=False)
        .head(args.per_label)
        .drop(columns="_hash")
        .reset_index(drop=True)
    )
    if "prompt_group" in frame.columns:
        frame["prompt_id"] = frame["prompt_group"].fillna("").astype(str)
    elif "prompt_canon" in frame.columns:
        frame["prompt_id"] = frame["prompt_canon"].fillna("").astype(str)
    else:
        frame["prompt_id"] = frame.index.astype(str)
    frame["sample_weight"] = 0.35
    keep = [
        column
        for column in [
            "id",
            "prompt_id",
            "text",
            "model",
            "source_dataset",
            "source_license",
            "match_weight",
            "sample_weight",
        ]
        if column in frame.columns
    ]
    frame = frame[keep].reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(
        f"pool rows={initial} cleaned_and_sampled={len(frame)} "
        f"counts={frame['model'].value_counts().to_dict()}"
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
