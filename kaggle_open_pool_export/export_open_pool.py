from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


LABELS = {
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
}


def find_pool() -> Path:
    candidates = glob.glob(
        "/kaggle/input/**/ntl_open_pool.parquet",
        recursive=True,
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one pool parquet, found {candidates}")
    return Path(candidates[0])


def main() -> None:
    source = find_pool()
    pool = pd.read_parquet(source)
    initial = len(pool)
    pool = pool[pool["model"].isin(LABELS)].copy()
    counts_before_filters = pool["model"].value_counts().sort_index().to_dict()
    filters = {}
    for column in (
        "flag_train_prompt_regen",
        "flag_near_train_prompt",
        "flag_gemini37_routing_risk",
        "flag_kb_proxy_after_ruling",
    ):
        if column in pool.columns:
            mask = pool[column].fillna(False).astype(bool)
            filters[column] = int(mask.sum())
            pool = pool[~mask]
        else:
            filters[column] = None
    if "gen_served_model" in pool.columns:
        served = pool["gen_served_model"].fillna("").astype(str)
        mismatch = served.ne("") & served.ne(pool["model"].astype(str))
        filters["served_mismatch"] = int(mismatch.sum())
        pool = pool[~mismatch]
    else:
        filters["served_mismatch"] = None
    pool["text"] = pool["text"].fillna("").astype(str)
    pool = pool[pool["text"].str.strip().ne("")]
    if "n_words" in pool.columns:
        word_mask = ~pool["n_words"].between(36, 275)
        filters["word_length"] = int(word_mask.sum())
        pool = pool[~word_mask]
    else:
        filters["word_length"] = None
    keep = [
        column
        for column in (
            "id",
            "text",
            "model",
            "prompt_group",
            "source_dataset",
            "n_words",
            "flag_train_prompt_regen",
            "flag_near_train_prompt",
            "flag_gemini37_routing_risk",
            "flag_kb_proxy_after_ruling",
            "gen_served_model",
        )
        if column in pool.columns
    ]
    output = Path("/kaggle/working/ntl_open_pool_filtered.parquet")
    pool[keep].to_parquet(output, index=False)
    summary = {
        "source": str(source),
        "initial_rows": initial,
        "label_rows": sum(counts_before_filters.values()),
        "rows": len(pool),
        "filters": filters,
        "counts": pool["model"].value_counts().sort_index().to_dict(),
        "columns": keep,
    }
    Path("/kaggle/working/summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
