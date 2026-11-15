from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import TEST_CSV, TRAIN_CSV  # noqa: E402
from namellm.lm_stats import LMStatConfig, score_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)[["id", "text"]].assign(split="train")
    test = pd.read_csv(TEST_CSV)[["id", "text"]].assign(split="test")
    frame = pd.concat([train, test], ignore_index=True)
    if args.limit:
        frame = frame.head(args.limit)

    features = score_texts(
        frame["id"].tolist(),
        frame["text"].tolist(),
        LMStatConfig(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
        ),
        device=args.device,
    )
    output = features.merge(frame[["id", "split"]], on="id")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"saved {args.output} ({len(output)} rows)")


if __name__ == "__main__":
    main()
