from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-4B-Base")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    train_idx, valid_idx = folds[args.fold]
    all_idx = np.concatenate([train_idx, valid_idx])

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        quantization_config=quantization_config,
        device_map={"": device.index or 0},
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    token_ids = [
        tokenizer.encode(
            frame["text"].iloc[row],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
        )
        for row in all_idx
    ]
    lengths = np.array([len(ids) for ids in token_ids], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total_tokens = int(offsets[-1])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = int(model.config.hidden_size)
    storage = np.lib.format.open_memmap(
        output_dir / "sequences.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_tokens, hidden_size),
    )

    started = time.time()
    for position, ids in enumerate(token_ids, start=1):
        inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)
        hidden = output.hidden_states[args.layer].float().squeeze(0)
        storage[offsets[position - 1] : offsets[position]] = hidden.half().cpu().numpy()
        if position % 50 == 0 or position == len(token_ids):
            elapsed = time.time() - started
            print(
                f"{position}/{len(token_ids)} elapsed={elapsed:.1f}s "
                f"remaining={elapsed / position * (len(token_ids) - position):.1f}s",
                flush=True,
            )
    storage.flush()
    del storage

    np.savez(
        output_dir / "metadata.npz",
        y=labels[all_idx],
        prompt_ids=frame["prompt_id"].to_numpy()[all_idx],
        row_indices=all_idx,
        train_mask=np.isin(all_idx, train_idx),
        valid_mask=np.isin(all_idx, valid_idx),
        offsets=offsets,
        lengths=lengths,
        layer=np.array([args.layer]),
        hidden_size=np.array([hidden_size]),
    )
    print(f"saved {output_dir} total_tokens={total_tokens}", flush=True)


if __name__ == "__main__":
    main()
