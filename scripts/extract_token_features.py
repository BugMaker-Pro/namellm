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


def token_summary(
    hidden_states: tuple[torch.Tensor, ...],
    layers: list[int],
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for layer_index in layers:
        hidden = hidden_states[layer_index].float().squeeze(0)
        blocks.extend(
            [
                hidden.mean(dim=0).cpu().numpy(),
                hidden.std(dim=0, unbiased=False).cpu().numpy(),
                hidden[-1].cpu().numpy(),
                (
                    (hidden[1:] - hidden[:-1]).abs().mean(dim=0).cpu().numpy()
                    if hidden.shape[0] > 1
                    else np.zeros(hidden.shape[1], dtype=np.float32)
                ),
            ]
        )
    return np.concatenate(blocks).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-4B-Base")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--layers", default="24,36")
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",")]
    frame = pd.read_csv(TRAIN_CSV)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = np.array([label_to_index[value] for value in frame["model"]])
    folds = list(GroupKFold(5).split(frame, labels, frame["prompt_id"]))
    train_idx, valid_idx = folds[args.fold]
    all_idx = np.concatenate([train_idx, valid_idx])

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
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

    features: np.ndarray | None = None
    started = time.time()
    for position, row in enumerate(all_idx, start=1):
        inputs = tokenizer(
            frame["text"].iloc[row],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)
        vector = token_summary(output.hidden_states, layers)
        if features is None:
            features = np.empty((len(all_idx), vector.shape[0]), dtype=np.float32)
        features[position - 1] = vector
        if position % 50 == 0 or position == len(all_idx):
            elapsed = time.time() - started
            print(
                f"{position}/{len(all_idx)} elapsed={elapsed:.1f}s "
                f"remaining={elapsed / position * (len(all_idx) - position):.1f}s",
                flush=True,
            )

    train_mask = np.isin(all_idx, train_idx)
    valid_mask = np.isin(all_idx, valid_idx)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=features,
        y=labels[all_idx],
        prompt_ids=frame["prompt_id"].to_numpy()[all_idx],
        row_indices=all_idx,
        train_mask=train_mask,
        valid_mask=valid_mask,
        layers=np.array(layers),
    )
    print(f"saved {output_path} shape={features.shape}", flush=True)


if __name__ == "__main__":
    main()

