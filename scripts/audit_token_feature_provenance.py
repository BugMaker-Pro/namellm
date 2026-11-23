from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def hidden_for_rows(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    max_length: int,
) -> list[np.ndarray]:
    outputs = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {
                "input_ids": torch.tensor(
                    [ids],
                    dtype=torch.long,
                    device="cuda",
                ),
                "attention_mask": torch.ones(
                    (1, len(ids)),
                    dtype=torch.long,
                    device="cuda",
                ),
            }
            value = model(
                **inputs,
                output_hidden_states=True,
            ).hidden_states[layer]
            outputs.append(
                value.float().squeeze(0).cpu().numpy()
            )
    return outputs


def compare(
    stored: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    if stored.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: {stored.shape} versus {candidate.shape}"
        )
    difference = (
        stored.astype(np.float32)
        - candidate.astype(np.float32)
    )
    return {
        "mean_abs": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Determine whether stored Qwen hidden sequences were produced "
            "by the base model or the fold-0 QLoRA adapter."
        )
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3-4B-Base",
    )
    parser.add_argument(
        "--adapter",
        default="artifacts/qlora_fold0",
    )
    parser.add_argument(
        "--sequence-dir",
        default="features/qwen3_seq_fold0",
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    frame = pd.read_csv(TRAIN_CSV)
    metadata = np.load(
        ROOT / args.sequence_dir / "metadata.npz",
        allow_pickle=True,
    )
    sequences = np.load(
        ROOT / args.sequence_dir / "sequences.npy",
        mmap_mode="r",
    )
    source_rows = metadata["row_indices"][: args.rows].astype(np.int64)
    texts = frame["text"].iloc[source_rows].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    ).eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    report: dict[str, object] = {
        "rows": source_rows.tolist(),
        "adapter": args.adapter,
        "sequence_dir": args.sequence_dir,
        "layers": {},
    }
    for layer in (8, 36):
        base_values = hidden_for_rows(
            model,
            tokenizer,
            texts,
            layer,
            args.max_length,
        )
        base_comparisons = []
        for position, candidate in enumerate(base_values):
            stored = sequences[
                metadata["offsets"][position]:
                metadata["offsets"][position + 1]
            ]
            base_comparisons.append(compare(stored, candidate))

        adapted = PeftModel.from_pretrained(
            model,
            ROOT / args.adapter,
        ).eval()
        adapter_values = hidden_for_rows(
            adapted,
            tokenizer,
            texts,
            layer,
            args.max_length,
        )
        adapter_comparisons = []
        for position, candidate in enumerate(adapter_values):
            stored = sequences[
                metadata["offsets"][position]:
                metadata["offsets"][position + 1]
            ]
            adapter_comparisons.append(compare(stored, candidate))
        report["layers"][str(layer)] = {
            "base": base_comparisons,
            "adapter": adapter_comparisons,
        }
        model = adapted.unload()

    print(report)
    output = ROOT / "validation" / "token_feature_provenance.json"
    import json

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
