from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_embedding_token_features import (  # noqa: E402
    TextRows,
    make_collator,
)
from namellm.constants import LABELS, TRAIN_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract leak-free Qwen3 base hidden token sequences without a "
            "fold-specific adapter."
        )
    )
    parser.add_argument("--model", default="models/Qwen3-4B-Base")
    parser.add_argument(
        "--output-root",
        default="features/qwen3_base_hidden_layers_8_36",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    frame = pd.read_csv(TRAIN_CSV)
    texts = frame["text"].fillna("").astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = np.asarray(
        [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=args.max_length,
                )
            )
            for text in texts
        ],
        dtype=np.int64,
    )
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total_tokens = int(offsets[-1])

    model = AutoModel.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    hidden_size = int(model.config.hidden_size)

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    layer_paths = {
        layer: output_root / f"layer{layer}" / "sequences.npy"
        for layer in (8, 36)
    }
    storage = {}
    for layer, path in layer_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        storage[layer] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(total_tokens, hidden_size),
        )

    loader = DataLoader(
        TextRows(texts, np.arange(len(texts))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer, args.max_length),
        num_workers=0,
    )
    started = time.time()
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            positions = batch.pop("row_indices").numpy()
            batch = {
                name: value.to("cuda")
                for name, value in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(
                    **batch,
                    output_hidden_states=True,
                    return_dict=True,
                )
            active_lengths = batch["attention_mask"].sum(dim=1)
            for local_row, source_row in enumerate(positions):
                length = int(active_lengths[local_row])
                start = int(offsets[source_row])
                stop = start + length
                for layer in (8, 36):
                    storage[layer][start:stop] = (
                        result.hidden_states[layer][
                            local_row, :length
                        ]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
            completed += len(positions)
            if completed % 50 == 0 or completed == len(texts):
                elapsed = time.time() - started
                print(
                    f"{completed}/{len(texts)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    for values in storage.values():
        values.flush()
    del storage
    labels = frame["model"].map(
        {label: index for index, label in enumerate(LABELS)}
    ).to_numpy(np.int64)
    for layer in (8, 36):
        np.savez(
            layer_paths[layer].parent / "metadata.npz",
            y=labels,
            prompt_ids=frame["prompt_id"].astype(str).to_numpy(),
            ids=frame["id"].astype(str).to_numpy(),
            row_indices=np.arange(len(frame), dtype=np.int64),
            offsets=offsets,
            lengths=lengths,
            layer=np.asarray([layer], dtype=np.int64),
            hidden_size=np.asarray([hidden_size], dtype=np.int64),
            source=np.asarray(["qwen3-4b-base-no-adapter"]),
        )
    print(
        f"saved {output_root} rows={len(texts)} "
        f"tokens={total_tokens} hidden={hidden_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
