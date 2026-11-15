from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def layer_blocks(data: np.lib.npyio.NpzFile) -> dict[int, np.ndarray]:
    layers = [int(value) for value in data["layers"]]
    features = data["X"]
    block_size = features.shape[1] // len(layers)
    return {
        layer: features[:, index * block_size : (index + 1) * block_size]
        for index, layer in enumerate(layers)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--layers", default="16,24,32,36")
    parser.add_argument("--mode", choices=["concat", "mean_stats"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",")]
    first = np.load(args.first, allow_pickle=True)
    second = np.load(args.second, allow_pickle=True)
    blocks = {**layer_blocks(first), **layer_blocks(second)}
    missing = [layer for layer in layers if layer not in blocks]
    if missing:
        raise ValueError(f"Missing requested layers: {missing}")

    if args.mode == "concat":
        features = np.concatenate([blocks[layer] for layer in layers], axis=1)
    else:
        stacked = np.stack([blocks[layer] for layer in layers], axis=1)
        hidden_size = stacked.shape[-1] // 4
        reshaped = stacked.reshape(
            stacked.shape[0],
            len(layers),
            4,
            hidden_size,
        )
        features = reshaped.mean(axis=1).reshape(stacked.shape[0], -1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=features.astype(np.float32),
        y=first["y"],
        ids=first["ids"],
        prompt_ids=first["prompt_ids"],
        row_indices=first["row_indices"],
        layers=np.array(layers),
    )
    print(
        f"saved {output} mode={args.mode} "
        f"layers={layers} shape={features.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
