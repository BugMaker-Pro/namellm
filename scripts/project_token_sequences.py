from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--fit-tokens", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = np.load(input_dir / "sequences.npy", mmap_mode="r")
    metadata = np.load(input_dir / "metadata.npz", allow_pickle=True)
    offsets = metadata["offsets"]
    lengths = metadata["lengths"]
    train_mask = metadata["train_mask"]
    train_rows = np.where(train_mask)[0]

    token_rows = np.concatenate(
        [
            np.arange(offsets[row], offsets[row + 1], dtype=np.int64)
            for row in train_rows
        ]
    )
    rng = np.random.default_rng(args.seed)
    if len(token_rows) > args.fit_tokens:
        token_rows = np.sort(
            rng.choice(token_rows, size=args.fit_tokens, replace=False)
        )

    pca = IncrementalPCA(
        n_components=args.components,
        batch_size=4096,
    )
    batch_size = 4096
    for start in range(0, len(token_rows), batch_size):
        selected = token_rows[start : start + batch_size]
        pca.partial_fit(np.asarray(storage[selected], dtype=np.float32))
        print(
            f"fit {min(start + batch_size, len(token_rows))}/{len(token_rows)}",
            flush=True,
        )

    projected = np.lib.format.open_memmap(
        output_dir / "sequences.npy",
        mode="w+",
        dtype=np.float16,
        shape=(storage.shape[0], args.components),
    )
    transform_batch = 32768
    for start in range(0, storage.shape[0], transform_batch):
        end = min(start + transform_batch, storage.shape[0])
        values = np.asarray(storage[start:end], dtype=np.float32)
        projected[start:end] = pca.transform(values).astype(np.float16)
        print(f"transform {end}/{storage.shape[0]}", flush=True)
    projected.flush()
    del projected

    shutil.copyfile(input_dir / "metadata.npz", output_dir / "metadata.npz")
    np.savez(
        output_dir / "pca.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    print(f"saved projected sequences to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
