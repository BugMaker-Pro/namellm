from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get("OWNED_NOTEBOOK_DIR", ROOT / "kaggle_owned_v1")
)


CODE = r'''
import glob
import os
import subprocess
import sys


def config_dirs():
    return [os.path.dirname(path) for path in glob.glob('/kaggle/input/**/config.json', recursive=True)]


artifact_hits = glob.glob('/kaggle/input/**/run_owned_kaggle_inference.py', recursive=True)
assert artifact_hits, 'Owned artifact dataset is not mounted'
artifact_script = artifact_hits[0]

embedding_hits = [
    path for path in config_dirs()
    if 'qwen-3-embedding' in path.lower() and '/4b/' in path.lower().replace('\\', '/')
]
assert embedding_hits, 'Qwen3-Embedding-4B model is not mounted'
embedding_model = embedding_hits[0]

lm_hits = [
    path for path in config_dirs()
    if 'qwen-3' in path.lower() and '4b-base' in path.lower()
]
assert lm_hits, 'Qwen3-4B-Base model is not mounted'
lm_model = lm_hits[0]

print('artifact_script =', artifact_script)
print('embedding_model =', embedding_model)
print('lm_model =', lm_model)

subprocess.run(
    [
        sys.executable,
        artifact_script,
        '--embedding-model',
        embedding_model,
        '--lm-model',
        lm_model,
        '--output',
        '/kaggle/working/submission.parquet',
    ],
    check=True,
)
'''


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Name That LLM Owned V1\n",
                    "\n",
                    "Loads pre-trained downstream artifacts and performs inference only.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": CODE.strip().splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUTPUT_DIR / "notebook.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "id": "yyyyhhhhwww/name-that-llm-owned-v1",
        "title": "Name That LLM Owned V1",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "yyyyhhhhwww/namellm-owned-artifacts",
        ],
        "kernel_sources": [],
        "competition_sources": ["name-that-llm"],
        "model_sources": [
            "qwen-lm/qwen-3-embedding/Transformers/4b/1",
            "qwen-lm/qwen-3/Transformers/4b-base/1",
        ],
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
