from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get("KAGGLE_NOTEBOOK_DIR", ROOT / "kaggle_safe_v1")
)
USERNAME = "yyyyhhhhwww"
KERNEL_SLUG = "name-that-llm-safe-v1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    code = (ROOT / "scripts" / "kaggle_safe_solution.py").read_text(
        encoding="utf-8"
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Name That LLM Safe V1\n",
                    "\n",
                    "Char/word TF-IDF baseline with prompt-local soft balance.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
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
        "id": f"{USERNAME}/{KERNEL_SLUG}",
        "title": "Name That LLM Safe V1",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
