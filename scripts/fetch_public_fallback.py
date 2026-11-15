from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials


DEFAULT_SOURCE = (
    "alexchilton/name-that-llm-0-696-all-public-what-failed"
)
DEFAULT_REF = "yyyyhhhhwww/name-that-llm-0696-fallback"


def refresh_access_token() -> None:
    creds_path = Path.home() / ".kaggle" / "credentials.json"
    creds = KaggleCredentials.load(
        client=KaggleClient(),
        file_path=str(creds_path),
    )
    if creds is None:
        raise RuntimeError(f"Kaggle OAuth credentials not found: {creds_path}")
    response = creds.generate_access_token()
    os.environ["KAGGLE_API_TOKEN"] = response.token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--title",
        default="Name That LLM 0696 Fallback",
    )
    parser.add_argument(
        "--output-dir",
        default="kaggle_public_fallback",
    )
    args = parser.parse_args()

    owner, slug = args.source.split("/", 1)
    refresh_access_token()
    api = KaggleApi()
    api.authenticate()
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    notebook = json.loads(response.blob.source)
    if "cells" not in notebook:
        raise ValueError("Source kernel is not a notebook")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )
    metadata = response.metadata
    kernel_metadata = {
        "id": args.ref,
        "title": args.title[:50],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": list(metadata.dataset_data_sources or []),
        "kernel_sources": list(metadata.kernel_data_sources or []),
        "competition_sources": list(
            metadata.competition_data_sources or []
        ),
        "model_sources": list(metadata.model_data_sources or []),
    }
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(kernel_metadata, indent=2),
        encoding="utf-8",
    )
    provenance = {
        "source_ref": args.source,
        "source_version": int(metadata.current_version_number),
        "source_title": metadata.title,
        "source_updated_at": str(metadata.last_run_time),
        "target_ref": args.ref,
        "dataset_sources": kernel_metadata["dataset_sources"],
        "kernel_sources": kernel_metadata["kernel_sources"],
        "model_sources": kernel_metadata["model_sources"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(
        f"fetched {args.source} version {metadata.current_version_number} "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
