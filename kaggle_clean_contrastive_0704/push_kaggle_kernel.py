from __future__ import annotations

import argparse
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials


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
    parser.add_argument("folder")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--timeout",
        default="32400",
        help="Kaggle kernel timeout in seconds.",
    )
    args = parser.parse_args()
    folder = Path(args.folder)
    if not (folder / "kernel-metadata.json").is_file():
        raise FileNotFoundError(folder / "kernel-metadata.json")
    refresh_access_token()
    api = KaggleApi()
    api.authenticate()
    response = api.kernels_push(
        str(folder),
        timeout=args.timeout,
        no_run=args.no_run,
    )
    print(response)


if __name__ == "__main__":
    main()
