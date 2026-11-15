from __future__ import annotations

import argparse
import json
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
    os.environ["KAGGLE_API_TOKEN"] = creds.generate_access_token().token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument("--file-name", default="submission.parquet")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    refresh_access_token()
    api = KaggleApi()
    api.authenticate()
    response = api.competition_submit_code(
        file_name=args.file_name,
        message=args.message,
        competition=args.competition,
        kernel=args.kernel,
        kernel_version=args.kernel_version,
    )
    payload = (
        response.to_dict()
        if hasattr(response, "to_dict")
        else vars(response)
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
