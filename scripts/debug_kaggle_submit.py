from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)
from kagglesdk.kaggle_client import KaggleClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="name-that-llm")
    parser.add_argument("--file", required=True)
    parser.add_argument("--message", default="debug submission")
    args = parser.parse_args()

    token = os.environ["KAGGLE_API_TOKEN"]
    file_path = Path(args.file)
    api = KaggleApi()
    try:
        with KaggleClient(api_token=token) as client:
            start = ApiStartSubmissionUploadRequest()
            start.competition_name = args.competition
            start.file_name = file_path.name
            start.content_length = file_path.stat().st_size
            start.last_modified_epoch_seconds = int(file_path.stat().st_mtime)
            upload = client.competitions.competition_api_client.start_submission_upload(
                start
            )
            status = api.upload_complete(
                str(file_path),
                upload.create_url,
                quiet=True,
            )
            print(f"upload_status={status}")
            submit = ApiCreateSubmissionRequest()
            submit.competition_name = args.competition
            submit.blob_file_tokens = upload.token
            submit.submission_description = args.message
            response = (
                client.competitions.competition_api_client.create_submission(
                    submit
                )
            )
            print(response)
    except requests.exceptions.HTTPError as error:
        print(f"status={error.response.status_code}")
        print(error.response.text)
        raise


if __name__ == "__main__":
    main()
