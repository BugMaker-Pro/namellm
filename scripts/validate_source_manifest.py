from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "kind",
    "url",
    "license",
    "role",
    "final_training",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/source_manifest.json",
    )
    parser.add_argument(
        "--require-final-only",
        action="store_true",
        help="Fail if any final-training source has an unknown license.",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest must contain a non-empty sources list")

    errors = []
    seen_ids = set()
    for index, source in enumerate(sources):
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            errors.append(f"source[{index}] missing {sorted(missing)}")
            continue
        source_id = str(source["id"])
        if source_id in seen_ids:
            errors.append(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)
        if not str(source["url"]).startswith(("http://", "https://")):
            errors.append(f"{source_id} has invalid url")
        if not isinstance(source["final_training"], bool):
            errors.append(f"{source_id} final_training must be boolean")
        if (
            args.require_final_only
            and source["final_training"]
            and source["license"] in {"unknown", "no_repository_license"}
        ):
            errors.append(
                f"{source_id} is final-training enabled with unresolved license"
            )
        local_path = source.get("local_path")
        expected_hash = source.get("sha256")
        if local_path or expected_hash:
            if not local_path or not expected_hash:
                errors.append(
                    f"{source_id} must provide both local_path and sha256"
                )
                continue
            resolved = Path(local_path)
            if not resolved.is_absolute():
                resolved = path.parents[1] / resolved
            if not resolved.is_file():
                errors.append(
                    f"{source_id} local file does not exist: {resolved}"
                )
                continue
            actual_hash = file_sha256(resolved)
            if actual_hash != str(expected_hash).upper():
                errors.append(
                    f"{source_id} sha256 mismatch: "
                    f"{actual_hash} != {expected_hash}"
                )

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"manifest valid: {len(sources)} sources")


if __name__ == "__main__":
    main()
