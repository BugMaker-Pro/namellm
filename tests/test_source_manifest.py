from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceManifestTests(unittest.TestCase):
    def test_checked_in_manifest_is_valid(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_source_manifest.py"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=result.stdout + result.stderr,
        )
        self.assertIn("manifest valid", result.stdout)

    def test_manifest_lists_fallback(self) -> None:
        payload = json.loads(
            (ROOT / "configs" / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        ids = {source["id"] for source in payload["sources"]}
        self.assertIn("public_0696", ids)
        self.assertIn("styledistance", ids)


if __name__ == "__main__":
    unittest.main()
