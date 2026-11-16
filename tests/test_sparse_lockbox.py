from __future__ import annotations

import importlib.util
import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sparse2_lockbox.py"
SPEC = importlib.util.spec_from_file_location("sparse2_lockbox", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Sparse2LockboxTest(unittest.TestCase):
    def test_protocol_and_calibration_are_well_formed(self) -> None:
        protocol = json.loads(
            (ROOT / "validation" / "sparse2_protocol.json").read_text()
        )
        self.assertEqual(protocol["responses_per_prompt"], 2)
        self.assertEqual(protocol["outer_folds"], 5)
        with (ROOT / "validation" / "lb_calibration.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(
            set(rows[0]),
            {
                "variant",
                "parent",
                "submission_ref",
                "public_lb",
                "sparse2_mean",
                "sparse2_std",
                "sparse2_p05",
                "seed_win_rate",
                "paired_delta_vs_parent",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "status",
                "notes",
            },
        )

    def test_sample_two_rows_per_prompt(self) -> None:
        frame = pd.DataFrame(
            {
                "prompt_id": np.repeat(
                    [f"p{index}" for index in range(4)],
                    5,
                )
            }
        )
        selected = MODULE.sample_two_rows(
            frame,
            np.random.default_rng(17),
            2,
        )
        counts = (
            frame.iloc[selected]["prompt_id"].value_counts().to_dict()
        )
        self.assertEqual(counts, {f"p{index}": 2 for index in range(4)})

    def test_fold_assignment_keeps_prompts_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "prompt_id": np.repeat(
                    [f"p{index}" for index in range(12)],
                    18,
                )
            }
        )
        labels = np.tile(np.arange(18), 12)
        assignment = MODULE.make_fold_assignment(frame, labels, 4)
        for fold in range(4):
            train_prompts = set(
                frame.loc[assignment != fold, "prompt_id"]
            )
            valid_prompts = set(
                frame.loc[assignment == fold, "prompt_id"]
            )
            self.assertFalse(train_prompts & valid_prompts)

    def test_cli_supports_callable_candidate(self) -> None:
        labels = list(MODULE.LABELS)
        frame = pd.DataFrame(
            {
                "id": [f"id{index}" for index in range(180)],
                "prompt_id": np.repeat(
                    [f"p{index}" for index in range(10)],
                    18,
                ),
                "text": ["text"] * 180,
                "model": np.tile(labels, 10),
            }
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            input_csv = directory / "train.csv"
            frame.to_csv(input_csv, index=False)
            baseline_path = directory / "baseline.npy"
            np.save(
                baseline_path,
                np.full((len(frame), len(labels)), 1 / len(labels)),
            )
            protocol = {
                "outer_folds": 2,
                "seeds": 2,
                "seed_start": 17,
                "responses_per_prompt": 2,
                "bootstrap_replicates": 3,
            }
            protocol_path = directory / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            output = directory / "result.json"
            per_seed = directory / "per_seed.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-csv",
                    str(input_csv),
                    "--protocol",
                    str(protocol_path),
                    "--baseline-static",
                    f"uniform={baseline_path}",
                    "--candidate-callable",
                    "perfect=tests.sparse2_callable_fixture:predict_truth",
                    "--baseline",
                    "uniform",
                    "--output",
                    str(output),
                    "--per-seed-output",
                    str(per_seed),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn("saved", completed.stdout)
            result = json.loads(output.read_text())
            self.assertAlmostEqual(
                result["candidates"]["perfect"]["score"]["mean"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
