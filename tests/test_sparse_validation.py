from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sparse_prompt_validation import (  # noqa: E402
    normalize_rows,
    sample_prompt_rows,
)


class SparseValidationTests(unittest.TestCase):
    def test_sample_prompt_rows_keeps_groups(self) -> None:
        groups = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 2)
        sampled = sample_prompt_rows(
            groups,
            responses_per_prompt=2,
            rng=np.random.default_rng(7),
        )
        counts = {
            group: int(np.sum(groups[sampled] == group))
            for group in np.unique(groups)
        }
        self.assertEqual(counts, {"a": 2, "b": 2, "c": 2})
        self.assertEqual(len(np.unique(sampled)), len(sampled))

    def test_large_sample_caps_at_group_size(self) -> None:
        groups = np.array(["a", "a", "b", "b", "b"])
        sampled = sample_prompt_rows(
            groups,
            responses_per_prompt=18,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(len(sampled), len(groups))

    def test_normalize_rows(self) -> None:
        first = np.zeros(18, dtype=np.float64)
        first[:2] = 2.0
        second = np.zeros(18, dtype=np.float64)
        second[0] = 1.0
        second[1] = 3.0
        values = normalize_rows(
            np.stack([first, second])
        )
        np.testing.assert_allclose(values.sum(axis=1), 1.0)

    def test_normalize_rejects_bad_rows(self) -> None:
        negative = np.ones(18, dtype=np.float64)
        negative[0] = -1.0
        empty = np.zeros(18, dtype=np.float64)
        with self.assertRaises(ValueError):
            normalize_rows(negative[None, :])
        with self.assertRaises(ValueError):
            normalize_rows(empty[None, :])


if __name__ == "__main__":
    unittest.main()
