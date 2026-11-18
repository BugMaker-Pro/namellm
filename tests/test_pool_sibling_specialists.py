from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pool_sibling_specialists import (  # noqa: E402
    apply_specialist,
    score_subset,
)


class PoolSiblingSpecialistTests(unittest.TestCase):
    def test_apply_preserves_lab_log_mass(self) -> None:
        base = np.arange(36, dtype=np.float64).reshape(2, 18) / 10
        specialist = np.asarray(
            [
                [0.2, -0.1, 0.4, -0.5],
                [-0.3, 0.6, -0.2, -0.1],
            ]
        )
        classes = np.asarray(
            ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5"]
        )
        columns = [0, 1, 2, 3]
        current = np.logaddexp.reduce(base[:, columns], axis=1)
        refined = apply_specialist(base, specialist, classes, 0.25)
        updated = np.logaddexp.reduce(refined[:, columns], axis=1)
        np.testing.assert_allclose(updated, current)
        keep = np.setdiff1d(np.arange(base.shape[1]), columns)
        np.testing.assert_allclose(refined[:, keep], base[:, keep])

    def test_score_subset_is_balanced(self) -> None:
        targets = np.asarray([0, 1, 0, 1])
        probabilities = np.zeros((4, 18))
        probabilities[np.arange(4), [0, 1, 1, 1]] = 1.0
        self.assertAlmostEqual(
            score_subset(targets, probabilities, np.asarray([True, True, False, False])),
            1.0,
        )
        self.assertAlmostEqual(
            score_subset(targets, probabilities, np.ones(4, dtype=bool)),
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
