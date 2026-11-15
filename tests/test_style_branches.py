from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_style_branches import build_design  # noqa: E402


class StyleBranchTests(unittest.TestCase):
    def test_absolute_design_has_no_relative_features(self) -> None:
        embedding = np.arange(12, dtype=np.float32).reshape(4, 3)
        style = np.ones((4, 2), dtype=np.float32)
        clusters = np.array([0, 0, 1, 1])
        matrix, extra = build_design(
            embedding,
            style,
            clusters,
            use_style=True,
            use_group_relative=False,
        )
        np.testing.assert_array_equal(matrix, embedding)
        np.testing.assert_array_equal(extra, style)

    def test_group_relative_is_opt_in(self) -> None:
        embedding = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0], [9.0, 11.0]],
            dtype=np.float32,
        )
        style = np.zeros((4, 1), dtype=np.float32)
        clusters = np.array([0, 0, 1, 1])
        _, extra = build_design(
            embedding,
            style,
            clusters,
            use_style=True,
            use_group_relative=True,
        )
        self.assertEqual(extra.shape, (4, 4))
        np.testing.assert_allclose(
            extra[:, 1:3].reshape(2, 2, 2).mean(axis=1),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
