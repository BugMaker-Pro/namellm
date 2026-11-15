from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_style_branches import build_design  # noqa: E402
from train_partial_group_gaussian import (  # noqa: E402
    DiagonalGaussian,
    group_relative_rows,
)


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

    def test_partial_group_gaussian(self) -> None:
        rng = np.random.default_rng(4)
        centers = np.linspace(-2.0, 2.0, 18)[:, None]
        features = np.vstack(
            [
                centers + rng.normal(0.0, 0.2, size=(18, 3)),
                centers + rng.normal(0.0, 0.2, size=(18, 3)),
            ]
        ).astype(np.float32)
        labels = np.repeat(np.arange(18), 2)
        model = DiagonalGaussian(0.2, 1e-4).fit(features, labels)
        scores = model.scores(features[:4])
        self.assertEqual(scores.shape, (4, 18))
        relative, eligible = group_relative_rows(
            features,
            np.array([0, 0, 1, 1] + [2] * 32),
            minimum=2,
            maximum=5,
        )
        self.assertTrue(eligible[:4].all())
        np.testing.assert_allclose(
            relative[:2].mean(axis=0),
            0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
