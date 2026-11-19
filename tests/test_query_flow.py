from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_sequence_downstream import QueryFlowModel  # noqa: E402


class QueryFlowTests(unittest.TestCase):
    def test_forward_shape_with_causal_queries(self) -> None:
        model = QueryFlowModel(
            input_size=16,
            summary_size=8,
            query_count=6,
            query_layers=2,
        )
        values = torch.randn(2, 9, 16)
        mask = torch.tensor(
            [
                [True] * 9,
                [True] * 5 + [False] * 4,
            ]
        )
        summary = torch.randn(2, 8)
        logits = model(values, mask, summary)
        self.assertEqual(tuple(logits.shape), (2, 18))

    def test_forward_shape_with_symmetric_queries(self) -> None:
        model = QueryFlowModel(
            input_size=8,
            query_count=4,
            query_layers=1,
            query_causal=False,
        )
        values = torch.randn(3, 7, 8)
        mask = torch.ones(3, 7, dtype=torch.bool)
        logits = model(values, mask)
        self.assertEqual(tuple(logits.shape), (3, 18))

    def test_forward_aux_returns_style_prediction(self) -> None:
        model = QueryFlowModel(
            input_size=8,
            query_count=4,
            query_layers=1,
            style_size=36,
        )
        values = torch.randn(2, 6, 8)
        mask = torch.ones(2, 6, dtype=torch.bool)
        logits, style, query_features = model.forward_aux(values, mask)
        self.assertEqual(tuple(logits.shape), (2, 18))
        self.assertEqual(tuple(style.shape), (2, 36))
        self.assertEqual(tuple(query_features.shape), (2, 384))


if __name__ == "__main__":
    unittest.main()
