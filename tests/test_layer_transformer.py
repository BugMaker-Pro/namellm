from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_layer_sequence import LayerTransformerModel  # noqa: E402


class LayerTransformerTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = LayerTransformerModel(
            input_size=16,
            extra_size=8,
            hidden_size=32,
            layer_count=7,
            transformer_layers=2,
            heads=4,
            causal=False,
            layer_delta=True,
        )
        sequence = torch.randn(3, 7, 16)
        extra = torch.randn(3, 8)
        logits = model(sequence, extra)
        self.assertEqual(tuple(logits.shape), (3, 18))

    def test_causal_forward(self) -> None:
        model = LayerTransformerModel(
            input_size=8,
            extra_size=0,
            hidden_size=16,
            layer_count=4,
            transformer_layers=1,
            heads=4,
            causal=True,
            layer_delta=False,
        )
        logits = model(torch.randn(2, 4, 8), torch.zeros(2, 0))
        self.assertEqual(tuple(logits.shape), (2, 18))


if __name__ == "__main__":
    unittest.main()
