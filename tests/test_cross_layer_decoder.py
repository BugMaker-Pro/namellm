from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_cross_layer_decoder import CrossLayerDecoder  # noqa: E402


class CrossLayerDecoderTests(unittest.TestCase):
    def test_forward_with_positions_and_padding(self) -> None:
        model = CrossLayerDecoder(
            input_size=16,
            style_size=8,
            d_model=32,
            decoder_layers=2,
            heads=4,
        )
        query = torch.randn(2, 9, 16)
        key = torch.randn(2, 9, 16)
        mask = torch.tensor(
            [
                [True] * 9,
                [True] * 5 + [False] * 4,
            ]
        )
        styles = torch.randn(2, 8)
        logits = model(query, key, mask, styles)
        self.assertEqual(tuple(logits.shape), (2, 18))


if __name__ == "__main__":
    unittest.main()
