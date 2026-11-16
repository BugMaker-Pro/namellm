from __future__ import annotations

import numpy as np
import pandas as pd

from namellm.constants import LABELS


def predict_truth(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    seed: int,
    fold: int,
) -> np.ndarray:
    del train_frame, seed, fold
    probabilities = np.zeros((len(valid_frame), len(LABELS)))
    for row, label in enumerate(valid_frame["model"]):
        probabilities[row, LABELS.index(label)] = 1.0
    return probabilities
