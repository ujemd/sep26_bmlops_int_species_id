"""Split conformal prediction (LAC score), Angelopoulos & Bates 2021, section 1.

Calibrate once on held-out data: qhat is the ceil((n+1)(1-alpha))/n quantile of 1 - p(true class).
At inference, the prediction set {k : p_k >= 1 - qhat} contains the true species with
probability >= 1 - alpha. A set that is not exactly one species means the model is unsure,
so the observation is sent to a human.
"""
import math

import numpy as np


def calibrate(probs: np.ndarray, y_idx: np.ndarray, alpha: float) -> float:
    n = len(y_idx)
    scores = 1.0 - probs[np.arange(n), y_idx]
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean mask (n, n_classes)."""
    return probs >= 1.0 - qhat


def needs_review(probs_row: np.ndarray, qhat: float, min_confidence: float) -> bool:
    return int((probs_row >= 1.0 - qhat).sum()) != 1 or float(probs_row.max()) < min_confidence


def coverage(sets: np.ndarray, y_idx: np.ndarray) -> float:
    return float(sets[np.arange(len(y_idx)), y_idx].mean())
