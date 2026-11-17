import numpy as np

from src.models import conformal


def test_coverage_guarantee_holds_on_synthetic_data():
    rng = np.random.default_rng(0)
    n, k, alpha = 4000, 10, 0.1
    logits = rng.normal(size=(n, k))
    y = rng.integers(0, k, n)
    logits[np.arange(n), y] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    qhat = conformal.calibrate(probs[:2000], y[:2000], alpha)
    cov = conformal.coverage(conformal.prediction_sets(probs[2000:], qhat), y[2000:])
    assert cov >= 1 - alpha - 0.02


def test_review_when_set_is_not_a_singleton():
    assert conformal.needs_review(np.array([0.5, 0.45, 0.05]), qhat=0.6, min_confidence=0.3)
    assert not conformal.needs_review(np.array([0.9, 0.05, 0.05]), qhat=0.3, min_confidence=0.5)


def test_review_when_confidence_too_low():
    assert conformal.needs_review(np.array([0.45, 0.3, 0.25]), qhat=0.6, min_confidence=0.5)
