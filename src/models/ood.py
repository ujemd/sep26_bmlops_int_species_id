"""Out-of-scope detection: does this photo look like anything seen in training?

The classifier always answers with one of the known species, even for a cat: softmax spreads
100% of the probability over the classes it knows. Conformal prediction does not catch this
either, because its guarantee assumes the photo comes from the same distribution as the training
data. So we measure how far a photo's embedding is from its k nearest training embeddings
(mean cosine distance; Sun et al., 2022, "Out-of-distribution detection with deep nearest
neighbors"), and compare it with a threshold calibrated on held-out in-scope photos. With
quantile 0.99, about 1% of normal photos are flagged; photos unlike any training photo are flagged.

It cannot catch look-alikes (a viceroy butterfly sits right next to the monarchs): only adding the
species to the perimeter solves that.

The memory bank (normalised training embeddings, float16) and the threshold are stored as
attributes of the sklearn pipeline, so they travel with the model in MLflow.
"""
import numpy as np


def _normalize(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn_distance(bank: np.ndarray, X, k: int) -> np.ndarray:
    """Mean cosine distance from each row of X to its k nearest neighbours in the bank."""
    sims = _normalize(X) @ bank.astype(np.float32).T
    k = min(k, bank.shape[0])
    nearest = -np.partition(-sims, k - 1, axis=1)[:, :k]  # the k largest similarities
    return 1.0 - nearest.mean(axis=1)


def fit(X_train, X_calib, k: int, quantile: float) -> tuple[np.ndarray, float]:
    """Memory bank from the training embeddings, threshold from the calibration embeddings."""
    bank = _normalize(X_train).astype(np.float16)
    threshold = float(np.quantile(knn_distance(bank, X_calib, k), quantile))
    return bank, threshold


def attach(head, bank: np.ndarray, threshold: float, k: int):
    head.ood_bank_, head.ood_threshold_, head.ood_k_ = bank, threshold, k


def check(head, X) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """(distances, out_of_scope flags), or (None, None) for models trained without a detector."""
    if not hasattr(head, "ood_bank_"):
        return None, None
    d = knn_distance(head.ood_bank_, X, head.ood_k_)
    return d, d > head.ood_threshold_
