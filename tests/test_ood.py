import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.models import ood

rng = np.random.default_rng(0)
CENTRES = rng.normal(size=(3, 64)) * 5  # three "species" in a 64-dim embedding space


def photos(n_per_species: int) -> np.ndarray:
    return np.vstack([c + rng.normal(size=(n_per_species, 64)) for c in CENTRES])


def fitted_head():
    X_tr, X_cal = photos(100), photos(100)
    head = make_pipeline(LogisticRegression(max_iter=500)).fit(X_tr, np.repeat([0, 1, 2], 100))
    ood.attach(head, *ood.fit(X_tr, X_cal, k=5, quantile=0.99), k=5)
    return head


def test_about_one_percent_of_normal_photos_are_flagged():
    _, flags = ood.check(fitted_head(), photos(1000))
    assert flags.mean() < 0.03


def test_a_photo_unlike_any_training_photo_is_flagged():
    stranger = rng.normal(size=(1, 64)) * 5  # a fourth, unknown "species"
    _, flags = ood.check(fitted_head(), stranger)
    assert flags[0]


def test_the_detector_travels_with_the_pickled_model():
    restored = pickle.loads(pickle.dumps(fitted_head()))
    assert restored.ood_bank_.dtype == np.float16 and restored.ood_threshold_ > 0


def test_models_trained_without_a_detector_still_work():
    head = make_pipeline(LogisticRegression()).fit(photos(10), np.repeat([0, 1, 2], 10))
    assert ood.check(head, photos(1)) == (None, None)
