import pandas as pd

from src.models.train import split

P = {"seed": 42, "test_size": 0.15, "calib_size": 0.15}


def test_split_is_stable_when_the_dataset_grows():
    small = split(pd.DataFrame({"photo_id": range(1000)}), P)
    big = split(pd.DataFrame({"photo_id": range(2000)}), P)
    assert (small == big[:1000]).all()


def test_split_proportions():
    part = split(pd.DataFrame({"photo_id": range(5000)}), P)
    assert abs((part == "test").mean() - 0.15) < 0.02
    assert abs((part == "calib").mean() - 0.15) < 0.02
