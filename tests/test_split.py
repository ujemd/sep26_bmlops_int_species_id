import pandas as pd

from src.models.train import split

P = {"test_size": 0.15, "calib_size": 0.15}
SPECIES = ("Apis mellifera", "Danaus plexippus", "Turdus migratorius")


def photos(n_per_species: int) -> pd.DataFrame:
    """n photos per species, observation ids increasing in ingestion order."""
    return pd.DataFrame([{"taxon_name": s, "observation_id": 1000 * k + i}
                         for k, s in enumerate(SPECIES) for i in range(n_per_species)])


def test_first_run_is_exactly_stratified():
    meta = photos(180)
    part = split(meta, P)
    for s in SPECIES:
        counts = pd.Series(part[(meta.taxon_name == s).to_numpy()]).value_counts()
        assert (counts["train"], counts["calib"], counts["test"]) == (126, 27, 27)


def test_parts_do_not_change_when_newer_observations_arrive():
    first, grown = photos(150), photos(180)
    before = split(first, P)
    after = split(grown, P)[grown.observation_id.isin(first.observation_id).to_numpy()]
    assert (before == after).all()


def test_row_order_does_not_matter():
    meta = photos(60)
    shuffled = meta.sample(frac=1, random_state=0)
    by_id = dict(zip(meta.observation_id, split(meta, P)))
    assert all(by_id[o] == p for o, p in zip(shuffled.observation_id, split(shuffled, P)))
