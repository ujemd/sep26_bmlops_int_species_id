import json
from pathlib import Path

from src.data.ingest import is_open_photo, parse_observation, sized_url

FIX = json.loads((Path(__file__).parent / "fixtures" / "inat_observations.json").read_text())
ALLOWED = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}
HOST = "inaturalist-open-data.s3.amazonaws.com"


def parse(obs):
    return parse_observation(obs, ALLOWED, HOST, "medium")


def test_open_photo_is_kept_with_attribution():
    o, photos = parse(FIX["results"][0])
    assert o.id == 101 and o.latitude == 48.85
    assert photos[0].license_code == "cc-by-nc"
    assert "alice" in photos[0].attribution
    assert photos[0].url.endswith("/9001/medium.jpg")


def test_all_rights_reserved_photo_is_rejected():
    o, photos = parse(FIX["results"][1])
    assert o is None and photos == []


def test_no_derivatives_license_skipped_but_next_open_photo_used():
    _, photos = parse(FIX["results"][2])
    assert [p.id for p in photos] == [9004]
    assert photos[0].license_code == "cc0"


def test_static_domain_is_never_accepted_even_with_open_license():
    assert not is_open_photo({"license_code": "cc0", "url": "https://static.inaturalist.org/photos/1/square.jpg"},
                             ALLOWED, HOST)


def test_sized_url():
    assert sized_url("https://x/photos/1/square.jpeg", "medium") == "https://x/photos/1/medium.jpeg"
