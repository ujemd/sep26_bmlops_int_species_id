"""Frozen pre-trained feature extractor (timm) and an embedding cache.

Embeddings are computed once per (extractor, photo) and cached in
data/processed/embeddings_<extractor>.npz, so retraining only embeds new photos.

Usage: python -m src.features.extract [--extractor NAME]
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("features")


class Extractor:
    """Wraps a timm backbone with its classifier removed (num_classes=0 -> pooled features)."""

    def __init__(self, name: str, pretrained: bool = True):
        import timm
        import torch

        self.torch = torch
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.name = name
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        with torch.inference_mode():  # num_features is not the pooled output size for every model
            self.dim = self.model(torch.zeros(1, *cfg["input_size"])).shape[1]
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def embed(self, images) -> np.ndarray:
        """images: list of PIL images -> (n, dim) float32 array."""
        batch = self.torch.stack([self.transform(im.convert("RGB")) for im in images])
        with self.torch.inference_mode():
            return self.model(batch).numpy().astype(np.float32)


def cache_path(extractor: str):
    return config.PROCESSED_DIR / f"embeddings_{extractor.replace('/', '_')}.npz"


def load_cache(extractor: str) -> dict[int, np.ndarray]:
    path = cache_path(extractor)
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["photo_id"].tolist(), z["X"]))


def build(extractor_name: str | None = None, pretrained: bool = True) -> pd.DataFrame:
    """Embed every photo listed in metadata.csv that is not cached yet. Returns timing stats."""
    from PIL import Image

    p = config.params()["features"]
    extractor_name = extractor_name or p["extractor"]
    meta = pd.read_csv(config.METADATA_CSV)
    cache = load_cache(extractor_name)
    todo = meta[~meta.photo_id.isin(cache.keys())]
    log.info("%s: %d cached, %d to embed", extractor_name, len(cache), len(todo))
    if len(todo):
        ex = Extractor(extractor_name, pretrained=pretrained)
        t0 = time.perf_counter()
        for i in range(0, len(todo), p["batch_size"]):
            chunk = todo.iloc[i: i + p["batch_size"]]
            ims = [Image.open(config.ROOT / path) for path in chunk.local_path]
            for pid, vec in zip(chunk.photo_id, ex.embed(ims)):
                cache[int(pid)] = vec
        log.info("embedded %d images in %.1fs", len(todo), time.perf_counter() - t0)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(cache.keys()), dtype=np.int64)
        np.savez_compressed(cache_path(extractor_name), photo_id=ids, X=np.stack([cache[i] for i in ids]))
    return meta


def dataset(extractor_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(X, y, meta) aligned on photos that have an embedding."""
    meta = pd.read_csv(config.METADATA_CSV)
    cache = load_cache(extractor_name)
    meta = meta[meta.photo_id.isin(cache.keys())].reset_index(drop=True)
    X = np.stack([cache[i] for i in meta.photo_id])
    return X, meta.taxon_name.to_numpy(), meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default=None)
    build(ap.parse_args().extractor)
