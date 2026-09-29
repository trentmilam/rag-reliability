"""Vendored tiny pure-python hashing embedder (deterministic, offline).

Feature-hashing (the "hashing trick") with BLAKE2b as the hash. No learned
weights, no network, no external deps beyond numpy. This stands in for a real
sentence embedder / retrieval model: it maps text -> a fixed-dim L2-normalized
vector so cosine similarity (== dot product on unit vectors) drives top-k
retrieval exactly the way a live vector index would.

Kept identical in spirit to ../vecstamp/embedder.py (the established
convention for this portfolio track).
"""

from __future__ import annotations

import hashlib

import numpy as np

REF_DIM = 256
REF_SEED = 0xC0FFEE


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def embed(
    text: str,
    *,
    weights_seed: int = REF_SEED,
    dim: int = REF_DIM,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a single string into a float32 vector (L2-normalized by default)."""
    acc = np.zeros(dim, dtype=np.float64)
    seed_bytes = int(weights_seed).to_bytes(8, "little", signed=False)
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode("utf-8"), key=seed_bytes, digest_size=8).digest()
        n = int.from_bytes(h, "little")
        idx = n % dim
        sign = 1.0 if (n >> 63) & 1 else -1.0
        acc[idx] += sign
    vec = acc.astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
    return vec.astype(np.float32)


def embed_batch(texts, **kwargs) -> np.ndarray:
    if not texts:
        return np.zeros((0, kwargs.get("dim", REF_DIM)), dtype=np.float32)
    return np.stack([embed(t, **kwargs) for t in texts], axis=0)
