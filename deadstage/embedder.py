"""Vendored tiny pure-python hashing embedder (deterministic, offline).

Feature-hashing ("hashing trick") with BLAKE2b as the hash. No learned
weights, no network, no deps beyond numpy. `weights_seed` stands in for
"which model / which trained weights". Self-contained copy so `deadstage`
has no cross-project import.

The knobs let a caller SIMULATE real load-time failure modes:

    healthy         -> defaults                       (baseline)
    zero-vectors    -> dead=True                      (embedder deprecated to 0)
    constant        -> constant=True                  (representational collapse)
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
    dead: bool = False,
    constant: bool = False,
) -> np.ndarray:
    """Embed one string into a float32 vector.

    `dead=True`     returns the zero vector (embedder deprecated / not loaded).
    `constant=True` returns the same fixed unit vector for every input
                    (weights present but representation collapsed / anisotropic).
    """
    if dead:
        return np.zeros(dim, dtype=np.float32)
    if constant:
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        return v

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
