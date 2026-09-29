"""Tiny, dependency-free hashing embedder (the vendored convention).

BLAKE2b hashing trick -> dim-256 signed bag-of-tokens -> L2-normalized.
Cosine similarity is just a dot product on the normalized vectors.
Fully deterministic: same text always yields the same vector (no RNG, no
learned weights, no network). This is intentionally weak: it captures
lexical overlap, which is all the reading-order signal needs.
"""
import hashlib
import re

import numpy as np

DIM = 256
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str):
    return _TOKEN.findall(text.lower())


def embed(text: str) -> np.ndarray:
    """Signed hashing-trick embedding of one string. L2-normalized (or zero)."""
    v = np.zeros(DIM, dtype=np.float64)
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(h, "big")
        idx = n % DIM
        sign = 1.0 if (n >> 8) & 1 else -1.0
        v[idx] += sign
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two vectors (0.0 if either is degenerate/zero)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
