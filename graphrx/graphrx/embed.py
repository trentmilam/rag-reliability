"""A tiny, vendored, pure-python hashing embedder (BLAKE2b hashing trick).

dim=256, L2-normalized, cosine similarity via dot product. No model download,
no network, fully deterministic. This is the SIMULATED embedding backend: it
stands in for a real sentence embedder so the linter runs offline. It is a bag
of hashed tokens: it captures lexical overlap (shared entity name / topic words),
which is exactly the signal the structural checks below rely on.
"""
import hashlib
import re

import numpy as np

DIM = 256
_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOK.findall(text.lower())


def embed(text, dim=DIM):
    """Hash the tokens of `text` into a signed, L2-normalized `dim`-vector."""
    v = np.zeros(dim, dtype=np.float64)
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(h, "big")
        idx = n % dim
        sign = 1.0 if ((n >> 8) & 1) else -1.0
        v[idx] += sign
    norm = np.linalg.norm(v)
    if norm > 0.0:
        v /= norm
    return v


def embed_many(texts, dim=DIM):
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)
    return np.stack([embed(t, dim) for t in texts])


def cosine(a, b):
    """Cosine of two L2-normalized vectors (== dot product)."""
    return float(np.dot(a, b))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0.0 else v
