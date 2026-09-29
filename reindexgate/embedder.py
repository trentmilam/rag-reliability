"""Vendored tiny pure-python hashing embedder (deterministic, offline).

Feature-hashing (the "hashing trick") with BLAKE2b as the hash. No learned
weights, no network, no external deps beyond numpy. This stands in for a real
sentence embedder so ReindexGate can be exercised fully offline.

The `dim` knob is the load-bearing one for ReindexGate: a smaller `dim` means
more hash-bucket collisions, i.e. a lower-capacity index. Shrinking `dim` is
our deterministic, clearly-labelled SIMULATION of a "dim truncation" reindex
regression (e.g. someone re-exports embeddings at fewer dimensions).
"""

from __future__ import annotations

import hashlib

import numpy as np

REF_DIM = 256
REF_SEED = 0xC0FFEE

# Hash family for the reference-relevance ORACLE. It is a DISTINCT family
# from REF_SEED (the default index seed used by OLD and NEW), so the pseudo-
# relevance signal is a genuinely third, near-orthogonal embedding family,
# independent of both indexes under test, not byte-identical to OLD. See
# gate.reference_simmatrix and tests in eval.py (independence check).
REF_ORACLE_SEED = 0xBADCAFE


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


def embed(text: str, *, dim: int = REF_DIM, weights_seed: int = REF_SEED) -> np.ndarray:
    """Embed a string into an L2-normalized float64 vector of length `dim`.

    Two 8-byte slices of one BLAKE2b digest give an independent (bucket, sign)
    pair per token, so the sign is not correlated with the bucket index.
    """
    acc = np.zeros(dim, dtype=np.float64)
    seed_bytes = int(weights_seed).to_bytes(8, "little", signed=False)
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode("utf-8"), key=seed_bytes, digest_size=16).digest()
        bucket = int.from_bytes(h[:8], "little") % dim
        sign = 1.0 if (h[8] & 1) else -1.0
        acc[bucket] += sign
    norm = float(np.linalg.norm(acc))
    if norm > 0.0:
        acc = acc / norm
    return acc


def embed_batch(texts, *, dim: int = REF_DIM, weights_seed: int = REF_SEED) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)
    return np.stack([embed(t, dim=dim, weights_seed=weights_seed) for t in texts], axis=0)
