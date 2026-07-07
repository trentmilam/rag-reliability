"""Vendored tiny pure-python hashing embedder (deterministic, offline).

Feature-hashing (a.k.a. the "hashing trick") with BLAKE2b as the hash.
No learned weights, no network, no external deps beyond numpy. The
`weights_seed` stands in for "which model / which trained weights": two
different seeds produce near-orthogonal embeddings, exactly the way two
genuinely different embedding models would.

The variant knobs (`quantize`, `normalize`, `dim`) let eval.py SIMULATE the
real-world load-time failure modes deterministically and clearly labelled:

    fp16 reference  -> quantize=None, normalize=True,  dim=256   (build baseline)
    q8 drift        -> quantize=8,    normalize=True,  dim=256   (tiny perturbation)
    wrong weights   -> different weights_seed                    (near-orthogonal)
    lost L2 norm    -> normalize=False                           (right dir, wrong len)
    dim mismatch    -> dim=192                                   (shape changed)
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
    quantize: int | None = None,
) -> np.ndarray:
    """Embed a single string into a float32 vector.

    `weights_seed` selects the hash family (== the model weights).
    `quantize=8` applies per-vector symmetric int8 quant then dequant.
    """
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

    if quantize is not None:
        amax = float(np.max(np.abs(vec)))
        if amax > 0.0:
            levels = (1 << (quantize - 1)) - 1  # e.g. 127 for int8
            scale = amax / levels
            q = np.round(vec / scale)
            vec = (q * scale).astype(np.float32)

    return vec.astype(np.float32)


def embed_batch(texts, **kwargs) -> np.ndarray:
    return np.stack([embed(t, **kwargs) for t in texts], axis=0)
