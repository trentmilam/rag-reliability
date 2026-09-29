"""VecStamp: re-embedding reproduction certificate with failure-typing.

Build/load-time identity check for a RAG embedding pipeline.

  build_manifest(embed_fn, probes)  -> at BUILD time, embed a fixed set of
      anchor-probe strings, record their vectors, dimension, and a canonical
      float-byte hash. This is the reproduction *certificate*.

  verify(manifest, embed_fn)        -> at LOAD time, re-embed the SAME probes
      with the LIVE embedder and assert bit-exact reproduction. On divergence,
      TYPE the failure via a decision tree:
          dim-mismatch | lost-l2-norm | quant-dtype-drift | wrong-weights

This is strictly a build-vs-load *identity* certificate (did the embedder that
serves queries reproduce the embedder that built the index?). It is distinct
from query-time liveness/answer checks; see README scope note.

Deterministic: probe generation is seeded; no wall-clock, no global random.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

SEED = 20260704

# --- decision-tree thresholds (documented, not magic) -----------------------
# cosine at/above this between live and reference direction == "same model,
# tiny numeric perturbation". Genuine wrong-weights hashing is near-orthogonal
# (cosine ~ 0), so the gap is huge and the threshold is not delicate.
COS_SAME_MODEL = 0.98
# |norm - 1| above this == the L2 normalization step was lost. q8 dequant
# perturbs the norm by <~1e-2, well inside tolerance.
NORM_TOL = 0.05


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
_WORD_POOL = [
    "vector", "index", "retrieval", "embedding", "cosine", "manifest",
    "quantize", "reproduce", "anchor", "probe", "corpus", "query",
    "certificate", "drift", "weights", "normalize", "hash", "load",
]


def make_probes(n: int = 12, seed: int = SEED) -> list[str]:
    """Deterministically synthesize n anchor-probe strings."""
    rng = np.random.default_rng(seed)
    probes = []
    for _ in range(n):
        k = int(rng.integers(3, 8))
        idx = rng.integers(0, len(_WORD_POOL), size=k)
        probes.append(" ".join(_WORD_POOL[i] for i in idx))
    return probes


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _canonical_hash(vecs: np.ndarray) -> str:
    """Hash the canonical float32 byte layout (C-order, little-endian)."""
    b = np.ascontiguousarray(vecs, dtype="<f4").tobytes()
    return hashlib.blake2b(b, digest_size=16).hexdigest()


@dataclass
class Manifest:
    probes: list[str]
    dim: int
    vectors: np.ndarray  # (n, dim) float32, the build-time reference
    float_hash: str
    embedder_name: str = "hashing-blake2b"  # cited: framework stores name/dim


def build_manifest(embed_fn, probes: list[str] | None = None,
                   embedder_name: str = "hashing-blake2b") -> Manifest:
    """BUILD time: persist anchor vectors + canonical float-byte hash."""
    probes = probes if probes is not None else make_probes()
    if not probes:
        raise ValueError("VecStamp: at least one anchor probe is required")
    vecs = np.stack([embed_fn(p) for p in probes], axis=0).astype("<f4")
    return Manifest(
        probes=list(probes),
        dim=int(vecs.shape[1]),
        vectors=vecs,
        float_hash=_canonical_hash(vecs),
        embedder_name=embedder_name,
    )


# ---------------------------------------------------------------------------
# Verify + failure typing
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    ok: bool
    verdict: str            # "reproduced" | one of the failure types
    detail: dict = field(default_factory=dict)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verify(manifest: Manifest, embed_fn) -> VerifyResult:
    """LOAD time: re-embed probes and assert reproduction; type divergence.

    Decision tree (evaluated in order):
      1. dim mismatch        -> live dim != manifest dim
      2. bit-exact match     -> canonical float-hash equal  => reproduced (PASS)
      3. lost L2 norm        -> direction cosine ~1 but |norm-1| >> tol
      4. quant/dtype drift   -> direction cosine >= COS_SAME_MODEL, norms ~1
      5. wrong weights       -> otherwise (low cosine)
    """
    if not manifest.probes:
        raise ValueError("VecStamp: at least one anchor probe is required")
    # re-embed with the LIVE embedder
    live_rows = [np.asarray(embed_fn(p), dtype=np.float32) for p in manifest.probes]
    live_dim = live_rows[0].shape[0]

    # (1) dim mismatch: shape changed, cannot even compare element-wise
    if live_dim != manifest.dim:
        return VerifyResult(
            ok=False, verdict="dim-mismatch",
            detail={"expected_dim": manifest.dim, "live_dim": live_dim},
        )

    live = np.stack(live_rows, axis=0).astype("<f4")

    # (2) bit-exact reproduction: the certificate holds
    live_hash = _canonical_hash(live)
    if live_hash == manifest.float_hash:
        return VerifyResult(
            ok=True, verdict="reproduced",
            detail={"float_hash": live_hash},
        )

    # diverged: measure direction cosine and norm per probe
    ref = manifest.vectors
    cosines = np.array([_row_cosine(live[i], ref[i]) for i in range(len(live))])
    live_norms = np.linalg.norm(live, axis=1)
    mean_cos = float(cosines.mean())
    max_norm_err = float(np.max(np.abs(live_norms - 1.0)))

    detail = {
        "mean_cosine": round(mean_cos, 6),
        "min_cosine": round(float(cosines.min()), 6),
        "max_norm_err": round(max_norm_err, 6),
        "live_hash": live_hash,
        "manifest_hash": manifest.float_hash,
    }

    # (3) lost L2 norm: same direction, wrong magnitude
    if mean_cos >= COS_SAME_MODEL and max_norm_err > NORM_TOL:
        return VerifyResult(ok=False, verdict="lost-l2-norm", detail=detail)

    # (4) quant/dtype drift: same model, tiny numeric perturbation
    if mean_cos >= COS_SAME_MODEL:
        return VerifyResult(ok=False, verdict="quant-dtype-drift", detail=detail)

    # (5) wrong weights: near-orthogonal, a genuinely different model
    return VerifyResult(ok=False, verdict="wrong-weights", detail=detail)
