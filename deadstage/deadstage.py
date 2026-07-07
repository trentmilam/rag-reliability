"""Deadstage — a judge-free RAG liveness gate that NAMES the dead stage.

A read-only probe over a RAG pipeline's *artifacts* (not its code). It asserts
one structural invariant per stage across

    ingest -> index -> embed -> retrieve -> score

and, walking the stages in order, reports the SINGLE earliest stage whose
invariant fails. Upstream death causes downstream symptoms, so naming the
FIRST dead stage points at the root cause instead of the last thing that threw.

Wedge (positioning, not invention): failover / fallback layers keep a broken
pipeline *answering* by masking the dead stage; Deadstage does the opposite —
it asserts and NAMES, and returns non-zero so CI stops. The invariants
themselves are standard (shape/norm/cardinality/ordering checks and an
anisotropy floor); we claim novelty only on the assert-and-name posture.

Everything here is deterministic and offline: numpy + stdlib only.

Public API
----------
    build_pipeline(docs, queries, *, embed_kwargs=None, k=3, seed=SEED) -> PipelineState
    check(state) -> Report          # read-only; .dead_stage is None when healthy
    main(argv) -> int               # CLI; exit 0 healthy, non-zero when a stage is dead
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

from embedder import REF_DIM, embed_batch

SEED = 1234

# Invariant thresholds (documented, not tuned to the fixture).
NORM_EPS = 1e-6        # a vector below this L2 norm is a zero/collapsed vector
ANISO_MAX = 0.98       # mean off-diagonal cosine above this == representational collapse
SCORE_EPS = 1e-9       # min top-k score spread for a query to be discriminative


@dataclass
class PipelineState:
    """The read-only artifacts a real RAG pipeline would emit at each stage."""

    docs: list[dict]                       # ingest:  [{"id":..., "text":...}, ...]
    index_ids: list                        # index:   ids registered in the vector index
    embeddings: np.ndarray                 # embed:   (n_index, dim) float matrix
    dim: int                               # embed:   expected embedding width
    queries: list[str]                     # retrieve inputs
    retrieval: list[list[tuple]]           # retrieve/score: per-query [(doc_id, score), ...]
    k: int                                 # requested top-k


@dataclass
class Report:
    dead_stage: str | None = None          # e.g. "embed" (None == healthy)
    reason: str | None = None              # short machine-ish tag, e.g. "collapsed/zero-norm"
    detail: str | None = None              # human sentence with the measured number
    stage_status: dict = field(default_factory=dict)  # stage -> "ok"/"dead"/"skipped"

    @property
    def healthy(self) -> bool:
        return self.dead_stage is None

    def line(self) -> str:
        if self.healthy:
            return "healthy: all stages live (ingest->index->embed->retrieve->score)"
        return f"{self.dead_stage}: {self.reason}"


# --------------------------------------------------------------------------- #
# Pipeline builder (a minimal, real cosine-retrieval RAG over the embedder).
# --------------------------------------------------------------------------- #
def _cosine_topk(qv: np.ndarray, mat: np.ndarray, ids: list, k: int) -> list[tuple]:
    qn = float(np.linalg.norm(qv))
    dn = np.linalg.norm(mat, axis=1)
    denom = dn * qn
    sims = np.zeros(mat.shape[0], dtype=np.float64)
    nz = denom > 0.0
    sims[nz] = (mat[nz] @ qv) / denom[nz]
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def build_pipeline(
    docs: list[dict],
    queries: list[str],
    *,
    embed_kwargs: dict | None = None,
    k: int = 3,
    seed: int = SEED,
) -> PipelineState:
    """Run a tiny real RAG and capture its per-stage artifacts (no repair)."""
    _ = np.random.default_rng(seed)  # determinism handshake; embedder is hash-based
    embed_kwargs = dict(embed_kwargs or {})
    dim = int(embed_kwargs.get("dim", REF_DIM))

    index_ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    embeddings = embed_batch(texts, **embed_kwargs)

    qmat = embed_batch(queries, **embed_kwargs) if queries else np.zeros((0, dim), np.float32)
    retrieval: list[list[tuple]] = []
    for i in range(len(queries)):
        if embeddings.shape[0] == 0:
            retrieval.append([])
        else:
            retrieval.append(_cosine_topk(qmat[i], embeddings, index_ids, k))

    return PipelineState(
        docs=docs,
        index_ids=index_ids,
        embeddings=embeddings,
        dim=dim,
        queries=queries,
        retrieval=retrieval,
        k=k,
    )


# --------------------------------------------------------------------------- #
# Artifact-ingestion path: probe artifacts CAPTURED from a real pipeline.
# --------------------------------------------------------------------------- #
def from_artifacts(
    docs: list[dict],
    index_ids: list,
    embeddings,
    queries: list[str],
    retrieval: list[list[tuple]],
    *,
    dim: int | None = None,
    k: int = 3,
) -> PipelineState:
    """Ingest externally-captured RAG artifacts into a read-only PipelineState.

    This is the counterpart to :func:`build_pipeline`. Where ``build_pipeline``
    manufactures self-consistent artifacts from the vendored embedder,
    ``from_artifacts`` accepts the ``index_ids`` / ``embeddings`` / ``retrieval``
    a *real* pipeline already emitted — so Deadstage can probe them directly.

    It performs NO repair and asserts NO invariants (that is :func:`check`'s
    job): it only marshals the inputs into the dataclass. Because the retrieval
    is provided rather than recomputed from the embeddings, the retrieve and
    score invariants become reachable on real, possibly-inconsistent data (e.g.
    a live retriever that returns out-of-index ids or tied scores) — states that
    ``build_pipeline`` can never produce.

    ``embeddings`` may be a numpy array or a nested list; it is coerced to
    float32. ``dim`` defaults to the embedding width (falling back to REF_DIM
    when the width can't be inferred).
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if dim is None:
        dim = int(emb.shape[1]) if emb.ndim == 2 and emb.shape[1] else REF_DIM
    retr = [[(doc_id, float(score)) for (doc_id, score) in res] for res in retrieval]
    return PipelineState(
        docs=list(docs),
        index_ids=list(index_ids),
        embeddings=emb,
        dim=int(dim),
        queries=list(queries),
        retrieval=retr,
        k=int(k),
    )


# --------------------------------------------------------------------------- #
# The probe: one invariant per stage, walked in order; name the first failure.
# --------------------------------------------------------------------------- #
def _check_ingest(s: PipelineState):
    if not s.docs:
        return "empty/no-docs", "ingest produced 0 documents"
    n_text = sum(1 for d in s.docs if str(d.get("text", "")).strip())
    if n_text == 0:
        return "empty/no-text", f"all {len(s.docs)} ingested docs have blank text"
    return None, None


def _check_index(s: PipelineState):
    if len(s.index_ids) != len(s.docs):
        return "count-mismatch", f"index has {len(s.index_ids)} ids for {len(s.docs)} docs"
    if len(set(s.index_ids)) != len(s.index_ids):
        return "duplicate-ids", "index contains duplicate doc ids"
    doc_ids = {d["id"] for d in s.docs}
    if any(i not in doc_ids for i in s.index_ids):
        return "dangling-ids", "index references ids absent from ingest"
    return None, None


def _check_embed(s: PipelineState):
    e = s.embeddings
    if e.ndim != 2 or e.shape[0] != len(s.index_ids):
        return "shape-mismatch", f"embeddings shape {e.shape} != ({len(s.index_ids)}, {s.dim})"
    if e.shape[1] != s.dim:
        return "dim-mismatch", f"embedding width {e.shape[1]} != expected {s.dim}"
    if not np.all(np.isfinite(e)):
        return "nan-or-inf", "embeddings contain NaN/Inf"
    norms = np.linalg.norm(e, axis=1)
    n_dead = int(np.sum(norms < NORM_EPS))
    if n_dead > 0:
        return "collapsed/zero-norm", (
            f"{n_dead}/{e.shape[0]} embedding vectors have ~zero L2 norm "
            f"(min={float(norms.min()):.3e})"
        )
    # anisotropy floor: are all vectors pointing essentially the same way?
    unit = e / norms[:, None]
    if unit.shape[0] >= 2:
        gram = unit @ unit.T
        off = gram[~np.eye(gram.shape[0], dtype=bool)]
        mean_off = float(np.mean(off))
        if mean_off > ANISO_MAX:
            return "anisotropic-collapse", (
                f"mean off-diagonal cosine {mean_off:.4f} > {ANISO_MAX} "
                "(vectors near-identical; no representational spread)"
            )
    return None, None


def _check_retrieve(s: PipelineState):
    if not s.queries:
        return None, None
    valid = set(s.index_ids)
    for qi, res in enumerate(s.retrieval):
        if len(res) == 0:
            return "empty-result", f"query {qi} returned no candidates"
        if any(doc_id not in valid for (doc_id, _) in res):
            return "invalid-ids", f"query {qi} returned ids not in the index"
    return None, None


def _check_score(s: PipelineState):
    if not s.queries:
        return None, None
    for qi, res in enumerate(s.retrieval):
        scores = [sc for (_, sc) in res]
        if any(not np.isfinite(sc) for sc in scores):
            return "non-finite", f"query {qi} has NaN/Inf scores"
        if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
            return "unordered", f"query {qi} scores not sorted descending"
        if len(scores) >= 2 and (max(scores) - min(scores)) <= SCORE_EPS:
            return "degenerate/tied", (
                f"query {qi} top-{len(scores)} scores all tied "
                f"(spread={max(scores) - min(scores):.3e}); ranking is meaningless"
            )
    return None, None


_STAGES = [
    ("ingest", _check_ingest),
    ("index", _check_index),
    ("embed", _check_embed),
    ("retrieve", _check_retrieve),
    ("score", _check_score),
]


def check(state: PipelineState) -> Report:
    """Read-only probe. Returns the earliest dead stage (assert, don't mask)."""
    rep = Report()
    hit = False
    for name, fn in _STAGES:
        if hit:
            rep.stage_status[name] = "skipped"
            continue
        reason, detail = fn(state)
        if reason is None:
            rep.stage_status[name] = "ok"
        else:
            rep.stage_status[name] = "dead"
            rep.dead_stage, rep.reason, rep.detail = name, reason, detail
            hit = True
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_state_from_json(path: str) -> PipelineState:
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    docs = blob["docs"]
    queries = blob.get("queries", [])
    return build_pipeline(
        docs,
        queries,
        embed_kwargs=blob.get("embed_kwargs"),
        k=int(blob.get("k", 3)),
        seed=int(blob.get("seed", SEED)),
    )


def _load_artifacts_from_json(path: str) -> PipelineState:
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    retrieval = [[tuple(pair) for pair in res] for res in blob.get("retrieval", [])]
    return from_artifacts(
        docs=blob["docs"],
        index_ids=blob["index_ids"],
        embeddings=blob["embeddings"],
        queries=blob.get("queries", []),
        retrieval=retrieval,
        dim=blob.get("dim"),
        k=int(blob.get("k", 3)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deadstage: name the dead RAG stage.")
    ap.add_argument("--pipeline", help="JSON file: {docs, queries, embed_kwargs, k, seed} "
                    "(builds a tiny cosine-retrieval RAG, then probes it)")
    ap.add_argument("--artifacts", help="JSON file of artifacts CAPTURED from a real "
                    "pipeline: {docs, index_ids, embeddings, queries, retrieval, dim, k}")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    if bool(args.pipeline) == bool(args.artifacts):
        ap.error("provide exactly one of --pipeline / --artifacts (see README)")

    if args.artifacts:
        state = _load_artifacts_from_json(args.artifacts)
    else:
        state = _load_state_from_json(args.pipeline)
    rep = check(state)

    if args.json:
        print(json.dumps({
            "dead_stage": rep.dead_stage,
            "reason": rep.reason,
            "detail": rep.detail,
            "stage_status": rep.stage_status,
            "healthy": rep.healthy,
        }, indent=2))
    else:
        print(rep.line())
        if not rep.healthy:
            print(f"  detail: {rep.detail}")
            print(f"  stages: {rep.stage_status}")

    return 0 if rep.healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
