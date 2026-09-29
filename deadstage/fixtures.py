"""Externally-captured RAG artifacts (deterministic, in-repo fixtures).

These stand in for the artifacts a REAL pipeline emits and that a caller feeds
to Deadstage through the ``from_artifacts`` ingestion path, NOT the
self-consistent artifacts ``build_pipeline`` manufactures from the vendored
embedder. Each fixture encodes a KNOWN injected root-cause stage plus the
realistic *downstream symptom(s)* that failure produces in a live pipeline
(a dead embedder still lets the retriever return crowded/tied candidates; a
dangling index id surfaces as an out-of-index retrieval; etc.).

Encoding the root cause AND its downstream symptoms is exactly what lets the
A/B in ``eval.py`` MEASURE root-cause attribution: the whole point of the
artifact-ingestion path is that ``build_pipeline`` can never emit a state where
an upstream stage is dead *and* a specific downstream stage carries an
independent symptom; it always manufactures a self-consistent state, so the
retrieve/score invariants were previously unreachable on realistic data.

Deterministic: fixed hashing embedder, no random, no wall-clock.
"""

from __future__ import annotations

import numpy as np

from deadstage import from_artifacts
from embedder import REF_DIM, embed_batch

_DOCS = [
    {"id": "d0", "text": "vector databases store dense embeddings for retrieval"},
    {"id": "d1", "text": "cosine similarity ranks candidate passages by angle"},
    {"id": "d2", "text": "chunking splits long documents into overlapping windows"},
    {"id": "d3", "text": "a reranker reorders the shortlist before the answerer"},
]
_QUERIES = ["how are passages ranked by similarity", "what does chunking do"]


def _healthy_embeddings() -> np.ndarray:
    """Real (near-orthogonal) vectors from the vendored hashing embedder."""
    return embed_batch([d["text"] for d in _DOCS])


def _zero_embeddings() -> np.ndarray:
    return np.zeros((len(_DOCS), REF_DIM), dtype=np.float32)


def build_fixtures() -> list[tuple[str, object, str | None]]:
    """Return [(name, PipelineState, true_root_stage), ...].

    ``true_root_stage`` is the ground-truth EARLIEST dead stage that was
    injected; None means the captured pipeline is genuinely healthy.
    """
    ids = ["d0", "d1", "d2", "d3"]

    # 1) HEALTHY: every captured stage is live. Control.
    healthy = from_artifacts(
        docs=_DOCS,
        index_ids=ids,
        embeddings=_healthy_embeddings(),
        queries=_QUERIES,
        retrieval=[
            [("d0", 0.91), ("d1", 0.40), ("d2", 0.10)],
            [("d2", 0.83), ("d0", 0.31), ("d1", 0.05)],
        ],
        k=3,
    )

    # 2) EMBED root: embedder deprecated to zero-vectors (root = embed).
    #    Downstream symptom: the retriever still returns crowded candidates with
    #    all-tied scores -> the SCORE stage looks broken too.
    embed_root = from_artifacts(
        docs=_DOCS,
        index_ids=ids,
        embeddings=_zero_embeddings(),
        queries=_QUERIES,
        retrieval=[
            [("d0", 0.0), ("d1", 0.0), ("d2", 0.0)],
            [("d0", 0.0), ("d1", 0.0), ("d2", 0.0)],
        ],
        k=3,
    )

    # 3) INDEX root: a dangling id registered in the index (root = index).
    #    Downstream symptom: retrieval surfaces an id absent from the index.
    index_root = from_artifacts(
        docs=_DOCS,
        index_ids=["d0", "d1", "d2", "dX"],  # dX dangling; count still matches
        embeddings=_healthy_embeddings(),
        queries=_QUERIES,
        retrieval=[
            [("d9", 0.90), ("d0", 0.40)],  # d9 not in index -> retrieve symptom
            [("d0", 0.80), ("d1", 0.30)],
        ],
        k=3,
    )

    # 4) RETRIEVE root: the retriever returns an out-of-index id (root = retrieve).
    #    Downstream symptom: those same results are tied -> the SCORE stage fails.
    retrieve_root = from_artifacts(
        docs=_DOCS,
        index_ids=ids,
        embeddings=_healthy_embeddings(),
        queries=_QUERIES,
        retrieval=[
            [("d9", 0.50), ("d1", 0.50)],  # d9 invalid AND tied
            [("d0", 0.70), ("d2", 0.20)],
        ],
        k=3,
    )

    # 5) SCORE root: retriever healthy, ranking is degenerate/tied (root = score).
    #    Single symptom at the LAST stage: a fairness control where the
    #    symptom-based baseline SHOULD get the attribution right.
    score_root = from_artifacts(
        docs=_DOCS,
        index_ids=ids,
        embeddings=_healthy_embeddings(),
        queries=_QUERIES,
        retrieval=[
            [("d0", 0.30), ("d1", 0.30)],  # tied -> degenerate score
            [("d2", 0.60), ("d0", 0.10)],
        ],
        k=3,
    )

    return [
        ("healthy", healthy, None),
        ("embed_root", embed_root, "embed"),
        ("index_root", index_root, "index"),
        ("retrieve_root", retrieve_root, "retrieve"),
        ("score_root", score_root, "score"),
    ]
