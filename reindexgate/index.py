"""Chunk + build + search a tiny in-memory vector index over the corpus.

An IndexConfig captures the two reindex knobs ReindexGate is built to police:

  * `embed_dim`  -- embedding dimensionality (fewer buckets = worse index).
                    Shrinking it is our labelled SIMULATION of a dim-truncation
                    reindex regression.
  * `chunker`    -- "sentence" (fine, one chunk per whitespace-delimited span)
                    or "whole" (coarse, one chunk per document). The coarse
                    chunker is a labelled SIMULATION of a worse-chunker reindex.

Both indexes are searched independently in their own embedding space; scores
are only ever compared *within* an index, so cross-dim cosine is never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embedder import REF_DIM, REF_SEED, embed, embed_batch
from corpus import Doc


@dataclass(frozen=True)
class IndexConfig:
    embed_dim: int = REF_DIM
    chunker: str = "sentence"  # "sentence" | "whole"
    embed_seed: int = REF_SEED  # hash family == "which embedding model"
    name: str = "index"


@dataclass
class Index:
    config: IndexConfig
    chunk_doc: np.ndarray  # (n_chunks,) doc_id per chunk
    matrix: np.ndarray = field(repr=False)  # (n_chunks, dim) L2-normalized


def _chunk(doc: Doc, chunker: str) -> list[str]:
    if chunker == "whole":
        return [doc.text]
    # "sentence": treat every ~4 tokens as a chunk span (fine-grained).
    toks = doc.text.split()
    span = 4
    return [" ".join(toks[i:i + span]) for i in range(0, len(toks), span)] or [doc.text]


def build_index(docs: list[Doc], config: IndexConfig) -> Index:
    texts: list[str] = []
    owners: list[int] = []
    for d in docs:
        for c in _chunk(d, config.chunker):
            texts.append(c)
            owners.append(d.doc_id)
    matrix = embed_batch(texts, dim=config.embed_dim, weights_seed=config.embed_seed)
    return Index(config=config, chunk_doc=np.asarray(owners, dtype=np.int64), matrix=matrix)


def search_docs(index: Index, query: str, k: int) -> list[int]:
    """Return the top-k DISTINCT doc ids for a query (chunk ranking collapsed).

    Cosine == dot product on L2-normalized rows. Ties broken by chunk order for
    determinism.
    """
    q = embed(query, dim=index.config.embed_dim, weights_seed=index.config.embed_seed)
    scores = index.matrix @ q
    order = np.argsort(-scores, kind="stable")
    ranked: list[int] = []
    seen: set[int] = set()
    for idx in order:
        d = int(index.chunk_doc[idx])
        if d not in seen:
            seen.add(d)
            ranked.append(d)
        if len(ranked) >= k:
            break
    return ranked
