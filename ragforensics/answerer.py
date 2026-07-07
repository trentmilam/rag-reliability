"""SIMULATED, deterministic RAG answerer + retriever (offline, no LLM, no network).

This module STANDS IN for a real (retriever + generator) RAG stack so the
forensics eval can run offline and deterministically. It is clearly labelled a
SIMULATION: there is no learned model here, only a transparent blend rule. The
point is that the *forensics mechanisms* in `forensics.py` treat this answerer as
an opaque callable `answer_fn(query_text, passages) -> str` and NEVER peek at its
internals or at any gold label -- exactly as they would against a real black-box
RAG endpoint. The faults the eval catches are emergent from this blend rule; the
verdicts are not hard-coded anywhere.

Blend rule (the only "intelligence"):

    context_support = best cosine(query, passage) over passages that carry an
                      answer token  (== "can I ground an answer in the docs?")
    par_conf        = the model's internal confidence it already knows the answer
                      ("parametric memory"); 0 for topics it has never seen.

    if par_conf >= context_support and par_conf >= PAR_FLOOR:  emit parametric
    elif context_support >= CTX_FLOOR:                          emit grounded
    else:                                                       emit "IDK"

A *parametric leak* is the first branch firing while good context was available
and the parametric answer disagrees with the docs -- the model asserts internal
memory over the provided evidence. Because emission ignores context in that
branch, the answer is INVARIANT when context is stripped: that invariance is the
label-free signal `forensics.py` keys on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embedder import embed

PAR_FLOOR = 0.35   # min internal confidence before the model trusts its memory
CTX_FLOOR = 0.15   # min doc support before the model will ground an answer
IDK = "IDK"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    answer_token: str | None = None   # the fact this passage carries, if any


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass
class SimAnswerer:
    """Opaque-to-forensics simulated RAG generator.

    parametric: query_id -> (answer_token, confidence in [0,1]).
    hallucinate: query_ids for which the generator IGNORES good context and emits
                 an unsupported token (a pure GENERATOR fault, distinct from a leak).
    """

    parametric: dict[str, tuple[str, float]] = field(default_factory=dict)
    hallucinate: dict[str, str] = field(default_factory=dict)

    def context_support(self, query_text: str, passages: list[Passage]) -> tuple[float, str | None]:
        qv = embed(query_text)
        best_sim, best_tok = 0.0, None
        for p in passages:
            if p.answer_token is None:
                continue
            sim = _cos(qv, embed(p.text))
            if sim > best_sim:
                best_sim, best_tok = sim, p.answer_token
        return best_sim, best_tok

    def answer(self, query: Query, passages: list[Passage]) -> str:
        support, ctx_ans = self.context_support(query.text, passages)

        # Pure generator fault: WITH relevant context available, emit a token that
        # is NOT supported by it (an unfaithful generation). Without context it
        # abstains -- so this is context-DEPENDENT (invariance stays low), which is
        # what separates a plain hallucination from a parametric leak.
        if query.id in self.hallucinate:
            return self.hallucinate[query.id] if support >= CTX_FLOOR else IDK

        par_ans, par_conf = self.parametric.get(query.id, (None, 0.0))

        if par_ans is not None and par_conf >= support and par_conf >= PAR_FLOOR:
            return par_ans
        if support >= CTX_FLOOR and ctx_ans is not None:
            return ctx_ans
        return IDK


@dataclass
class Retriever:
    """Cosine top-k retriever over a passage corpus (vendored hashing embedder)."""

    corpus: list[Passage]

    def retrieve(self, query_text: str, k: int = 4, exclude: set[str] | None = None) -> list[Passage]:
        exclude = exclude or set()
        qv = embed(query_text)
        scored = [
            (_cos(qv, embed(p.text)), p)
            for p in self.corpus
            if p.id not in exclude
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scored[:k]]
