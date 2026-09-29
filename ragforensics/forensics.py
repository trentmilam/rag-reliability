"""RAGForensics: label-free retriever-vs-generator attribution + parametric-leak gate.

Scope: the individual mechanisms here are COMMODITY. Answer
invariance under context removal, self-consistency resampling, and answer-in-context
support checks all appear in the literature (see README: ContextCite, ReDeEP /
arXiv 2510.12668, RAGChecker/RAG-E). RAGForensics does NOT reimplement any of them.
What it packages is a narrow wedge:

  * LABEL-FREE end to end: no gold answers, no relevance judgments. Every signal
    is computed from the model's own behavior under intervention and from the
    retrieved context, never against a held-out answer key.
  * an AUTO-CALIBRATED, corpus-specific leak threshold (`calibrate_leak_threshold`)
    tuned from held-out queries, instead of a magic global constant, so the gate
    adapts to a corpus's baseline context-independence. This is the one net-new bit.
  * shipped as a CI gate you can run offline / local-first.

The forensics functions treat the RAG system as an opaque callable
`answer_fn(Query, [Passage]) -> str`. They never inspect model internals or labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from answerer import IDK, Passage, Query, Retriever, SimAnswerer, _cos
from embedder import embed


# --------------------------------------------------------------------------- #
# answer comparison (token-level, label-free)
# --------------------------------------------------------------------------- #
def _is_answer(a: str) -> bool:
    return bool(a) and a.strip().upper() != IDK


def answer_similarity(a: str, b: str) -> float:
    """Jaccard token overlap in [0,1]; identical non-empty answers -> 1.0."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------- #
# intervention signals (label-free)
# --------------------------------------------------------------------------- #
def context_removal_invariance(answer_fn, query: Query, passages: list[Passage]) -> float:
    """How much does the answer survive deleting the retrieved context?

    High invariance on a query that HAD real context = the model is answering
    from parametric memory (a leak). If the model declines to answer with full
    context (IDK), invariance is undefined -> reported as 0.0 (no leak to flag).
    """
    a_full = answer_fn(query, passages)
    if not _is_answer(a_full):
        return 0.0
    a_empty = answer_fn(query, [])
    return answer_similarity(a_full, a_empty)


def self_consistency(answer_fn, query: Query, passages: list[Passage],
                     rng: np.random.Generator, k: int = 5) -> float:
    """Fraction of resampled contexts (shuffle + drop-one) that agree with the
    modal answer. Low = unstable generation. Label-free."""
    answers = [answer_fn(query, passages)]
    n = len(passages)
    for _ in range(k):
        perm = list(rng.permutation(n))
        if n > 1 and rng.random() < 0.7:
            perm = perm[:-1]  # drop one passage
        answers.append(answer_fn(query, [passages[i] for i in perm]))
    modal = max(set(answers), key=answers.count)
    return answers.count(modal) / len(answers)


def retrieval_quality(query: Query, passages: list[Passage]) -> float:
    """Best cosine(query, retrieved passage). Label-free proxy for 'did we fetch
    anything topically relevant?'."""
    if not passages:
        return 0.0
    qv = embed(query.text)
    return max(_cos(qv, embed(p.text)) for p in passages)


def answer_supported(answer: str, passages: list[Passage]) -> bool:
    """Faithfulness: is the emitted answer token present in the retrieved context?
    (ContextCite-style support check, computed against CONTEXT not gold.)"""
    if not _is_answer(answer):
        return False
    toks = set()
    for p in passages:
        toks.update(p.text.lower().split())
        if p.answer_token:
            toks.add(p.answer_token.lower())
    return all(t in toks for t in answer.lower().split())


# --------------------------------------------------------------------------- #
# auto-calibrated, corpus-specific leak threshold  (the net-new piece)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeakCalibration:
    threshold: float
    q90: float
    margin: float
    floor: float
    ceil: float
    n_calib: int
    scores: tuple[float, ...]


def calibrate_leak_threshold(answer_fn, calib_queries: list[Query],
                             contexts: dict[str, list[Passage]],
                             *, margin: float = 0.15,
                             floor: float = 0.35, ceil: float = 0.95) -> LeakCalibration:
    """Tune the leak threshold from held-out queries. NO labels.

    Assumes the held-out set is representative of the corpus's *normal* operating
    behavior (a reference grounded-ish traffic). We take the 90th percentile of
    the invariance distribution plus a margin: a query is flagged only if its
    context-removal invariance exceeds what the corpus normally exhibits. Clamped
    to [floor, ceil] so a perfectly-grounded corpus can't drive the gate to 0 and a
    pathological one can't disable it. The RESULT is corpus-specific: corpora with
    higher baseline context-independence get a higher bar."""
    scores = [context_removal_invariance(answer_fn, q, contexts[q.id]) for q in calib_queries]
    arr = np.asarray(scores, dtype=np.float64)
    q90 = float(np.quantile(arr, 0.90)) if arr.size else 0.0
    threshold = float(np.clip(q90 + margin, floor, ceil))
    return LeakCalibration(threshold=threshold, q90=q90, margin=margin,
                           floor=floor, ceil=ceil, n_calib=len(calib_queries),
                           scores=tuple(round(s, 4) for s in scores))


# --------------------------------------------------------------------------- #
# per-query forensics verdict (label-free)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verdict:
    query_id: str
    answer: str
    invariance: float
    self_consistency: float
    retrieval_quality: float
    supported: bool
    leak_flag: bool
    attribution: str   # "retriever" | "generator" | "ok"


def attribute(retr_quality: float, supported: bool, *, retr_floor: float) -> str:
    """Label-free retriever-vs-generator attribution.

    Retriever fault : nothing topically relevant was fetched (rq below the
                      corpus-derived floor), so the generator never had a chance.
    Generator fault : good context was fetched but the answer is not supported by
                      it (unfaithful / hallucinated).
    ok              : relevant context fetched and answer grounded in it."""
    if retr_quality < retr_floor:
        return "retriever"
    if not supported:
        return "generator"
    return "ok"


def analyze(answer_fn, query: Query, passages: list[Passage],
            calib: LeakCalibration, *, retr_floor: float,
            rng: np.random.Generator) -> Verdict:
    ans = answer_fn(query, passages)
    inv = context_removal_invariance(answer_fn, query, passages)
    sc = self_consistency(answer_fn, query, passages, rng)
    rq = retrieval_quality(query, passages)
    sup = answer_supported(ans, passages)
    leak = inv >= calib.threshold and _is_answer(ans)
    attr = attribute(rq, sup, retr_floor=retr_floor)
    return Verdict(query_id=query.id, answer=ans, invariance=round(inv, 4),
                   self_consistency=round(sc, 4), retrieval_quality=round(rq, 4),
                   supported=sup, leak_flag=leak, attribution=attr)


def derive_retrieval_floor(retriever: Retriever, calib_queries: list[Query],
                           *, k: int = 4, frac: float = 0.5) -> float:
    """Corpus-derived relevance floor: a fraction of the median best-retrieval
    quality over held-out queries. Commodity heuristic, not the novel bit."""
    qualities = [retrieval_quality(q, retriever.retrieve(q.text, k=k)) for q in calib_queries]
    med = float(np.median(qualities)) if qualities else 0.0
    return frac * med


# --------------------------------------------------------------------------- #
# tiny CLI (function API is the primary surface; see eval.py)
# --------------------------------------------------------------------------- #
def _demo() -> int:
    import fixtures  # local demo corpus
    rng = np.random.default_rng(fixtures.SEED)
    bundle = fixtures.build(rng)
    calib = calibrate_leak_threshold(bundle.answer_fn, bundle.calib_queries, bundle.calib_contexts)
    retr_floor = derive_retrieval_floor(bundle.retriever, bundle.calib_queries)
    print(f"leak threshold (auto-calibrated) = {calib.threshold:.4f}  "
          f"(q90={calib.q90:.4f}, floor={calib.floor})")
    print(f"retrieval floor (corpus-derived) = {retr_floor:.4f}")
    for q, ctx in bundle.test_cases:
        v = analyze(bundle.answer_fn, q, ctx, calib, retr_floor=retr_floor, rng=rng)
        print(f"  {q.id:14s} ans={v.answer:10s} inv={v.invariance:.2f} "
              f"leak={v.leak_flag!s:5s} attr={v.attribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
