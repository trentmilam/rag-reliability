"""Deterministic synthetic corpus + labelled fault scenarios for RAGForensics.

Everything here is SIMULATED and clearly labelled. The forensics mechanisms never
see these labels (parametric store, hallucinate set, which passage is 'relevant');
they are used ONLY by eval.py to check that the label-free verdicts are correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from answerer import Passage, Query, Retriever, SimAnswerer

SEED = 20260704

# (query_id, keyword phrase, true answer token, a decoy answer token)
_TOPICS = [
    ("zephyria", "capital city nation zephyria", "portmyr", "mnemos"),
    ("kessaline", "capital city nation kessaline", "vantreve", "oldharbor"),
    ("droman", "capital city nation droman", "solmere", "greyfen"),
    ("halberd", "capital city nation halberd", "kirn", "ashford"),
    ("volturn", "capital city nation volturn", "delfar", "brackwater"),
    ("marisen", "capital city nation marisen", "tallow", "everdeep"),
    ("oskander", "capital city nation oskander", "reike", "farrow"),
    ("pellucid", "capital city nation pellucid", "ombre", "castille"),
    ("nimbus", "capital city nation nimbus", "vell", "starholt"),
    ("cindral", "capital city nation cindral", "morrin", "quill"),
]


def _qtext(qid: str) -> str:
    # query is dominated by the unique nation token so the relevant passage wins
    return f"{qid} {qid} capital gazetteer lookup"


def _relevant(qid: str, phrase: str, ans: str) -> Passage:
    # answer-bearing passage: repeats the unique nation token so cosine sim to its
    # own query is high and cross-topic similarity stays low
    return Passage(id=f"p_{qid}",
                   text=f"{qid} {qid} {qid} gazetteer capital is {ans} entry {ans}",
                   answer_token=ans)


def _distractor(qid: str, phrase: str, decoy: str) -> Passage:
    # off-topic tourism note; carries the decoy token, shares little with the query
    return Passage(id=f"d_{qid}",
                   text=f"{decoy} tourism weather markets seasons brochure {decoy}",
                   answer_token=decoy)


@dataclass
class Bundle:
    corpus: list[Passage]
    answer_fn: object
    retriever: Retriever
    calib_queries: list[Query]
    calib_contexts: dict[str, list[Passage]]
    # named test handles (with their ground-truth labels, for eval assertions only)
    grounded_query: Query
    grounded_ctx: list[Passage]
    grounded_truth: str
    leak_query: Query
    leak_ctx: list[Passage]
    leak_parametric: str          # what the model leaks (disagrees with docs)
    leak_doc_truth: str           # what the docs actually say
    retriever_query: Query
    retriever_ctx: list[Passage]  # relevant passage MISSING (injected retrieval miss)
    generator_query: Query
    generator_ctx: list[Passage]
    generator_halluc: str
    test_cases: list[tuple[Query, list[Passage]]]


def build(rng: np.random.Generator, *, leaky_calib: int = 0,
          inject_leak: bool = True, inject_halluc: bool = True) -> Bundle:
    """leaky_calib: how many of the 6 held-out calibration topics the model answers
    from memory. 0 = a well-grounded corpus (threshold -> floor); 6 = a highly
    context-independent corpus (threshold -> ceil); in between the auto-calibrated
    threshold lands strictly inside the range, proving it is data-derived."""
    corpus: list[Passage] = []
    for qid, phrase, ans, decoy in _TOPICS:
        corpus.append(_relevant(qid, phrase, ans))
        corpus.append(_distractor(qid, phrase, decoy))

    retriever = Retriever(corpus)

    parametric: dict[str, tuple[str, float]] = {}
    hallucinate: dict[str, str] = {}

    # --- calibration set: first 6 topics. `leaky_calib` of them answer from memory
    # (agreeing with docs but context-independent), which lifts the baseline. ---
    for qid, _phrase, ans, _decoy in _TOPICS[:leaky_calib]:
        parametric[qid] = (ans, 0.9)

    calib_queries: list[Query] = []
    calib_contexts: dict[str, list[Passage]] = {}
    for qid, phrase, _ans, _decoy in _TOPICS[:6]:
        q = Query(id=qid, text=_qtext(qid))
        calib_queries.append(q)
        calib_contexts[qid] = retriever.retrieve(q.text, k=4)

    # --- test topics (held OUT of calibration) ---
    # grounded (green): topic 'oskander' -- no parametric, faithful
    g_id, g_phrase, g_ans, _ = _TOPICS[6]
    grounded_query = Query(id=g_id, text=_qtext(g_id))
    grounded_ctx = retriever.retrieve(grounded_query.text, k=4)

    # leak (red): topic 'pellucid' -- model KNOWS a different answer, ignores docs
    l_id, l_phrase, l_ans, l_decoy = _TOPICS[7]
    if inject_leak:
        parametric[l_id] = (l_decoy, 0.95)  # leaks the decoy, disagreeing with docs
    leak_query = Query(id=l_id, text=_qtext(l_id))
    leak_ctx = retriever.retrieve(leak_query.text, k=4)

    # retriever fault (red): topic 'nimbus' -- relevant passage EXCLUDED from index
    r_id, r_phrase, r_ans, _ = _TOPICS[8]
    retriever_query = Query(id=r_id, text=_qtext(r_id))
    retriever_ctx = retriever.retrieve(retriever_query.text, k=4, exclude={f"p_{r_id}"})

    # generator fault (red): topic 'cindral' -- good context, unfaithful generation
    gen_id, gen_phrase, gen_ans, gen_decoy = _TOPICS[9]
    if inject_halluc:
        hallucinate[gen_id] = "phantomtok"  # token not present in the context
    generator_query = Query(id=gen_id, text=_qtext(gen_id))
    generator_ctx = retriever.retrieve(generator_query.text, k=4)

    answerer = SimAnswerer(parametric=parametric, hallucinate=hallucinate)

    test_cases = [
        (grounded_query, grounded_ctx),
        (leak_query, leak_ctx),
        (retriever_query, retriever_ctx),
        (generator_query, generator_ctx),
    ]

    return Bundle(
        corpus=corpus, answer_fn=answerer.answer, retriever=retriever,
        calib_queries=calib_queries, calib_contexts=calib_contexts,
        grounded_query=grounded_query, grounded_ctx=grounded_ctx, grounded_truth=g_ans,
        leak_query=leak_query, leak_ctx=leak_ctx,
        leak_parametric=l_decoy, leak_doc_truth=l_ans,
        retriever_query=retriever_query, retriever_ctx=retriever_ctx,
        generator_query=generator_query, generator_ctx=generator_ctx,
        generator_halluc="phantomtok",
        test_cases=test_cases,
    )
