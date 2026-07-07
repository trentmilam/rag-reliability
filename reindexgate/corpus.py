"""Deterministic synthetic corpus + label-free probe-query mining.

The corpus is public-style prose built from a fixed vocabulary so the whole
demo is reproducible with no data files and no network. Structure:

  * N_DOCS documents, paired up (2j, 2j+1) as near-duplicate PARAPHRASES that
    share most of their distinctive "signature" tokens. Paraphrase pairs give
    each query MORE THAN ONE genuinely-relevant document, which is what makes
    the pooling-bias correction have teeth (a system can be the *only* one to
    find the paraphrase partner).
  * Each doc also gets shared filler tokens (topical noise) so that a
    lower-capacity index (fewer hash buckets) genuinely confuses documents.

Probe-query mining is LABEL-FREE: for each document we mine a query from that
document's own lowest-document-frequency (most distinctive) tokens. The source
document id is therefore known *structurally*, not by any human/LLM label. This
is the known-item / inverse-cloze convention.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

N_DOCS = 60  # 30 paraphrase pairs
SIG_PER_DOC = 3  # distinctive signature tokens carried by a doc
QUERY_LEN = 3  # tokens mined per probe query

_FILLER = [
    "system", "data", "model", "value", "result", "process", "method",
    "case", "report", "index", "record", "field", "entry", "summary",
    "section", "table", "note", "review", "status", "detail",
]


@dataclass(frozen=True)
class Doc:
    doc_id: int
    text: str
    pair_id: int  # paraphrase group; both docs in a pair share this


@dataclass(frozen=True)
class Probe:
    query: str
    source_doc: int  # structurally-known origin (not a human label)


def build_corpus(seed: int = 1234) -> list[Doc]:
    """Build the fixed synthetic corpus. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    docs: list[Doc] = []
    n_pairs = N_DOCS // 2
    for j in range(n_pairs):
        # Two signatures SHARED by the paraphrase pair, plus one unique each.
        shared = [f"sig{j:02d}alpha", f"sig{j:02d}beta"]
        uniq_a = f"uniq{2 * j:02d}"
        uniq_b = f"uniq{2 * j + 1:02d}"
        filler = list(rng.choice(_FILLER, size=6, replace=True))
        # Repeat signatures so they dominate the bag-of-words over filler.
        body_a = (shared * 4) + [uniq_a] * 4 + filler
        body_b = (shared * 4) + [uniq_b] * 4 + list(rng.choice(_FILLER, size=6, replace=True))
        rng.shuffle(body_a)
        rng.shuffle(body_b)
        docs.append(Doc(doc_id=2 * j, text=" ".join(body_a), pair_id=j))
        docs.append(Doc(doc_id=2 * j + 1, text=" ".join(body_b), pair_id=j))
    return docs


def _doc_freq(docs: list[Doc]) -> Counter:
    df: Counter = Counter()
    for d in docs:
        for tok in set(d.text.split()):
            df[tok] += 1
    return df


def mine_probes(docs: list[Doc], query_len: int = QUERY_LEN) -> list[Probe]:
    """Mine one probe query per document from its most distinctive tokens.

    Distinctiveness = lowest corpus document-frequency. Ties broken by token
    string for determinism. This is fully label-free: no relevance judgments.
    """
    df = _doc_freq(docs)
    probes: list[Probe] = []
    for d in docs:
        toks = sorted(set(d.text.split()), key=lambda t: (df[t], t))
        q = toks[:query_len]
        probes.append(Probe(query=" ".join(q), source_doc=d.doc_id))
    return probes
