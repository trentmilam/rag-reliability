"""Leakprobe -- retrievability-ranked PII audit with minimal redaction.

The wedge (vs prior art):

  * Presidio and friends are EXISTENCE-based: they detect PII and redact every
    occurrence. That over-redacts -- most PII in a corpus is never actually
    surfaced by retrieval, yet blanket masking degrades every chunk it touches.
  * rag-corpus-profiler and similar are STATIC: they profile the corpus but do
    not model what a retriever can actually reach.

Leakprobe is RETRIEVABILITY-ranked: it scores each detected PII span by whether
a LIVE vector index actually surfaces the span's chunk in top-k for an
auto-synthesized, PII-eliciting query, and recommends the MINIMAL redaction set
-- redact only what is reachable. Headline claim: recall preserved vs a blanket
mask, because unreachable PII (boilerplate, tails) is left intact.

Everything here is deterministic and offline. Retrieval uses the vendored hash
embedder (embedder.py); no learned model, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from embedder import embed, embed_batch

# --------------------------------------------------------------------------
# PII detection
# --------------------------------------------------------------------------

# SSN checked before phone (both are 3-digit-led) so an SSN is not mislabeled.
EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b")

# "names-in-fixture": a closed known-names list (stands in for an NER gazetteer).
DEFAULT_NAMES = (
    "jane doe",
    "john roe",
    "mary major",
    "richard miles",
)

TYPE_KEYWORDS = {
    "email": ["email", "address", "contact"],
    "ssn": ["social", "security", "number", "ssn"],
    "phone": ["phone", "number", "call"],
    "name": ["name", "person", "who"],
}


@dataclass(frozen=True)
class Span:
    chunk_id: int
    kind: str  # email | ssn | phone | name
    value: str
    start: int  # char offset in the chunk (inclusive)
    end: int  # char offset in the chunk (exclusive)

    def key(self) -> tuple:
        return (self.chunk_id, self.start, self.end, self.kind)


@dataclass
class Chunk:
    id: int
    text: str


def detect_pii(chunk: Chunk, names: tuple[str, ...] = DEFAULT_NAMES) -> list[Span]:
    """Regex + gazetteer PII detection. Returns non-overlapping spans."""
    found: list[Span] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    def add(kind: str, m: re.Match) -> None:
        s, e = m.start(), m.end()
        if not overlaps(s, e):
            found.append(Span(chunk.id, kind, chunk.text[s:e], s, e))
            claimed.append((s, e))

    for m in EMAIL_RE.finditer(chunk.text):
        add("email", m)
    for m in SSN_RE.finditer(chunk.text):
        add("ssn", m)
    for m in PHONE_RE.finditer(chunk.text):
        add("phone", m)
    for name in names:
        for m in re.finditer(re.escape(name), chunk.text, re.IGNORECASE):
            add("name", m)

    found.sort(key=lambda sp: sp.start)
    return found


# --------------------------------------------------------------------------
# Query synthesis (auto-elicit PII)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _context_tokens(text: str, start: int, end: int, window: int) -> list[str]:
    """Tokens within `window` words on each side of [start,end), excluding the
    span itself. This models an attacker who knows the CONTEXT of a secret
    (who/what it is about) but not the secret value, and queries for it."""
    before = [m.group(0).lower() for m in _WORD_RE.finditer(text[:start])]
    after = [m.group(0).lower() for m in _WORD_RE.finditer(text[end:])]
    left = before[-window:] if window else []
    right = after[:window] if window else []
    return left + right


def synthesize_query(chunk: Chunk, span: Span, window: int = 6) -> str:
    """Build a PII-eliciting query from the span's type keywords + context.

    Deliberately does NOT include the secret value -- retrieval must reach the
    chunk from context alone, which is what makes reachability meaningful."""
    kw = TYPE_KEYWORDS.get(span.kind, [])
    ctx = _context_tokens(chunk.text, span.start, span.end, window)
    return " ".join(kw + ctx)


# --------------------------------------------------------------------------
# Live index + retrievability scoring
# --------------------------------------------------------------------------


@dataclass
class Index:
    chunks: list[Chunk]
    matrix: np.ndarray  # (n_chunks, dim), rows aligned to chunks order
    id_to_row: dict[int, int]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Index":
        mat = embed_batch([c.text for c in chunks])
        id_to_row = {c.id: i for i, c in enumerate(chunks)}
        return cls(chunks, mat, id_to_row)

    def topk(self, query: str, k: int) -> list[int]:
        """Return chunk ids of the top-k most similar chunks (cosine)."""
        q = embed(query)
        sims = self.matrix @ q  # unit vectors -> dot == cosine
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.chunks[i].id for i in order]


@dataclass
class SpanScore:
    span: Span
    query: str
    reachable: bool
    rank: int  # 1-based rank of the span's chunk for its query, or -1 if > k


def score_retrievability(
    index: Index, spans: list[Span], k: int, window: int = 6
) -> list[SpanScore]:
    """For each span, synthesize its eliciting query and check whether the
    span's own chunk lands in the LIVE index top-k."""
    id_to_chunk = {c.id: c for c in index.chunks}
    scores: list[SpanScore] = []
    for sp in spans:
        chunk = id_to_chunk[sp.chunk_id]
        q = synthesize_query(chunk, sp, window)
        hits = index.topk(q, k)
        if sp.chunk_id in hits:
            rank = hits.index(sp.chunk_id) + 1
            scores.append(SpanScore(sp, q, True, rank))
        else:
            scores.append(SpanScore(sp, q, False, -1))
    return scores


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

MASK_TOKEN = "redactedtoken"
MASK_REPEAT = 4  # replacement injects this many mask tokens per span


def _mask_replacement() -> str:
    return " " + " ".join([MASK_TOKEN] * MASK_REPEAT) + " "


def apply_redaction(chunks: list[Chunk], spans: list[Span]) -> list[Chunk]:
    """Return new chunks with the given spans replaced by mask tokens.

    Replacement (not deletion) models a real masking pipeline that leaves a
    placeholder; the placeholder dilutes the chunk vector, which is exactly the
    retrieval-quality cost this tool is quantifying."""
    by_chunk: dict[int, list[Span]] = {}
    for sp in spans:
        by_chunk.setdefault(sp.chunk_id, []).append(sp)
    out: list[Chunk] = []
    for c in chunks:
        spans_here = sorted(by_chunk.get(c.id, []), key=lambda s: s.start, reverse=True)
        text = c.text
        for sp in spans_here:  # right-to-left keeps offsets valid
            text = text[: sp.start] + _mask_replacement() + text[sp.end :]
        out.append(Chunk(c.id, text))
    return out


# --------------------------------------------------------------------------
# Recall measurement (utility preserved?)
# --------------------------------------------------------------------------


def recall_at_k(chunks: list[Chunk], utility_queries: list[tuple[str, int]], k: int) -> float:
    """Fraction of utility queries whose gold chunk is in the top-k of a freshly
    built index over `chunks`."""
    if not utility_queries:
        return 1.0
    idx = Index.build(chunks)
    hit = 0
    for q, gold in utility_queries:
        if gold in idx.topk(q, k):
            hit += 1
    return hit / len(utility_queries)


def over_redaction_rate(res: AuditResult) -> float:
    """Fraction of detected PII spans that a Presidio-style *blanket* mask would
    redact but Leakprobe's minimal (reachable-only) set spares.

    over_redaction_rate = (blanket_n - minimal_n) / blanket_n

    This is the corpus-scale head-to-head number: how much redaction the
    incumbent existence-based approach performs that reachability analysis shows
    to be unnecessary. 0.0 means nothing was spared (minimal == blanket)."""
    blanket_n = len(res.blanket_redaction)
    if blanket_n == 0:
        return 0.0
    return (blanket_n - len(res.minimal_redaction)) / blanket_n


# --------------------------------------------------------------------------
# Top-level audit
# --------------------------------------------------------------------------


@dataclass
class AuditResult:
    all_spans: list[Span]
    scores: list[SpanScore]
    reachable_spans: list[Span] = field(default_factory=list)
    minimal_redaction: list[Span] = field(default_factory=list)  # == reachable
    blanket_redaction: list[Span] = field(default_factory=list)  # == all spans

    @property
    def leak_count(self) -> int:
        return len(self.reachable_spans)


def audit(chunks: list[Chunk], k: int = 3, window: int = 6,
          names: tuple[str, ...] = DEFAULT_NAMES) -> AuditResult:
    """Full audit: detect PII, score retrievability against a live index,
    recommend the minimal (reachable-only) redaction set."""
    all_spans: list[Span] = []
    for c in chunks:
        all_spans.extend(detect_pii(c, names))
    index = Index.build(chunks)
    scores = score_retrievability(index, all_spans, k, window)
    reachable = [s.span for s in scores if s.reachable]
    return AuditResult(
        all_spans=all_spans,
        scores=scores,
        reachable_spans=reachable,
        minimal_redaction=list(reachable),
        blanket_redaction=list(all_spans),
    )
