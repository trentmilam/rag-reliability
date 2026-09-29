"""Legigate: reference-free per-chunk OCR/parse legibility gate.

Scores ALREADY-EXTRACTED text (no images, no ground truth) on two structural
signals that survive extraction but are corrupted by bad OCR/layout parsing:

  1. reading-order scramble  (column bleed / interleaved lines)
     Mechanism: local-coherence transition scoring with a vendored hashing
     embedder. In correctly-ordered prose, adjacent lines are more lexically
     related than lines two apart (mean cos(i,i+1) > mean cos(i,i+2)). When two
     columns are interleaved (A1,B1,A2,B2,...), the *true* continuation sits two
     lines away, so that inequality INVERTS. We score the signed margin.

  2. collapsed / merged table structure  (cell-boundary loss)
     Mechanism: content that *looks* tabular (short lines, regular token counts,
     numeric density) is re-parsed for cell structure via multi-space / tab /
     pipe delimiters. A well-formed table yields >=2 aligned, consistently-
     counted cells per row; a collapsed table (delimiters flattened to single
     spaces or rows merged) yields one blob per row, so the structure score is about 0.

Each signal is scored in [0,1]; higher = more legible. A chunk is QUARANTINED
if any *applicable* signal falls below its threshold. Signals that do not apply
to a chunk (table checks on prose, order checks on a table) score 1.0 and are
reported as not-applicable, so the gate never penalizes the wrong content type.

NOT in scope (delegated to prior art): mojibake/encoding repair
(ftfy) and generic gibberish/language-id filtering (NeMo Curator, datatrove).
Legigate owns only the underserved *structural* signals.
"""
import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from embed import cosine, embed

SEED = 1234  # determinism convention; the pipeline itself uses no RNG.

# --- thresholds (calibrated on the fixtures, then frozen) --------------------
ORDER_THRESHOLD = 0.50   # order_score below this => reading-order scramble
TABLE_THRESHOLD = 0.50   # table_score below this => collapsed table structure
_ORDER_K = 6.0           # sharpness of the margin -> [0,1] squashing

_DELIM = re.compile(r"\t|\s{2,}|\s*\|\s*")   # tab, 2+ spaces, or pipe = cell boundary


def _lines(text):
    return [ln for ln in text.splitlines() if ln.strip()]


def _tok_count(line):
    return len(line.split())


def _numeric_ratio(lines):
    total = 0
    numeric = 0
    for ln in lines:
        for t in ln.split():
            total += 1
            if any(c.isdigit() for c in t):
                numeric += 1
    return (numeric / total) if total else 0.0


# --- signal 1: reading-order coherence --------------------------------------
def score_reading_order(lines):
    """Return (score in [0,1], margin, applicable). Applies to multi-line prose."""
    if len(lines) < 4:
        return 1.0, 0.0, False
    vecs = [embed(ln) for ln in lines]
    lag1 = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    lag2 = [cosine(vecs[i], vecs[i + 2]) for i in range(len(vecs) - 2)]
    if not lag1 or not lag2:
        return 1.0, 0.0, False
    margin = statistics.fmean(lag1) - statistics.fmean(lag2)
    # margin > 0 (adjacent more coherent than 2-apart) => well-ordered.
    # margin < 0 (2-apart more coherent) => alternation / column interleave.
    score = 0.5 + 0.5 * float(np.tanh(_ORDER_K * margin))
    return score, margin, True


# --- signal 2: table structure ----------------------------------------------
def _looks_tabular(lines):
    """Does this content look like it was *meant* to be a table?

    Judged on content, not delimiters, so a COLLAPSED table (delimiters gone)
    is still recognized as tabular content whose structure we then find missing.
    """
    if len(lines) < 3:
        return False
    counts = [_tok_count(ln) for ln in lines]
    if statistics.median(counts) < 3:
        return False
    short = statistics.fmean(1.0 if len(ln) < 60 else 0.0 for ln in lines)
    if short < 0.6:
        return False
    if _numeric_ratio(lines) < 0.15:
        return False
    mode = statistics.mode(counts)
    regular = statistics.fmean(1.0 if c == mode else 0.0 for c in counts)
    return regular >= 0.6


def _alignment_score(lines):
    """Fraction of expected columns whose start position recurs across rows."""
    starts_per_row = []
    for ln in lines:
        starts, pos = [], 0
        # walk cells, recording each cell's start column
        for cell in _DELIM.split(ln):
            if cell == "":
                continue
            idx = ln.find(cell, pos)
            if idx >= 0:
                starts.append(idx)
                pos = idx + len(cell)
        if len(starts) >= 2:
            starts_per_row.append(starts)
    if len(starts_per_row) < 2:
        return 0.0
    from collections import Counter
    col_hits = Counter(s for row in starts_per_row for s in set(row))
    ncols = statistics.median(len(r) for r in starts_per_row)
    aligned = sum(1 for _, h in col_hits.items() if h >= 0.6 * len(starts_per_row))
    return min(1.0, aligned / ncols) if ncols else 0.0


def score_table(lines):
    """Return (score in [0,1], detail, applicable). Applies to tabular content."""
    if not _looks_tabular(lines):
        return 1.0, {"tabular": False}, False
    cell_counts = [len([c for c in _DELIM.split(ln) if c != ""]) for ln in lines]
    frac_multicell = statistics.fmean(1.0 if c >= 2 else 0.0 for c in cell_counts)
    multi = [c for c in cell_counts if c >= 2]
    if multi:
        mode = statistics.mode(multi)
        consistency = statistics.fmean(1.0 if c == mode else 0.0 for c in multi)
    else:
        consistency = 0.0
    alignment = _alignment_score(lines)
    score = 0.5 * frac_multicell + 0.3 * consistency + 0.2 * alignment
    detail = {
        "tabular": True,
        "frac_multicell": round(frac_multicell, 3),
        "consistency": round(consistency, 3),
        "alignment": round(alignment, 3),
        "cell_counts": cell_counts,
    }
    return score, detail, True


# --- the gate ----------------------------------------------------------------
@dataclass
class ChunkVerdict:
    chunk_id: str
    order_score: float
    order_margin: float
    order_applicable: bool
    table_score: float
    table_applicable: bool
    table_detail: dict = field(default_factory=dict)
    legible: bool = True
    reasons: list = field(default_factory=list)


def score_chunk(text, chunk_id="chunk"):
    """Score one chunk on both signals and decide quarantine (reference-free)."""
    lines = _lines(text)
    t_score, t_detail, t_appl = score_table(lines)
    # Reading-order (lexical adjacency) coherence is meaningful only for prose;
    # tabular content is scored by the table signal, not the order signal.
    if t_appl:
        o_score, o_margin, o_appl = 1.0, 0.0, False
    else:
        o_score, o_margin, o_appl = score_reading_order(lines)

    reasons = []
    if o_appl and o_score < ORDER_THRESHOLD:
        reasons.append(
            f"reading-order scramble (score {o_score:.3f} < {ORDER_THRESHOLD}, "
            f"lag1-lag2 margin {o_margin:+.3f} => adjacent lines less coherent "
            f"than lines two apart, i.e. column interleave)")
    if t_appl and t_score < TABLE_THRESHOLD:
        reasons.append(
            f"collapsed table structure (score {t_score:.3f} < {TABLE_THRESHOLD}, "
            f"multicell={t_detail.get('frac_multicell')} "
            f"align={t_detail.get('alignment')})")

    return ChunkVerdict(
        chunk_id=chunk_id,
        order_score=o_score, order_margin=o_margin, order_applicable=o_appl,
        table_score=t_score, table_applicable=t_appl, table_detail=t_detail,
        legible=(len(reasons) == 0), reasons=reasons,
    )


def gate(chunks):
    """chunks: list of (chunk_id, text). Returns (kept, quarantined) verdicts."""
    verdicts = [score_chunk(text, cid) for cid, text in chunks]
    kept = [v for v in verdicts if v.legible]
    quarantined = [v for v in verdicts if not v.legible]
    return kept, quarantined
