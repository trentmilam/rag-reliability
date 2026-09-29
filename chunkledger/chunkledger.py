"""ChunkLedger: label-free source->chunk mass balance per structural-element type.

The problem: a RAG ingestion chunker takes a source document and emits chunks.
Silently, chunkers DROP or DUPLICATE structural content. A whole table
vanishes, a code block gets cut, a list is truncated, and nobody notices
until retrieval quality craters weeks later. There is usually no label saying
"this document contained 3 tables", so you cannot check completeness against a
gold count.

The core idea: ChunkLedger is SELF-REFERENTIAL. The parsed source is its OWN
reference. We parse the source into structural elements (tables, code blocks,
headings, list items, links, numeric spans), each with a byte range, then check
element by element whether that element's content survived into the union
of the emitted chunks. The check is a real anchoring test (character-shingle
coverage / longest-common-shingle overlap), NOT a token-count heuristic and NOT
a hard-coded expectation.

Outputs:
  * a per-structural-type CONSERVATION ratio ("tables: 2/3 conserved"),
  * a byte-level MANIFEST of every dropped / duplicated span,
  * a run-over-run DRIFT GATE that trips when any per-type ratio regresses.

Deterministic, offline, numpy+stdlib only. No network, no wall-clock, no RNG
in the measurement path (a fixed SEED is threaded through eval for its fixtures).

Clean-room metric names (coined here, not borrowed from any product):
  * conservation ratio: fraction of a type's elements that survived
  * element coverage: per-element shingle-overlap fraction in [0,1]
  * drift gate: run-over-run per-type regression tripwire
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict

# ---- tuning knobs (documented, not magic) --------------------------------
SHINGLE_K = 12       # char n-gram length used for anchoring
CONSERVE_TAU = 0.85  # element coverage >= tau  => the element survived
DUP_TAU = 0.85       # coverage >= tau in >=2 chunks => duplicated

ELEMENT_TYPES = (
    "tables",
    "code_blocks",
    "headings",
    "list_items",
    "links",
    "numeric_spans",
)


# ---- normalization + anchoring primitives --------------------------------
def normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces, strip, lowercase.

    Applied IDENTICALLY to element text and to chunk text so anchoring is not
    fooled by re-wrapping / re-indentation that a chunker may introduce.
    """
    return re.sub(r"\s+", " ", s).strip().lower()


def shingles(text: str, k: int = SHINGLE_K) -> frozenset[str]:
    """Character k-gram set of the normalized text.

    For text shorter than k the whole normalized string is a single shingle,
    so short elements (headings, tiny numeric spans) are still checkable.
    """
    s = normalize(text)
    if not s:
        return frozenset()
    if len(s) <= k:
        return frozenset((s,))
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


CHUNK_SEP = "\x00"  # boundary marker: no real shingle spans two chunks


def _token_boundary_match(needle: str, haystack: str) -> bool:
    """True iff `needle` occurs in `haystack` as a COHERENT token span.

    Requires at least one occurrence whose neighbouring characters do not extend
    the token: if the needle's first char is alphanumeric the char before it must
    not be alphanumeric, and likewise at the tail. This is what stops a short
    element (a bare numeric span like "12") from being falsely matched inside
    an unrelated larger token such as "512".
    """
    n = len(needle)
    if n == 0:
        return True
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return False
        before = haystack[i - 1] if i > 0 else ""
        after = haystack[i + n] if i + n < len(haystack) else ""
        left_ok = not (needle[0].isalnum() and before.isalnum())
        right_ok = not (needle[-1].isalnum() and after.isalnum())
        if left_ok and right_ok:
            return True
        start = i + 1


def _member(s: str, target_text: str) -> bool:
    """Membership of one shingle against target text, anchored to identity.

    A full-length (>= SHINGLE_K char) shingle is specific enough that plain
    substring membership is safe. A SHORTER shingle arises only from a short
    element's anchor token, where raw substring membership fails open (a dropped
    "12" matches inside "512"), so it must match on token boundaries.
    """
    if len(s) >= SHINGLE_K:
        return s in target_text
    return _token_boundary_match(s, target_text)


def coverage(elem_sh: frozenset[str], target_text: str) -> float:
    """Fraction of an element's (anchor) shingles that occur in target_text.

    target_text is a normalized chunk (or the normalized union of chunks joined
    with a boundary marker). Membership is token-boundary aware for short anchor
    shingles (see ``_member``) so a genuinely dropped short element is not falsely
    reported conserved by a coincidental substring hit.
    """
    if not elem_sh:
        return 1.0  # empty element is trivially conserved
    return sum(1 for s in elem_sh if _member(s, target_text)) / len(elem_sh)


def _byte_to_char(src: str, byte_off: int) -> int:
    """Char offset of a byte offset that lands on a UTF-8 char boundary.

    parse_elements records byte offsets (via _byte_off) taken at char
    boundaries, so this round-trips exactly; needed to slice ``src`` (a str)
    when growing a short element to its surrounding token.
    """
    return len(src.encode("utf-8")[:byte_off].decode("utf-8"))


def anchor_shingles(src: str, e: "Element", k: int = SHINGLE_K) -> frozenset[str]:
    """Occurrence-anchored shingles for an element.

    A long element's own text is already >= k normalized chars, so its k-grams
    are specific: plain ``shingles(e.text)``.

    A SHORT element (a heading, a bare numeric span like "12") normalizes to
    fewer than k chars, so a single short shingle matches as a substring almost
    anywhere: a genuinely DROPPED "12" is falsely reported conserved because
    its digits appear inside an unrelated token like "512" elsewhere in the chunk
    union. To anchor it to ITS OWN identity we grow the source slice outward over
    contiguous ALPHANUMERIC characters to its maximal enclosing token (so a digit
    embedded in a word, "1" in "Q1", anchors as "q1", which survives with its
    word), then rely on token-boundary membership (see ``_member``) so a
    standalone short token is not matched inside a larger one. Growing over
    alphanumerics only never crosses a whitespace/block boundary, so an element
    near a chunk edge is not falsely split.
    """
    if len(normalize(e.text)) >= k:
        return shingles(e.text, k)
    cs = _byte_to_char(src, e.byte_start)
    ce = _byte_to_char(src, e.byte_end)
    while cs > 0 and src[cs - 1].isalnum():
        cs -= 1
    while ce < len(src) and src[ce].isalnum():
        ce += 1
    ext = normalize(src[cs:ce])
    if not ext:
        return frozenset()
    if len(ext) >= k:
        return shingles(ext, k)
    return frozenset((ext,))


# ---- structural-element parser (markdown) --------------------------------
@dataclass
class Element:
    type: str
    byte_start: int
    byte_end: int
    text: str


def _byte_off(src: str, char_off: int) -> int:
    return len(src[:char_off].encode("utf-8"))


_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+\S")
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def parse_elements(src: str) -> list[Element]:
    """Parse markdown source into typed structural elements with byte ranges.

    Fenced code blocks are detected first and take priority: their interior is
    NOT re-parsed as tables/lists/etc. Links and numeric spans are scanned over
    the whole source except inside code fences.
    """
    # line spans (char offsets), splitting on '\n' but preserving offsets
    lines: list[tuple[int, int, str]] = []
    off = 0
    for ln in src.split("\n"):
        lines.append((off, off + len(ln), ln))
        off += len(ln) + 1

    elements: list[Element] = []
    code_char_ranges: list[tuple[int, int]] = []

    # --- fenced code blocks ---
    i = 0
    in_code = False
    fence_start = 0
    for idx, (cs, ce, ln) in enumerate(lines):
        if ln.strip().startswith("```") or ln.strip().startswith("~~~"):
            if not in_code:
                in_code = True
                fence_start = cs
            else:
                in_code = False
                block_end = ce
                text = src[fence_start:block_end]
                elements.append(
                    Element("code_blocks", _byte_off(src, fence_start),
                            _byte_off(src, block_end), text)
                )
                code_char_ranges.append((fence_start, block_end))
    # (an unterminated fence is not an error; it is treated as prose)

    def in_code_char(pos: int) -> bool:
        return any(a <= pos < b for a, b in code_char_ranges)

    def line_in_code(cs: int) -> bool:
        return any(a <= cs < b for a, b in code_char_ranges)

    # --- tables: contiguous runs of pipe lines containing a separator row ---
    run: list[int] = []

    def flush_run(run_idx: list[int]) -> None:
        if not run_idx:
            return
        has_sep = any(
            "|" in lines[j][2] and _SEP_RE.match(lines[j][2])
            for j in run_idx
        )
        if has_sep and len(run_idx) >= 2:
            cs = lines[run_idx[0]][0]
            ce = lines[run_idx[-1]][1]
            elements.append(
                Element("tables", _byte_off(src, cs), _byte_off(src, ce),
                        src[cs:ce])
            )

    for idx, (cs, ce, ln) in enumerate(lines):
        if line_in_code(cs):
            flush_run(run); run = []
            continue
        if "|" in ln and ln.strip():
            run.append(idx)
        else:
            flush_run(run); run = []
    flush_run(run)

    # --- headings & list items (line-level, outside code) ---
    for idx, (cs, ce, ln) in enumerate(lines):
        if line_in_code(cs):
            continue
        if ln.lstrip().startswith("#"):
            elements.append(
                Element("headings", _byte_off(src, cs), _byte_off(src, ce), ln)
            )
        elif _LIST_RE.match(ln):
            elements.append(
                Element("list_items", _byte_off(src, cs), _byte_off(src, ce), ln)
            )

    # --- links & numeric spans (inline, outside code) ---
    for m in _LINK_RE.finditer(src):
        if in_code_char(m.start()):
            continue
        elements.append(
            Element("links", _byte_off(src, m.start()),
                    _byte_off(src, m.end()), m.group(0))
        )
    for m in _NUM_RE.finditer(src):
        if in_code_char(m.start()):
            continue
        elements.append(
            Element("numeric_spans", _byte_off(src, m.start()),
                    _byte_off(src, m.end()), m.group(0))
        )

    elements.sort(key=lambda e: (e.byte_start, e.type))
    return elements


# ---- the conservation ledger ---------------------------------------------
def build_ledger(src: str, chunks: list[str],
                 k: int = SHINGLE_K, tau: float = CONSERVE_TAU) -> dict:
    """Compute per-type conservation + a byte-level drop/dup manifest.

    Mechanism (real, label-free): each source element is shingled, then anchored
    against (a) the UNION of all chunk shingles, to decide survived vs dropped,
    and (b) each individual chunk, to detect duplication.
    """
    elements = parse_elements(src)
    chunk_norm = [normalize(c) for c in chunks]
    corpus_text = CHUNK_SEP.join(chunk_norm)

    per_type: dict[str, dict] = {
        t: {"total": 0, "conserved": 0, "dropped": [], "duplicated": []}
        for t in ELEMENT_TYPES
    }
    manifest: list[dict] = []

    for e in elements:
        esh = anchor_shingles(src, e, k)
        cov = coverage(esh, corpus_text)
        hits = [i for i, ct in enumerate(chunk_norm) if coverage(esh, ct) >= DUP_TAU]
        if cov >= tau:
            status = "duplicated" if len(hits) >= 2 else "conserved"
        else:
            status = "dropped"

        pt = per_type.setdefault(
            e.type, {"total": 0, "conserved": 0, "dropped": [], "duplicated": []}
        )
        pt["total"] += 1
        span = [e.byte_start, e.byte_end]
        if status == "dropped":
            pt["dropped"].append(span)
        else:
            pt["conserved"] += 1
            if status == "duplicated":
                pt["duplicated"].append({"span": span, "chunks": hits})

        manifest.append({
            "type": e.type,
            "byte_start": e.byte_start,
            "byte_end": e.byte_end,
            "coverage": round(cov, 4),
            "status": status,
            "chunk_hits": hits,
            "preview": normalize(e.text)[:60],
        })

    ratios = {}
    for t, pt in per_type.items():
        pt["ratio"] = (pt["conserved"] / pt["total"]) if pt["total"] else 1.0
        ratios[t] = round(pt["ratio"], 6)

    return {
        "params": {"shingle_k": k, "conserve_tau": tau, "dup_tau": DUP_TAU},
        "per_type": per_type,
        "ratios": ratios,
        "manifest": manifest,
        "n_chunks": len(chunks),
        "source_bytes": len(src.encode("utf-8")),
    }


# ---- the CI drift gate ---------------------------------------------------
def drift_gate(prior: dict, current: dict, eps: float = 1e-9) -> dict:
    """Trip when any per-type conservation ratio REGRESSES vs the prior run.

    Also reports byte spans newly-dropped since the prior run for triage. This
    is the run-over-run regression tripwire (the "gate").
    """
    regressions = []
    prior_ratios = prior.get("ratios", {})
    cur_ratios = current.get("ratios", {})
    for t in sorted(set(prior_ratios) | set(cur_ratios)):
        p = prior_ratios.get(t, 1.0)
        c = cur_ratios.get(t, 1.0)
        if c < p - eps:
            cur_dropped = current.get("per_type", {}).get(t, {}).get("dropped", [])
            regressions.append({
                "type": t,
                "prior_ratio": round(p, 6),
                "current_ratio": round(c, 6),
                "dropped_spans": cur_dropped,
            })
    return {"tripped": bool(regressions), "regressions": regressions}


# ---- (de)serialization + tiny CLI ----------------------------------------
def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"file not found: {path}")


def _dump(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ChunkLedger: source->chunk conservation")
    ap.add_argument("--source", required=True, help="markdown source file")
    ap.add_argument("--chunks", required=True, help="JSON file: list[str] of chunks")
    ap.add_argument("--prior", help="prior ledger JSON to run the drift gate against")
    ap.add_argument("--out", help="write the current ledger JSON here")
    args = ap.parse_args(argv)

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        raise SystemExit(f"file not found: {args.source}")
    chunks = _load(args.chunks)
    if not isinstance(chunks, list):
        raise SystemExit("--chunks must be a JSON list of strings")

    ledger = build_ledger(src, chunks)
    for t in ELEMENT_TYPES:
        pt = ledger["per_type"][t]
        print(f"  {t:14s}: {pt['conserved']}/{pt['total']} conserved"
              + (f"  DROPPED {pt['dropped']}" if pt["dropped"] else ""))
    if args.out:
        _dump(ledger, args.out)

    rc = 0
    if args.prior:
        gate = drift_gate(_load(args.prior), ledger)
        if gate["tripped"]:
            print("DRIFT GATE: TRIPPED")
            for r in gate["regressions"]:
                print(f"  regression {r['type']}: "
                      f"{r['prior_ratio']} -> {r['current_ratio']} "
                      f"dropped={r['dropped_spans']}")
            rc = 2
        else:
            print("DRIFT GATE: clean")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
