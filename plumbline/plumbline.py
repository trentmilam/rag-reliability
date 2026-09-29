# -*- coding: utf-8 -*-
"""Plumbline: deterministic chunk->source provenance gate.

The wedge (narrow claim):
  Prove, at INDEX time and with a DETERMINISTIC mechanism, that a stored chunk's
  text traces back to a span in its source document under whitespace/OCR/
  normalization-aware FUZZY alignment. Then, across a reindex, diff the resolved
  provenance to report exactly which citations LOST coverage and where the survivors
  DRIFTED.

Why this is not the same as existing provenance validators:
  * Guardrails' `ProvenanceLLM` / embedding-provenance validators run at ANSWER time
    and are non-deterministic (LLM/embedding thresholded). Plumbline runs at INDEX
    time, is deterministic (pure string canonicalization + substring alignment), and
    needs no model.
  * Content-hash + document versioning tells you a byte changed; it does NOT tell you
    a citation still resolves to a (moved) span, nor by how much it drifted. Plumbline
    reports the realigned span and the coverage-loss delta.

Everything here is CPU-only, stdlib + numpy, offline, and seed-deterministic.

Ambiguity policy (a real limitation, see README "Limitations"):
  a quote that occurs MORE THAN ONCE in the (canonical) source cannot be resolved
  to a single span without more context than a substring match has. Rather than
  silently returning the first occurrence, which can be the WRONG one and mask
  real drift, or blame the wrong occurrence, resolve()/naive_resolve() return
  the AMBIGUOUS sentinel so the caller can see the citation was not (and cannot
  be, by this mechanism) safely resolved.

Public API:
  canonicalize(text) -> (canon_str, index_map)
  resolve(source, quoted_text) -> (start, end) | None | AMBIGUOUS        # fuzzy, canonical-aware
  naive_resolve(source, quoted_text) -> (start, end) | None | AMBIGUOUS  # exact-substring baseline
  build_manifest(source, citations, resolver=resolve) -> dict # {cite_id: span|None|AMBIGUOUS}
  coverage_diff(old_manifest, new_manifest) -> dict    # LOST / DRIFTED / STABLE / GAINED / AMBIGUOUS
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Dict, List, Optional, Tuple

Span = Tuple[int, int]

# Distinguished sentinel returned by resolve()/naive_resolve() when a quote
# matches more than one span: the mechanism cannot safely pick one, so it must
# not silently guess (see the module docstring's "Ambiguity policy").
AMBIGUOUS = "AMBIGUOUS"

# --- canonicalization tables -------------------------------------------------
# One-to-one OCR confusion classes: each variant folds to a single canonical rep.
# Kept intentionally small and one-directional so the canonical<->source index map
# stays exact. (Digits map to letters: a defined tradeoff for OCR text; noted in
# the README. Extend via config, not by hand-editing derived output.)
_CONFUSION = {
    "0": "o",
    "1": "l",
    "|": "l",
    "5": "s",
}

# One-to-many ligature expansions (single source char -> several canonical chars;
# every expanded char maps back to the one source offset).
_LIGATURES = {
    "ﬁ": "fi",   # U+FB01 LATIN SMALL LIGATURE FI
    "ﬂ": "fl",   # U+FB02 LATIN SMALL LIGATURE FL
}


def _fold(ch: str) -> str:
    """Case-fold one char, then apply the OCR confusion class."""
    c = ch.lower()
    return _CONFUSION.get(c, c)


def canonicalize(text: str) -> Tuple[str, List[int]]:
    """Return (canonical_string, index_map).

    index_map[i] is the offset in `text` of the source char that produced
    canonical char i. Whitespace runs collapse to a single space (mapped to the
    first whitespace char); leading/trailing whitespace is dropped; ligatures
    expand; OCR confusion classes fold. The transform is applied IDENTICALLY to
    source and to chunk text so legitimate OCR/normalization variants converge to
    the same canonical bytes.
    """
    canon: List[str] = []
    imap: List[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space and canon:  # collapse run; skip leading ws
                canon.append(" ")
                imap.append(i)
                prev_space = True
            continue
        prev_space = False
        if ch in _LIGATURES:
            for c in _LIGATURES[ch]:
                canon.append(c)
                imap.append(i)
            continue
        canon.append(_fold(ch))
        imap.append(i)
    while canon and canon[-1] == " ":  # strip trailing space
        canon.pop()
        imap.pop()
    return "".join(canon), imap


def resolve(source: str, quoted_text: str) -> Optional[object]:
    """Fuzzy, canonicalization-aware provenance resolution.

    Canonicalize both source and the quoted chunk text, locate the quote's
    canonical form as a substring of the source's canonical form, and map that
    canonical span back to ORIGINAL source offsets [start, end).

    Returns None if the quote does not trace to any source span (coverage
    lost). Returns AMBIGUOUS if the canonical quote occurs MORE THAN ONCE in
    the canonical source: a naive first-match would silently pick a span that
    may be the wrong one (see the module docstring's "Ambiguity policy").
    """
    cq, _ = canonicalize(quoted_text)
    if not cq:
        return None
    csrc, imap = canonicalize(source)
    pos = csrc.find(cq)
    if pos < 0:
        return None
    if csrc.count(cq) > 1:
        return AMBIGUOUS
    start = imap[pos]
    end = imap[pos + len(cq) - 1] + 1
    return (start, end)


def naive_resolve(source: str, quoted_text: str) -> Optional[object]:
    """Exact-substring baseline (what a hash/exact-match provenance check does).

    Fails (FALSE-NEGATIVE) whenever the stored chunk was OCR/whitespace-
    normalized relative to the source, even though the chunk is legitimate.
    Also returns AMBIGUOUS on a repeated exact match, for the same reason
    `resolve` does (see the module docstring's "Ambiguity policy").
    """
    if not quoted_text:
        return None
    pos = source.find(quoted_text)
    if pos < 0:
        return None
    if source.count(quoted_text) > 1:
        return AMBIGUOUS
    return (pos, pos + len(quoted_text))


def build_manifest(
    source: str,
    citations: Dict[str, str],
    resolver: Callable[[str, str], Optional[Span]] = resolve,
) -> Dict[str, Optional[Span]]:
    """Resolve every citation against `source`; return a diffable {cite_id: span}."""
    return {cid: resolver(source, quote) for cid, quote in sorted(citations.items())}


def coverage_diff(
    old: Dict[str, Optional[Span]],
    new: Dict[str, Optional[Span]],
) -> Dict[str, object]:
    """Cross-reindex coverage-loss diff.

    Compares two provenance manifests keyed by citation id:
      * LOST:      resolved before, no longer resolves (coverage gone).
      * DRIFTED:   still resolves but to a different source span (moved); reports
                   from/to spans and the byte delta of the start offset.
      * STABLE:    resolves to the same span.
      * GAINED:    unresolved in the baseline, now resolves (coverage newly
                   acquired). Reported separately rather than folded into
                   STABLE, which would misleadingly imply no change.
      * AMBIGUOUS: either side resolved to more than one candidate span
                   (see AMBIGUOUS); no LOST/DRIFTED/STABLE/GAINED verdict can
                   be made correctly, so the citation is reported separately
                   instead of being silently folded into one of the above.
    """
    lost: List[str] = []
    drifted: List[Dict[str, object]] = []
    stable: List[str] = []
    gained: List[str] = []
    ambiguous: List[str] = []
    for cid in sorted(old):
        o = old[cid]
        n = new.get(cid)
        if o == AMBIGUOUS or n == AMBIGUOUS:
            ambiguous.append(cid)
        elif o is not None and n is None:
            lost.append(cid)
        elif o is not None and n is not None and o != n:
            drifted.append({"cite_id": cid, "from": list(o), "to": list(n),
                            "delta": n[0] - o[0]})
        elif o == n:
            stable.append(cid)
        elif o is None and n is not None:  # newly resolved vs. the baseline
            gained.append(cid)
        else:  # o is None and n is None: unresolved on both sides
            stable.append(cid)
    return {"lost": lost, "drifted": drifted, "stable": stable, "gained": gained,
            "ambiguous": ambiguous}


def render_manifest(source: str, manifest: Dict[str, Optional[Span]]) -> str:
    """Human/diff-friendly rendering: one JSON line per citation, id-sorted."""
    lines = []
    for cid in sorted(manifest):
        span = manifest[cid]
        if span == AMBIGUOUS:
            rec = {"cite_id": cid, "span": AMBIGUOUS, "text": None}
        else:
            rec = {"cite_id": cid, "span": list(span) if span else None,
                   "text": source[span[0]:span[1]] if span else None}
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


# --- tiny CLI ----------------------------------------------------------------
def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plumbline provenance gate (demo CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="resolve citations against a source file")
    a.add_argument("--source", required=True)
    a.add_argument("--citations", required=True,
                   help="JSON file: {cite_id: quoted_text}")
    a.add_argument("--naive", action="store_true", help="use exact-substring baseline")
    args = p.parse_args(argv)

    with open(args.source, encoding="utf-8") as f:
        source = f.read()
    with open(args.citations, encoding="utf-8") as f:
        citations = json.load(f)
    resolver = naive_resolve if args.naive else resolve
    manifest = build_manifest(source, citations, resolver=resolver)
    print(render_manifest(source, manifest))
    missing = [c for c, s in manifest.items() if s is None]
    ambiguous = [c for c, s in manifest.items() if s == AMBIGUOUS]
    if ambiguous:
        print(f"AMBIGUOUS (quote occurs more than once in the source): {ambiguous}",
              file=sys.stderr)
    if missing:
        print(f"UNRESOLVED: {missing}", file=sys.stderr)
    if missing or ambiguous:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
