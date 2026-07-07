# -*- coding: utf-8 -*-
"""Plumbline FIRST-MILESTONE red/green self-test.

Milestone: prove the provenance gate CATCHES real provenance faults across a
reindex via the REAL alignment mechanism (RED), while PASSING the clean case
that a naive exact-substring baseline FALSE-FAILS on (GREEN). No rigging: every
verdict is computed by the mechanism; ground truth is derived from how the
fixture was mutated, not hard-coded into the tool.

Scenario:
  * A clean source document (source_v1).
  * Citations whose stored text is OCR/whitespace-normalized copies of source
    spans (simulating an OCR/extraction index) -- clearly synthetic, deterministic.
  * GREEN wedge: Plumbline resolves ALL citations against source_v1; the naive
    exact-substring baseline FALSE-FAILS on the normalization (>0 false misses).
  * RED reindex: source_v2 inserts a paragraph up top (shifts every downstream
    offset -> DRIFT) and deletes one cited region (-> coverage LOST). Plumbline's
    coverage_diff reports exactly the lost citation + the drifted survivors with
    their destination spans. If it fails to catch this, eval exits nonzero.
  * GREEN control: reindex to an IDENTICAL source -> zero LOST, zero DRIFT (no
    false alarms).

Deterministic: numpy default_rng(SEED) drives the (bounded) OCR degradation; no
wall-clock, no uncontrolled randomness.

Exit 0 iff the RED fault is genuinely caught AND both GREEN checks are clean.
"""

from __future__ import annotations

import sys

import numpy as np

from plumbline import (
    AMBIGUOUS,
    build_manifest,
    coverage_diff,
    naive_resolve,
    render_manifest,
    resolve,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # ligature char in fixtures
except Exception:
    pass

SEED = 20260704
_rng = np.random.default_rng(SEED)

# --- clean source document (ground truth) ------------------------------------
SOURCE_V1 = (
    "Annual Financial Review\n\n"
    "The board confirmed that revenue rose to 1200 units in the northern region.\n\n"
    "Operating margin held at a stable level despite inflationary pressure.\n\n"
    "The audit committee flagged one reconciliation gap in the ledger.\n\n"
    "Cash reserves cover roughly eleven months of operating expenditure.\n\n"
    "Management reaffirmed its outlook for the following fiscal year.\n"
)

# Citations = exact spans a RAG answer cited (ground-truth text lives in SOURCE_V1).
_CITE_TEXT = {
    "C1": "The board confirmed that revenue rose to 1200 units in the northern region.",
    "C2": "Operating margin held at a stable level despite inflationary pressure.",
    "C3": "The audit committee flagged one reconciliation gap in the ledger.",
    "C4": "Cash reserves cover roughly eleven months of operating expenditure.",
    "C5": "Management reaffirmed its outlook for the following fiscal year.",
}


def _ocr_degrade(text: str, rng: np.random.Generator) -> str:
    """Deterministically apply OCR/whitespace normalization to a citation.

    Uses ONLY transforms that canonicalize back to the source (o<->0, l<->1,
    fi->ligature, whitespace doubling), so a legitimate chunk is preserved under
    Plumbline's canonicalization but broken under exact substring match. Always
    applies >=1 change so the naive baseline genuinely false-fails.
    """
    out = list(text)
    changed = 0
    # o -> 0 : replace a bounded number of 'o's chosen deterministically
    k = int(rng.integers(1, 3))
    for idx, ch in enumerate(out):
        if ch == "o" and changed < k:
            out[idx] = "0"
            changed += 1
    s = "".join(out)
    # l -> 1 on the first lowercase 'l'
    s = s.replace("l", "1", 1)
    # fi ligature
    s = s.replace("fi", "ﬁ", 1)
    # double one internal space
    sp = s.find(" ", 1)
    if sp != -1:
        s = s[:sp] + "  " + s[sp + 1:]
    if s == text:  # guarantee a real change for the baseline contrast
        s = s.replace("e", "e ", 1)
    return s


def _build_citations() -> dict:
    return {cid: _ocr_degrade(txt, _rng) for cid, txt in sorted(_CITE_TEXT.items())}


_INSERTED = "Preliminary Note: figures are unaudited pending sign-off.\n\n"
_C3_REPLACEMENT = "This section has been withdrawn for review."


def _reindex_source_drift_and_loss() -> str:
    """source_v2: insert a header paragraph (shifts all downstream offsets) and
    delete the C3-cited region (coverage loss)."""
    s2 = _INSERTED + SOURCE_V1
    c3 = _CITE_TEXT["C3"]
    assert c3 in s2
    s2 = s2.replace(c3, _C3_REPLACEMENT)
    return s2


def _expected_deltas() -> dict:
    """Ground-truth per-citation start-offset drift, derived from the mutation.

    Every citation shifts by the inserted header length; citations located AFTER
    the deleted C3 region shift by an additional (replacement_len - c3_len)."""
    ins = len(_INSERTED)
    net_c3 = len(_C3_REPLACEMENT) - len(_CITE_TEXT["C3"])
    c3_start = SOURCE_V1.index(_CITE_TEXT["C3"])
    exp = {}
    for cid, txt in _CITE_TEXT.items():
        if cid == "C3":
            continue
        after_c3 = SOURCE_V1.index(txt) > c3_start
        exp[cid] = ins + (net_c3 if after_c3 else 0)
    return exp


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


_REPEATED_QUOTE_SOURCE = (
    "Revenue increased. The board noted strong demand in the northern region.\n\n"
    "Revenue increased. The board noted continued momentum into the next quarter.\n"
)
_REPEATED_QUOTE = "Revenue increased."


def _check_ambiguous_repeated_quote() -> bool:
    """P0-1 regression: a quote that legitimately occurs MORE THAN ONCE in the
    source must surface as AMBIGUOUS, not silently resolve to the first
    occurrence (which may be the wrong one and mask real drift)."""
    print("\n--- ambiguity check: a quote repeated in the source ---")
    got = resolve(_REPEATED_QUOTE_SOURCE, _REPEATED_QUOTE)
    got_naive = naive_resolve(_REPEATED_QUOTE_SOURCE, _REPEATED_QUOTE)
    print(f"resolve()       -> {got!r}")
    print(f"naive_resolve() -> {got_naive!r}")
    if got != AMBIGUOUS:
        print(f"FAIL: repeated quote did not surface AMBIGUOUS, got {got!r} "
              "-- would silently mis-resolve to the first occurrence.")
        return False
    if got_naive != AMBIGUOUS:
        print(f"FAIL: naive_resolve did not surface AMBIGUOUS on a repeated "
              f"quote, got {got_naive!r}.")
        return False
    print("GREEN: repeated quote surfaces AMBIGUOUS, not a silent first-match "
          "guess.")
    return True


def main() -> int:
    if not _check_ambiguous_repeated_quote():
        _fail("ambiguity check failed")

    citations = _build_citations()

    print("=" * 70)
    print("Plumbline milestone self-test")
    print("=" * 70)

    # --- GREEN wedge: Plumbline resolves all; naive exact-substring false-fails
    pm_v1 = build_manifest(SOURCE_V1, citations, resolver=resolve)
    nm_v1 = build_manifest(SOURCE_V1, citations, resolver=naive_resolve)

    plumb_miss = [c for c, s in pm_v1.items() if s is None]
    naive_miss = [c for c, s in nm_v1.items() if s is None]
    print(f"\n[wedge] Plumbline unresolved on clean OCR corpus: {plumb_miss}")
    print(f"[wedge] Naive exact-substring unresolved (FALSE fails): {naive_miss}")

    if plumb_miss:
        _fail(f"Plumbline failed to resolve legitimate OCR citations: {plumb_miss}")
    if not naive_miss:
        _fail("naive baseline did not false-fail -- fixture not exercising the wedge")

    # Provenance manifest must point at the TRUE source text for every citation.
    for cid, span in pm_v1.items():
        got = SOURCE_V1[span[0]:span[1]]
        if got != _CITE_TEXT[cid]:
            _fail(f"{cid} resolved to wrong span: {got!r} != {_CITE_TEXT[cid]!r}")
    print("[wedge] GREEN: all citations trace to their true source spans.")

    print("\n--- provenance manifest (source_v1) ---")
    print(render_manifest(SOURCE_V1, pm_v1))

    # --- RED: reindex introduces real drift + one coverage loss ---------------
    source_v2 = _reindex_source_drift_and_loss()
    pm_v2 = build_manifest(source_v2, citations, resolver=resolve)
    diff = coverage_diff(pm_v1, pm_v2)

    lost = set(diff["lost"])
    drifted = {d["cite_id"] for d in diff["drifted"]}
    stable = set(diff["stable"])

    print("\n--- coverage-loss diff (v1 -> v2 reindex) ---")
    print(f"LOST    : {sorted(lost)}")
    print(f"DRIFTED : {sorted(drifted)}")
    print(f"STABLE  : {sorted(stable)}")
    for d in diff["drifted"]:
        print(f"   {d['cite_id']}: {d['from']} -> {d['to']} (delta {d['delta']:+d})")

    # Ground truth derived from the mutation: C3 deleted -> LOST; every other
    # citation shifted by the inserted header -> DRIFTED; none STABLE.
    expected_lost = {"C3"}
    expected_drift = set(_CITE_TEXT) - expected_lost
    if lost != expected_lost:
        _fail(f"coverage-loss miss: expected LOST {expected_lost}, got {lost}")
    if drifted != expected_drift:
        _fail(f"drift miss: expected DRIFTED {expected_drift}, got {drifted}")
    if stable:
        _fail(f"unexpected STABLE across a shifting reindex: {stable}")
    # destinations must match the drift the mutation actually induced (the header
    # shifts all; the C3 deletion additionally shifts everything after it).
    expected_deltas = _expected_deltas()
    for d in diff["drifted"]:
        exp = expected_deltas[d["cite_id"]]
        if d["delta"] != exp:
            _fail(f"{d['cite_id']} reported wrong drift delta {d['delta']} != {exp}")
    print("[reindex] RED: fault genuinely caught -- 1 lost, 4 drifted with "
          "destinations.")

    # --- GREEN control: identical reindex -> no false alarms ------------------
    pm_ctrl = build_manifest(SOURCE_V1, citations, resolver=resolve)
    diff_ctrl = coverage_diff(pm_v1, pm_ctrl)
    if diff_ctrl["lost"] or diff_ctrl["drifted"]:
        _fail(f"false alarm on identical reindex: {diff_ctrl}")
    print("[control] GREEN: identical reindex -> 0 lost, 0 drifted (no false "
          "alarms).")

    print("\n" + "=" * 70)
    print("PASS: RED fault caught + GREEN clean case + wedge over naive baseline.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
