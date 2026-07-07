"""Leakprobe -- RED/GREEN self-test (first milestone).

Milestone under test:
  Given a corpus seeded with known PII, rank which spans a synthesized,
  PII-eliciting query actually surfaces in a LIVE index top-k, then show that
  the MINIMAL (reachable-only) redaction set (a) closes those leaks and
  (b) preserves MEASURABLY more retrieval recall than a blanket mask -- while a
  corpus whose PII is not reachable needs no redaction at all.

RED   = the audit CATCHES the injected reachable-PII leak via the real
        retrieval mechanism, and the recall gap (minimal > blanket) is real.
GREEN = the clean corpus (PII present but unreachable) yields an empty
        redaction set and full recall.

Deterministic by construction: there is NO randomness anywhere in the pipeline
(the embedder is a fixed-key BLAKE2b feature hash, retrieval is a stable argsort,
everything else is pure logic). No RNG, no wall-clock, no network. Output is
therefore byte-identical across runs -- and A6 below PROVES it by re-running the
audit and asserting identical results, rather than merely asserting it. exit 0
on pass.
"""

from __future__ import annotations

import sys

from leakprobe import (
    Chunk,
    apply_redaction,
    audit,
    recall_at_k,
)

RETRIEVAL_K = 3  # top-k for reachability
RECALL_K = 1  # recall@1 for the utility measurement

DEBUG = "--debug" in sys.argv


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
# Shared boilerplate footer -- decoys reuse it so a boilerplate-only elicited
# query is drowned by non-PII chunks (that is what makes tail PII UNREACHABLE).
_BOILER = "confidential document prepared on behalf of {who} per standard policy footer"


def fault_corpus() -> tuple[list[Chunk], list[tuple[str, int]]]:
    """A corpus with TWO kinds of PII:

      * chunk 100: an email in DISTINCTIVE context  -> reachable  (a real leak)
      * chunk 200: a name in BOILERPLATE context    -> unreachable (a false-positive
                   for existence-based tools) AND it is the gold doc for a utility
                   query, so blanket-masking it costs recall.
    """
    chunks = [
        Chunk(100,
              "Project Zephyr deployment lead onboarding reach the engineer at "
              "zephyr.lead@corp.example about staging rollout keys"),
        Chunk(200,
              "quarterly revenue finance report summary shows strong margin growth "
              + _BOILER.format(who="jane doe")),
        # boilerplate decoys (no PII) -- these outrank chunk 200 for the
        # boilerplate-heavy name query, making the name unreachable.
        Chunk(300, _BOILER.format(who="the operations group") + " notice alpha"),
        Chunk(301, _BOILER.format(who="the audit group") + " notice bravo"),
        Chunk(302, _BOILER.format(who="the vendor group") + " notice charlie"),
        Chunk(303, _BOILER.format(who="the review group") + " notice delta"),
        # utility competitor for the finance query: shares only 3 utility tokens
        # and is padded, so at baseline chunk 200 wins -- but once chunk 200 is
        # mask-diluted by a blanket redaction, this competitor overtakes it.
        Chunk(400, "quarterly revenue finance overview appendix highlights "
                   "extended annexes tables notes exhibits"),
    ]
    utility = [("quarterly revenue finance report summary", 200)]
    return chunks, utility


def clean_corpus() -> tuple[list[Chunk], list[tuple[str, int]]]:
    """PII present but UNREACHABLE (name buried in boilerplate, drowned by
    decoys). Needs no redaction."""
    chunks = [
        Chunk(500,
              "annual sustainability metrics report overview highlights improvements "
              + _BOILER.format(who="john roe")),
        Chunk(600, _BOILER.format(who="the operations group") + " notice alpha"),
        Chunk(601, _BOILER.format(who="the audit group") + " notice bravo"),
        Chunk(602, _BOILER.format(who="the vendor group") + " notice charlie"),
        Chunk(603, _BOILER.format(who="the review group") + " notice delta"),
    ]
    utility = [("annual sustainability metrics report overview", 500)]
    return chunks, utility


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def _fmt_span(sp) -> str:
    return f"[{sp.kind}:{sp.value!r}@chunk{sp.chunk_id}]"


def _fingerprint(res) -> tuple:
    """Order-stable, hashable serialization of an audit result. Used to PROVE
    the pipeline is deterministic (identical fingerprint on a re-run) instead of
    merely claiming it."""
    return (
        tuple((s.span.key(), s.query, s.reachable, s.rank) for s in res.scores),
        tuple(sp.key() for sp in res.reachable_spans),
        tuple(sp.key() for sp in res.minimal_redaction),
        tuple(sp.key() for sp in res.blanket_redaction),
    )


def run() -> int:
    failures: list[str] = []

    # ---- RED: fault corpus ------------------------------------------------
    chunks, utility = fault_corpus()
    res = audit(chunks, k=RETRIEVAL_K)

    if DEBUG:
        print("=== FAULT audit ===")
        for sc in res.scores:
            print(f"  {_fmt_span(sc.span):40s} reachable={sc.reachable} rank={sc.rank}")
            print(f"      query: {sc.query}")

    all_n = len(res.all_spans)
    reach_n = len(res.reachable_spans)

    # A1: the leak is caught by the real mechanism
    if reach_n < 1:
        failures.append(f"A1: expected >=1 reachable PII span, got {reach_n}")

    # A2: mechanism is honest both ways -- reachable spans are email@100 only,
    #     the boilerplate name@200 is correctly NOT flagged.
    reachable_kinds = sorted((s.kind, s.chunk_id) for s in res.reachable_spans)
    if ("email", 100) not in reachable_kinds:
        failures.append("A2a: email@100 (distinctive context) should be reachable")
    if ("name", 200) in reachable_kinds:
        failures.append("A2b: name@200 (boilerplate) should NOT be reachable (over-report)")

    # A3: minimal redaction is strictly smaller than blanket (not 'redact all')
    min_n = len(res.minimal_redaction)
    blank_n = len(res.blanket_redaction)
    if not (min_n < blank_n):
        failures.append(f"A3: minimal({min_n}) not < blanket({blank_n})")

    # A4: applying minimal redaction closes the leak (re-audit -> 0 reachable)
    redacted_min = apply_redaction(chunks, res.minimal_redaction)
    res_after = audit(redacted_min, k=RETRIEVAL_K)
    # after redaction the reachable secret value is gone; the residual check is
    # that no *remaining* PII span is reachable.
    if res_after.leak_count != 0:
        failures.append(f"A4: minimal redaction left {res_after.leak_count} reachable leak(s)")

    # A5: recall preserved -- minimal beats blanket, measured.
    r_base = recall_at_k(chunks, utility, RECALL_K)
    r_min = recall_at_k(redacted_min, utility, RECALL_K)
    redacted_blanket = apply_redaction(chunks, res.blanket_redaction)
    r_blank = recall_at_k(redacted_blanket, utility, RECALL_K)
    if not (r_min > r_blank):
        failures.append(f"A5a: recall not preserved: minimal={r_min:.3f} !> blanket={r_blank:.3f}")
    if not (r_min >= r_base):
        failures.append(f"A5b: minimal recall {r_min:.3f} regressed vs baseline {r_base:.3f}")

    # A6: determinism PROOF (substantiates the README's "byte-identical across
    #     runs" claim through the real mechanism). Re-run the SAME audit from
    #     scratch; the full result fingerprint must be identical. This is the
    #     red/green case that guards the fix: before, a dead numpy RNG *labeled*
    #     output as seeded while nothing verified reproducibility -- a fake seed
    #     would silently "pass". Now, any non-determinism (or reintroduced
    #     randomness) makes this fail as it should.
    fp1 = _fingerprint(audit(fault_corpus()[0], k=RETRIEVAL_K))
    fp2 = _fingerprint(audit(fault_corpus()[0], k=RETRIEVAL_K))
    deterministic = fp1 == fp2
    if not deterministic:
        failures.append("A6: audit is not deterministic across runs (fingerprint mismatch)")

    # ---- GREEN: clean corpus ---------------------------------------------
    c_chunks, c_util = clean_corpus()
    c_res = audit(c_chunks, k=RETRIEVAL_K)
    if DEBUG:
        print("=== CLEAN audit ===")
        for sc in c_res.scores:
            print(f"  {_fmt_span(sc.span):40s} reachable={sc.reachable} rank={sc.rank}")

    if c_res.leak_count != 0:
        failures.append(f"B1: clean corpus reports {c_res.leak_count} reachable leak(s)")
    if len(c_res.minimal_redaction) != 0:
        failures.append(f"B2: clean corpus recommends {len(c_res.minimal_redaction)} redactions")

    # ---- report -----------------------------------------------------------
    print("Leakprobe self-test (retrievability-ranked PII audit)")
    print(f"  determinism         = pure hashing/logic, no RNG (re-run identical: {deterministic})")
    print(f"  PII spans detected  = {all_n} (fault corpus)")
    print(f"  reachable (leaks)   = {reach_n}  -> {[_fmt_span(s) for s in res.reachable_spans]}")
    print(f"  minimal redaction   = {min_n} span(s)   blanket = {blank_n} span(s)")
    print(f"  recall@{RECALL_K}  baseline={r_base:.3f}  minimal={r_min:.3f}  blanket={r_blank:.3f}")
    print(f"  recall preserved    = +{(r_min - r_blank):.3f} vs blanket mask")
    print(f"  leaks after minimal = {res_after.leak_count}")
    print(f"  clean corpus leaks  = {c_res.leak_count}  redactions = {len(c_res.minimal_redaction)}")

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS (red caught, green clean, recall preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
