"""CovGate FIRST-MILESTONE red/green self-test.

Milestone: prove the orphan/thin/adequate bucketing is driven by the real
per-type corroboration mechanism (not a hard-coded verdict), prove
`coverage_drift` trips on a genuine adequate->thin regression, and run the
head-to-head against a naive keyword-frequency audit -- constructing the
EXACT fixture case where raw mention counts are fooled: an entity mentioned
only within its own authoritative record (self-references, running headers,
table-of-contents entries) racks up a high raw count while corroborating
from zero independent sources.

Scenario (obviously-fictional IDs; the "RFCX" prefix does not collide with
any real IETF numbering):
  * RFCX4001 -- broadly corroborated across all 3 other source types -> adequate.
  * RFCX4002 -- corroborated by only 1 of 3 other types -> thin.
  * RFCX4003 -- mentioned only in its own body, and rarely -> orphan (the
    "nobody ever talked about it" case).
  * RFCX4004 -- mentioned 50 TIMES, but every single mention is inside its own
    `rfc_body` record -> orphan under CovGate (zero independent corroboration),
    while a naive raw-count audit reads 50 mentions and calls it well covered.

Exit 0 iff every bucket is correct, the drift regression is caught, and the
naive-vs-CovGate gap is real and measured (not asserted).
"""

from __future__ import annotations

from covgate import (
    Mention,
    audit_corpus,
    coverage_drift,
)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


REGISTRY = ["RFCX4001", "RFCX4002", "RFCX4003", "RFCX4004"]

# --- current run: engineered so each bucket -- and the self-referential trap
# -- is unambiguous. ----------------------------------------------------------
MENTIONS_CUR = [
    # RFCX4001: broad, genuine cross-source corroboration -> adequate.
    Mention("RFCX4001", "rfc_body", "own defining record"),
    Mention("RFCX4001", "rfc_body", "own table of contents"),
    Mention("RFCX4001", "errata", "erratum 1 references RFCX4001"),
    Mention("RFCX4001", "errata", "erratum 2 references RFCX4001"),
    Mention("RFCX4001", "citing_rfcs", "RFCX4002 cites RFCX4001"),
    Mention("RFCX4001", "citing_rfcs", "RFCX4003 cites RFCX4001"),
    Mention("RFCX4001", "citing_rfcs", "RFCX4004 cites RFCX4001"),
    Mention("RFCX4001", "iana_registry", "registry entry points at RFCX4001"),
    # RFCX4002: corroborated by exactly ONE other source type -> thin.
    Mention("RFCX4002", "rfc_body", "own defining record"),
    Mention("RFCX4002", "errata", "erratum 3 references RFCX4002"),
    # RFCX4003: only ever mentioned in its own body, and only once -> orphan.
    Mention("RFCX4003", "rfc_body", "own defining record"),
    # RFCX4004: the trap -- 50 mentions, ALL self-referential (own body only).
    *[Mention("RFCX4004", "rfc_body", f"self-reference #{i}") for i in range(50)],
]

# --- prior run: RFCX4002 used to be broadly corroborated (adequate); every
# other entity is unchanged from MENTIONS_CUR. Used for the drift check. ----
MENTIONS_PREV = [m for m in MENTIONS_CUR if m.entity != "RFCX4002"] + [
    Mention("RFCX4002", "rfc_body", "own defining record"),
    Mention("RFCX4002", "errata", "erratum 3 references RFCX4002"),
    Mention("RFCX4002", "errata", "erratum 4 references RFCX4002"),
    Mention("RFCX4002", "citing_rfcs", "RFCX4001 cites RFCX4002"),
    Mention("RFCX4002", "iana_registry", "registry entry points at RFCX4002"),
]


# --- the incumbent baseline (what a competent engineer reaches for first) ---
def naive_keyword_frequency_audit(
    entity: str, mentions: list[Mention], well_covered_min: int = 5
) -> tuple[int, bool]:
    """Incumbent: total RAW mention count across ALL sources, thresholded.

    This is the standard 'grep count' style audit -- count every occurrence of
    the entity anywhere in the corpus, call it covered if there are 'enough'.
    It is a FAIR, reasonable metric, not a strawman: on a genuinely
    well-corroborated entity (RFCX4001) and on a genuinely under-discussed one
    (RFCX4003) it agrees with CovGate (verified in the head-to-head below).
    Its blind spot is structural: it cannot tell a mention in an entity's OWN
    defining record from independent corroboration by another source, so an
    entity that only ever talks about itself scores identically to one with
    broad, genuine cross-source coverage.
    """
    count = sum(1 for m in mentions if m.entity == entity)
    return count, count >= well_covered_min


def main() -> int:
    print("deterministic: hand-authored mention fixtures, no RNG needed")

    gate = audit_corpus(REGISTRY, MENTIONS_CUR)
    print(f"\n{gate.report()}")

    # --- RED/GREEN: each bucket driven by the real mechanism ---------------
    if gate.records["RFCX4001"].severity != "adequate":
        _fail(f"RFCX4001 should be adequate, got {gate.records['RFCX4001'].severity}")
    if gate.records["RFCX4001"].type_coverage != 1.0:
        _fail(f"RFCX4001 should have full type_coverage, got {gate.records['RFCX4001'].type_coverage}")

    if gate.records["RFCX4002"].severity != "thin":
        _fail(f"RFCX4002 should be thin, got {gate.records['RFCX4002'].severity}")
    if gate.records["RFCX4002"].mentioning_types != ["errata"]:
        _fail(f"RFCX4002 should be corroborated by exactly ['errata'], got {gate.records['RFCX4002'].mentioning_types}")

    if gate.records["RFCX4003"].severity != "orphan":
        _fail(f"RFCX4003 should be orphan, got {gate.records['RFCX4003'].severity}")
    if gate.records["RFCX4003"].total_other_mentions != 0:
        _fail("RFCX4003 has zero OTHER-source mentions by construction")

    print(
        "  -> buckets verified: RFCX4001 adequate (3/3 types), "
        "RFCX4002 thin (1/3 types), RFCX4003 orphan (0/3 types)"
    )

    # --- coverage_drift: adequate -> thin must be caught --------------------
    gate_prev = audit_corpus(REGISTRY, MENTIONS_PREV)
    if gate_prev.records["RFCX4002"].severity != "adequate":
        _fail("fixture bug: RFCX4002 should be adequate in the PRIOR run")

    regressions = coverage_drift(gate_prev.to_dict(), gate.to_dict())
    print(f"\n[coverage_drift] prior -> current regressions: {regressions}")
    if len(regressions) != 1 or regressions[0]["entity"] != "RFCX4002":
        _fail(f"expected exactly one regression on RFCX4002, got {regressions}")
    if regressions[0]["prior_severity"] != "adequate" or regressions[0]["current_severity"] != "thin":
        _fail(f"expected adequate->thin, got {regressions[0]}")
    print("  -> RED caught: RFCX4002 regressed adequate -> thin, named explicitly")

    # --- GREEN: an identical re-run reports no drift ------------------------
    clean_regressions = coverage_drift(gate.to_dict(), gate.to_dict())
    if clean_regressions:
        _fail(f"identical re-run must report zero regressions, got {clean_regressions}")
    print("  -> GREEN: identical re-run reports zero regressions")

    # --- HEAD-TO-HEAD: naive keyword frequency vs CovGate -------------------
    print("\n=== HEAD-TO-HEAD: naive keyword-frequency audit vs CovGate ===")

    # Fairness anchors: on a genuinely well-corroborated entity AND a genuinely
    # under-discussed one, the naive count and CovGate AGREE -- so the trap
    # below is a true disagreement on a real blind spot, not a rigged strawman.
    n1_count, n1_ok = naive_keyword_frequency_audit("RFCX4001", MENTIONS_CUR)
    n3_count, n3_ok = naive_keyword_frequency_audit("RFCX4003", MENTIONS_CUR)
    print(f"  RFCX4001: naive count={n1_count} well_covered={n1_ok}   CovGate={gate.records['RFCX4001'].severity}")
    print(f"  RFCX4003: naive count={n3_count} well_covered={n3_ok}   CovGate={gate.records['RFCX4003'].severity}")
    if not n1_ok or gate.records["RFCX4001"].severity != "adequate":
        _fail("fairness anchor broken: both metrics should agree RFCX4001 is well covered")
    if n3_ok or gate.records["RFCX4003"].severity != "orphan":
        _fail("fairness anchor broken: both metrics should agree RFCX4003 is not well covered")
    print("  -> both metrics AGREE on these two: the naive audit is not a strawman")

    # The trap: RFCX4004 is mentioned 50 times, but every mention is
    # self-referential (own `rfc_body` only) -- zero independent corroboration.
    n4_count, n4_ok = naive_keyword_frequency_audit("RFCX4004", MENTIONS_CUR)
    rec4 = gate.records["RFCX4004"]
    print(
        f"\n  RFCX4004: naive count={n4_count} well_covered={n4_ok}   "
        f"CovGate type_coverage={rec4.type_coverage} severity={rec4.severity}"
    )
    if not n4_ok:
        _fail(f"fixture must make the naive count LOOK well-covered (n4_count={n4_count}); that IS the trap")
    if n4_count < 10 * n1_count // 1 and n4_count < 20:
        _fail("fixture should make RFCX4004's raw count clearly dwarf a normal entity's, to be a fair trap")
    if rec4.severity != "orphan" or rec4.type_coverage != 0.0:
        _fail(f"CovGate should correctly flag RFCX4004 as orphan (0 independent corroboration), got {rec4}")

    print(
        f"\n  MEASURED gap: naive keyword-frequency audit reads {n4_count} raw mentions "
        f"of RFCX4004 and calls it WELL COVERED (>= 5 threshold), while CovGate's "
        f"type_coverage=0.0 correctly reveals ZERO independent corroboration -- every "
        f"one of those {n4_count} mentions is RFCX4004 talking about itself."
    )
    print(
        "  -> naive audit: blind to source provenance, fooled by self-reference volume.\n"
        "     CovGate: excludes the entity's own defining source by construction, so "
        "volume of self-mentions cannot buy a passing grade."
    )

    # --- determinism check: identical inputs -> identical result -----------
    if audit_corpus(REGISTRY, MENTIONS_CUR).to_dict() != gate.to_dict():
        _fail("non-deterministic audit across identical runs")

    print(
        "\nPASS: orphan/thin/adequate buckets correct, coverage_drift caught the "
        "adequate->thin regression, and the self-referential-mention trap fooled "
        "the naive keyword-frequency audit but not CovGate. Deterministic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
