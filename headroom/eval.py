"""RED/GREEN self-test for Headroom: the hardware-aware RAG budget governor.

FIRST MILESTONE: prove the governor gates on *measured hardware headroom*.

  RED:   On a card-a trajectory that heats toward the 90C abort line and the
         15000MB VRAM wedge, an UNGOVERNED agentic loop escalates a 3rd hop
         and physically BREACHES the limit (this is the real fault). The SAME
         trajectory GOVERNED by Headroom is stopped BEFORE the breach: the
         decision comes from the real mechanism (measured slope extrapolation
         over telemetry), and the logged reason cites the predicted telemetry.

  GREEN: On an ample-headroom trajectory, Headroom allows every hop with zero
         DENY/DEFER (no false positives) and the loop completes normally.

Determinism: numpy.random.default_rng(SEED); no wall-clock, no random.
exit 0 iff both RED and GREEN pass. No hard-coded verdicts.
"""

from __future__ import annotations

import sys

from baselines import TOKEN_BUDGET, TOKENS_PER_HOP, CostGovernor
from headroom import CARD_A, Decision, Headroom, Trajectory
from loop import run_loop

# ---- SIMULATED telemetry schedules (post-hop: vram_mb, temp_c, hop_ms) -------
# card-a climbing toward the wall. Thermal + VRAM both accelerate; hop 3
# crosses 90C AND the 15000MB ceiling. (Clearly simulated; see README.)
NEAR_CEILING = [
    (11472.0, 82.0, 800.0),   # hop 1
    (13072.0, 86.0, 850.0),   # hop 2
    (15272.0, 91.5, 900.0),   # hop 3  <-- breaches both limits
]

# Ample headroom: cool, low VRAM, fast. Nothing should ever gate.
AMPLE = [
    (4472.0, 65.0, 600.0),   # hop 1
    (4972.0, 66.0, 620.0),   # hop 2
    (5472.0, 67.0, 640.0),   # hop 3
    (5972.0, 68.0, 660.0),   # hop 4
]

# ---- HEAD-TO-HEAD scenario set (card-a) --------------------------------------
# A skeptic dismisses "governed vs ungoverned" as a strawman. The real proof is
# vs a REASONABLE incumbent: a token-cost / sufficiency governor of the
# Adaptive-RAG / CA-RAG family (see baselines.CostGovernor). Every DANGEROUS
# trajectory below wedges the card at hop 3-4; a competent cost governor, whose
# token budget and sufficiency signal are NOWHERE near binding at that point,
# sails straight into the wall, because it is blind to VRAM/thermal physics.

# vram breaches first, temperature stays moderate.
VRAM_LED = [
    (11472.0, 77.0, 800.0),   # hop 1
    (13272.0, 79.0, 820.0),   # hop 2
    (15172.0, 81.0, 840.0),   # hop 3  <-- vram 15172 >= 15000 ceiling
]

# temperature breaches first, vram stays moderate.
THERMAL_LED = [
    (10472.0, 81.0, 800.0),   # hop 1
    (11272.0, 86.0, 820.0),   # hop 2
    (11972.0, 91.5, 840.0),   # hop 3  <-- 91.5C >= 90C abort
]

# a slower 4-hop climb that breaches on the 4th.
SLOW_CLIMB = [
    (11472.0, 80.0, 780.0),   # hop 1
    (12472.0, 82.6, 800.0),   # hop 2
    (13472.0, 86.0, 820.0),   # hop 3
    (15072.0, 90.5, 840.0),   # hop 4  <-- vram AND temp breach
]

DANGEROUS = {
    "thermal+vram near-ceiling": NEAR_CEILING,
    "vram-led wedge":            VRAM_LED,
    "thermal-led abort":         THERMAL_LED,
    "slow 4-hop climb":          SLOW_CLIMB,
}

# steady low cruise: another clean case (no gate from either governor).
STEADY = [
    (4672.0, 64.0, 590.0),
    (4772.0, 64.5, 595.0),
    (4872.0, 65.0, 600.0),
    (4972.0, 65.5, 605.0),
    (5072.0, 66.0, 610.0),
]
SAFE = {
    "ample headroom": AMPLE,
    "steady cruise":  STEADY,
}

# A long, perfectly SAFE climb-free schedule: the card never gets near a limit,
# so Headroom allows every hop, but the incumbent's TOKEN BUDGET legitimately
# binds first. Proves the incumbent is a real, working governor (not a no-op).
# Fast, cool, low-VRAM hops: safe on ALL of Headroom's signals (vram, thermal,
# AND latency: 25 x 200ms = 5000ms < the 6000ms deadline) so Headroom allows
# every hop, while the incumbent's token budget binds around hop 19.
LONG_AMPLE = [(5472.0 + i * 4.0, 67.0, 200.0) for i in range(25)]


# A baseline (hop 0) sample that already breaches the thermal static guard, so
# the governor DENYs on the very first gate() call, before any hop executes.
HOP1_DENY = [(12472.0, 97.0, 700.0)]


def _p(msg):
    print(msg, flush=True)


def red() -> bool:
    _p("=== RED: card-a climbing to the abort line ===")

    # 1) UNGOVERNED baseline: must actually breach (proves the fault is real).
    ung = run_loop(Trajectory(NEAR_CEILING), CARD_A, governor=None)
    _p(f"  ungoverned: hops={ung.hops_executed} breached={ung.breached} "
       f"peak_temp={ung.peak_temp_c:.1f}C peak_vram={ung.peak_vram_mb:.0f}MB "
       f"stop={ung.stopped_by}")
    _p(f"    breach_reason: {ung.breach_reason or '(none)'}")
    if not ung.breached:
        _p("  FAIL: ungoverned loop did not breach -- fault not reproduced.")
        return False

    # 2) GOVERNED: same trajectory, Headroom must stop it BEFORE the breach.
    gov = Headroom(CARD_A)
    grn = run_loop(Trajectory(NEAR_CEILING), CARD_A, governor=gov)
    _p(f"  governed:   hops={grn.hops_executed} breached={grn.breached} "
       f"peak_temp={grn.peak_temp_c:.1f}C peak_vram={grn.peak_vram_mb:.0f}MB "
       f"stop={grn.stopped_by}")

    # the governor's own audit log for the gate that stopped it
    gating = [g for g in gov.log if g.decision in (Decision.DENY, Decision.DEFER)]
    for g in gating:
        _p(f"    gate -> {g.decision.value.upper()}: {g.reason}")
        _p(f"      observed temp={g.observed_temp_c:.1f}C vram={g.observed_vram_mb:.0f}MB"
           f" | predicted next temp={g.pred_temp_c:.1f}C vram={g.pred_vram_mb:.0f}MB")

    ok = True
    if grn.breached:
        _p("  FAIL: governed loop still breached -- governor did not protect.")
        ok = False
    if grn.stopped_by != "governor":
        _p(f"  FAIL: governed loop stopped by '{grn.stopped_by}', not the governor.")
        ok = False
    if grn.hops_executed >= ung.hops_executed:
        _p("  FAIL: governor did not cut the loop short of the ungoverned run.")
        ok = False
    if not gating:
        _p("  FAIL: governor logged no DENY/DEFER -- no real gate fired.")
        ok = False
    _p(f"  RED result: {'PASS' if ok else 'FAIL'} "
       f"(fault reproduced, then caught by measured-telemetry gate)")
    return ok


def green() -> bool:
    _p("\n=== GREEN: ample headroom -- must NOT over-block ===")
    gov = Headroom(CARD_A)
    res = run_loop(Trajectory(AMPLE), CARD_A, governor=gov)
    denies = [g for g in gov.log if g.decision != Decision.ALLOW]
    _p(f"  governed: hops={res.hops_executed} breached={res.breached} "
       f"peak_temp={res.peak_temp_c:.1f}C peak_vram={res.peak_vram_mb:.0f}MB "
       f"stop={res.stopped_by} non-allow-gates={len(denies)}")

    ok = True
    if res.breached:
        _p("  FAIL: breached on an ample profile (impossible -- schedule bug).")
        ok = False
    if res.hops_executed != len(AMPLE):
        _p(f"  FAIL: only {res.hops_executed}/{len(AMPLE)} hops -- false gating.")
        ok = False
    if denies:
        _p(f"  FAIL: {len(denies)} false DENY/DEFER on ample headroom.")
        ok = False
    _p(f"  GREEN result: {'PASS' if ok else 'FAIL'} (all hops allowed, no breach)")
    return ok


def hop1_deny() -> bool:
    """P1-1 regression: a governor DENY on the very first gate() call (before
    any hop executes) must report the REAL observed baseline peak telemetry,
    not an impossible 0.0/0.0."""
    _p("\n=== HOP-1 DENY: governor denies before any hop executes ===")
    gov = Headroom(CARD_A)
    res = run_loop(Trajectory(HOP1_DENY), CARD_A, governor=gov)
    _p(f"  hops={res.hops_executed} stopped_by={res.stopped_by} "
       f"peak_temp={res.peak_temp_c:.1f}C peak_vram={res.peak_vram_mb:.0f}MB")

    ok = True
    if res.hops_executed != 0 or res.stopped_by != "governor":
        _p("  FAIL: expected an immediate hop-1 governor deny -- fixture bug.")
        ok = False
    if not gov.log or gov.log[0].decision != Decision.DENY:
        _p("  FAIL: governor did not log a DENY on the first gate() call.")
        ok = False
    else:
        baseline = gov.log[0]
        _p(f"    baseline observed: temp={baseline.observed_temp_c:.1f}C "
           f"vram={baseline.observed_vram_mb:.0f}MB")
        if (res.peak_temp_c != baseline.observed_temp_c
                or res.peak_vram_mb != baseline.observed_vram_mb):
            _p(f"  FAIL: reported peak {res.peak_temp_c:.1f}C/"
               f"{res.peak_vram_mb:.0f}MB != real observed baseline "
               f"{baseline.observed_temp_c:.1f}C/{baseline.observed_vram_mb:.0f}MB "
               "(the impossible-0.0-peak bug).")
            ok = False
    if res.peak_temp_c == 0.0 and res.peak_vram_mb == 0.0:
        _p("  FAIL: peak telemetry reported impossible 0.0/0.0 despite a real "
           "baseline sample.")
        ok = False
    _p(f"  HOP-1 DENY result: {'PASS' if ok else 'FAIL'} "
       f"(peak reflects the observed baseline, not a silent 0.0)")
    return ok


def _new_incumbent(*, sufficient_after=None):
    """A FRESH incumbent with the SAME fixed config for every scenario.

    Fair-baseline discipline: one configuration across the whole set. Only the
    scenario (trajectory + query difficulty) changes. Config is defensible, not
    tuned to lose: TOKEN_BUDGET is a common context window and TOKENS_PER_HOP is
    DERIVED from the real corpus (see baselines.py). `sufficient_after` is a
    property of the QUERY (how many hops until the answer is good enough), not a
    handicap.
    """
    return CostGovernor(token_budget=TOKEN_BUDGET,
                        tokens_per_hop=TOKENS_PER_HOP,
                        sufficient_after=sufficient_after)


def _fairness_gate() -> bool:
    """PROVE the incumbent is a real governor, not a rigged no-op.

    If the comparator never stops, the whole A/B is a strawman. So first show it
    DOES stop, for its OWN legitimate reasons, when those signals bind.
    """
    _p("\n=== FAIR-BASELINE CHECK: the incumbent is a working governor ===")
    ok = True

    # (a) sufficiency binds: the answer is good enough after 2 hops -> stop.
    inc = _new_incumbent(sufficient_after=2)
    res = run_loop(Trajectory(AMPLE), CARD_A, governor=inc)
    stop_reason = inc.log[-1][1] if inc.log else "(none)"
    _p(f"  sufficiency case: incumbent stopped after {res.hops_executed} hops "
       f"(stop={res.stopped_by}) reason='{stop_reason}'")
    if res.hops_executed != 2 or res.stopped_by != "governor" or res.breached:
        _p("  FAIL: incumbent did not stop on a satisfied answer -- not a real governor.")
        ok = False

    # (b) token budget binds: a long, hardware-SAFE run must still stop on cost.
    inc = _new_incumbent()  # answer never sufficient
    res = run_loop(Trajectory(LONG_AMPLE), CARD_A, governor=inc)
    tok = res.hops_executed * TOKENS_PER_HOP
    stop_reason = inc.log[-1][1] if inc.log else "(none)"
    _p(f"  token-budget case: incumbent stopped after {res.hops_executed}/"
       f"{len(LONG_AMPLE)} hops (~{tok} tok vs {TOKEN_BUDGET} budget) "
       f"stop={res.stopped_by} reason='{stop_reason}'")
    if res.stopped_by != "governor" or res.breached or res.hops_executed >= len(LONG_AMPLE):
        _p("  FAIL: incumbent's token budget never bound -- comparator is a no-op.")
        ok = False
    # ...and Headroom would have allowed that whole safe run (complementary).
    hr = Headroom(CARD_A)
    hr_res = run_loop(Trajectory(LONG_AMPLE), CARD_A, governor=hr)
    if hr_res.hops_executed != len(LONG_AMPLE) or hr_res.breached:
        _p("  FAIL: Headroom over-blocked the long SAFE run.")
        ok = False
    _p(f"  (Headroom allowed all {hr_res.hops_executed}/{len(LONG_AMPLE)} safe hops "
       f"-- economics vs physics are complementary)")

    _p(f"  FAIR-BASELINE result: {'PASS' if ok else 'FAIL'} "
       f"(incumbent stops on sufficiency AND on token budget)")
    return ok


def head_to_head() -> bool:
    """The money demo: incumbent cost governor vs Headroom on the SAME cards.

    MEASURED, not asserted. For every dangerous trajectory we run the identical
    telemetry under (1) the token/sufficiency incumbent and (2) Headroom, and
    count physical breaches. The wedge is real iff the incumbent breaches where
    Headroom holds.
    """
    _p("\n=== HEAD-TO-HEAD: token/sufficiency incumbent vs Headroom ===")
    n = len(DANGEROUS)
    inc_breaches = 0
    hr_breaches = 0
    for label, sched in DANGEROUS.items():
        inc = _new_incumbent()  # hard query: never sufficient within the run
        r_inc = run_loop(Trajectory(sched), CARD_A, governor=inc)
        hr = Headroom(CARD_A)
        r_hr = run_loop(Trajectory(sched), CARD_A, governor=hr)
        inc_breaches += int(r_inc.breached)
        hr_breaches += int(r_hr.breached)
        tok_at_stop = r_inc.hops_executed * TOKENS_PER_HOP
        _p(f"  [{label}]")
        _p(f"    incumbent: hops={r_inc.hops_executed} breached={r_inc.breached} "
           f"peak_temp={r_inc.peak_temp_c:.1f}C peak_vram={r_inc.peak_vram_mb:.0f}MB "
           f"-> {r_inc.breach_reason or 'ok'}")
        _p(f"      (spent ~{tok_at_stop}/{TOKEN_BUDGET} tok, answer still "
           f"insufficient -- its economics said 'keep going')")
        _p(f"    headroom : hops={r_hr.hops_executed} breached={r_hr.breached} "
           f"peak_temp={r_hr.peak_temp_c:.1f}C peak_vram={r_hr.peak_vram_mb:.0f}MB "
           f"stop={r_hr.stopped_by}")

    # SAFE scenarios: neither governor may over-block (no false stops).
    over_block = False
    for label, sched in SAFE.items():
        inc = _new_incumbent()
        r_inc = run_loop(Trajectory(sched), CARD_A, governor=inc)
        hr = Headroom(CARD_A)
        r_hr = run_loop(Trajectory(sched), CARD_A, governor=hr)
        if r_inc.hops_executed != len(sched) or r_hr.hops_executed != len(sched):
            over_block = True
            _p(f"  [{label}] FAIL: false gating (inc={r_inc.hops_executed} "
               f"hr={r_hr.hops_executed} of {len(sched)})")
        else:
            _p(f"  [{label}] both completed all {len(sched)} hops (no over-block)")

    inc_rate = inc_breaches / n
    hr_rate = hr_breaches / n
    gap = inc_rate - hr_rate
    _p("\n  --- MEASURED wedge over the dangerous scenario set ---")
    _p(f"    incumbent (token+sufficiency) breach rate: {inc_breaches}/{n} = {inc_rate:.0%}")
    _p(f"    Headroom (hardware-physics)    breach rate: {hr_breaches}/{n} = {hr_rate:.0%}")
    _p(f"    MEASURED GAP (breaches prevented): {gap:.0%}")

    # ASSERT the gap: the bar is that incumbent must FAIL where Headroom holds.
    ok = True
    if inc_breaches != n:
        _p("  FAIL: incumbent did not breach on every dangerous scenario "
           "-- wedge not demonstrated.")
        ok = False
    if hr_breaches != 0:
        _p("  FAIL: Headroom breached on a dangerous scenario -- no protection.")
        ok = False
    if over_block:
        _p("  FAIL: a governor over-blocked a safe scenario.")
        ok = False
    if not gap > 0.0:
        _p("  FAIL: no measured advantage over the incumbent.")
        ok = False
    _p(f"  HEAD-TO-HEAD result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    r = red()
    g = green()
    d = hop1_deny()
    f = _fairness_gate()
    h = head_to_head()
    _p("\n" + "=" * 52)
    _p(f"RED (catch injected fault):    {'PASS' if r else 'FAIL'}")
    _p(f"GREEN (clean case):            {'PASS' if g else 'FAIL'}")
    _p(f"HOP-1 DENY (real peak telemetry): {'PASS' if d else 'FAIL'}")
    _p(f"FAIR-BASELINE (real incumbent):{'PASS' if f else 'FAIL'}")
    _p(f"HEAD-TO-HEAD (measured wedge):  {'PASS' if h else 'FAIL'}")
    ok = r and g and d and f and h
    _p(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
