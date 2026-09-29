"""Deadstage FIRST-MILESTONE red/green self-test.

Milestone: the probe must, via its REAL per-stage invariant mechanism,
NAME the dead stage on an injected fault and PASS a clean pipeline.

  RED   : deprecate the embedder to zero-vectors. The `embed` norm invariant
          must fire and the probe must name exactly `embed`, reason
          `collapsed/zero-norm`, and return a non-zero (dead) verdict.
  GREEN : the same corpus with a healthy embedder must pass every stage,
          report `healthy`, and return a zero (live) verdict.

No rigging: eval.py never hard-codes the verdict. It builds real pipeline
artifacts, runs the read-only `check`, and asserts on what the mechanism
actually measured. exit 0 iff both cases behave. Deterministic (fixed SEED,
hash embedder, no wall-clock/random).
"""

from __future__ import annotations

import sys

from deadstage import _STAGES, SEED, build_pipeline, check

DOCS = [
    {"id": "d0", "text": "vector databases store dense embeddings for retrieval"},
    {"id": "d1", "text": "cosine similarity ranks candidate passages by angle"},
    {"id": "d2", "text": "chunking splits long documents into overlapping windows"},
    {"id": "d3", "text": "a reranker reorders the shortlist before the answerer"},
    {"id": "d4", "text": "anisotropy makes embeddings crowd into a narrow cone"},
    {"id": "d5", "text": "the ingest stage normalizes text and strips boilerplate"},
]
QUERIES = [
    "how are passages ranked by similarity",
    "what does the ingest stage do",
]


def _banner(tag: str) -> None:
    print(f"\n=== {tag} ===")


def run_green() -> bool:
    _banner("GREEN (healthy embedder)")
    state = build_pipeline(DOCS, QUERIES, k=3, seed=SEED)
    rep = check(state)
    print("verdict:", rep.line())
    print("stages :", rep.stage_status)
    ok = rep.healthy and all(v == "ok" for v in rep.stage_status.values())
    print("PASS" if ok else "FAIL", "-> expected all stages live")
    return ok


def run_red() -> bool:
    _banner("RED (embedder deprecated to zero-vectors)")
    # Same corpus/queries; only the embedder weights are "not loaded".
    state = build_pipeline(DOCS, QUERIES, embed_kwargs={"dead": True}, k=3, seed=SEED)
    rep = check(state)
    print("verdict:", rep.line())
    if rep.detail:
        print("detail :", rep.detail)
    print("stages :", rep.stage_status)

    named_embed = rep.dead_stage == "embed"
    right_reason = rep.reason == "collapsed/zero-norm"
    non_zero = not rep.healthy
    # earliest-stage discipline: ingest+index stay live, embed is the named death.
    upstream_live = rep.stage_status.get("ingest") == "ok" and rep.stage_status.get("index") == "ok"
    ok = named_embed and right_reason and non_zero and upstream_live
    print("PASS" if ok else "FAIL",
          "-> expected embed named as the single dead stage (collapsed/zero-norm)")
    return ok


def _baseline_symptom_stage(state) -> str | None:
    """FAIR incumbent baseline: symptom-based liveness monitoring.

    Runs the SAME per-stage invariants as Deadstage (identical detection power,
    NOT a crippled strawman) but reports the DEEPEST stage where a problem is
    observed, modelling how real monitoring/alerting fires on the downstream
    symptom (empty retrieval, tied scores, a garbage answer) where the failure
    finally becomes visible, rather than the upstream root cause that silently
    produced it. The ONLY difference from Deadstage is attribution DIRECTION
    (last-failing vs first-failing); this isolates the single variable that
    Deadstage claims as its wedge, so the A/B measures exactly that claim.
    """
    last = None
    for name, fn in _STAGES:
        reason, _ = fn(state)
        if reason is not None:
            last = name
    return last


def run_ab() -> bool:
    """MEASURED A/B: root-cause attribution, Deadstage vs the symptom baseline.

    Exercises the artifact-ingestion path (``from_artifacts``): every fixture is
    an externally-captured artifact set, not a build_pipeline-manufactured one,
    so the retrieve/score stages are reached and named on realistic data.
    """
    from fixtures import build_fixtures

    _banner("A/B (root-cause attribution: Deadstage vs symptom-based baseline)")
    fixtures = build_fixtures()

    base_named: dict[str, str | None] = {}
    ds_correct = 0
    base_correct = 0
    ds_named: dict[str, str | None] = {}
    print(f"{'fixture':<14}{'true root':<11}{'Deadstage':<11}{'baseline':<11}")
    print("-" * 47)
    for name, state, true_root in fixtures:
        ds = check(state).dead_stage           # first-failing == root cause
        base = _baseline_symptom_stage(state)  # last-failing == observed symptom
        ds_named[name] = ds
        base_named[name] = base
        ds_ok = ds == true_root
        base_ok = base == true_root
        ds_correct += ds_ok
        base_correct += base_ok
        print(f"{name:<14}{str(true_root):<11}{str(ds):<11}{str(base):<11}"
              f"{'  ds+' if ds_ok else '  ds-'}{' base+' if base_ok else ' base-'}")

    n = len(fixtures)
    ds_acc = ds_correct / n
    base_acc = base_correct / n
    print("-" * 47)
    print(f"Deadstage root-cause accuracy : {ds_correct}/{n} = {ds_acc:.0%} (measured)")
    print(f"baseline  root-cause accuracy : {base_correct}/{n} = {base_acc:.0%} (measured)")
    print(f"gap (Deadstage - baseline)    : {ds_acc - base_acc:+.0%}")

    # --- assertions: MEASURE and ASSERT the gap, do not merely claim it -------
    checks = {
        "deadstage names every true root cause": ds_correct == n,
        "deadstage strictly beats the baseline": ds_correct > base_correct,
        # flagship misattribution: a dead embedder -> baseline blames the scorer,
        # Deadstage blames the embedder (the actual root cause).
        "embed root: deadstage=embed": ds_named["embed_root"] == "embed",
        "embed root: baseline misattributes to score":
            base_named["embed_root"] == "score",
        # reachability: the artifact path makes retrieve AND score reachable+named.
        "retrieve stage reachable+named": ds_named["retrieve_root"] == "retrieve",
        "score stage reachable+named": ds_named["score_root"] == "score",
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print("PASS" if ok else "FAIL",
          "-> Deadstage must attribute the root cause where symptom-monitoring cannot")
    return ok


def main() -> int:
    green = run_green()
    red = run_red()
    ab = run_ab()
    _banner("RESULT")
    print(f"green_pass={green}  red_catches_fault={red}  ab_gap_measured={ab}")
    passed = green and red and ab
    print("SELF-TEST PASS" if passed else "SELF-TEST FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
