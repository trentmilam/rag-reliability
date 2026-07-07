"""RAGForensics eval -- RED/GREEN self-test of the FIRST MILESTONE. Exits 0 on pass.

    python ragforensics/eval.py

Deterministic (numpy default_rng(SEED); no wall-clock, no stdlib random). Proves,
with the REAL label-free mechanisms (no gold answers, no hard-coded verdicts):

  GREEN  a grounded query is NOT leak-flagged and attributes to "ok".
  RED    a parametrically-known query IS leak-flagged (answer invariant under
         context removal) -- and disagrees with the docs.
  RED    an injected retrieval miss attributes to "retriever".
  RED    an injected unfaithful generation attributes to "generator" (and is NOT
         mislabelled a leak -- the two axes are separable).
  CONTROL  the SAME leak query, with the parametric knowledge removed, drops the
           flag -> the flag is behavioral, not keyed on the query id (anti-rig).
  CALIB  the auto-calibrated threshold is data-derived: monotonic in the corpus
         baseline and strictly inside (floor, ceil) for a mixed corpus.
  DETERMINISM  two runs are identical.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import ablation  # noqa: E402
import fixtures  # noqa: E402
from forensics import (  # noqa: E402
    analyze,
    calibrate_leak_threshold,
    context_removal_invariance,
    derive_retrieval_floor,
)

SEED = fixtures.SEED


def _run(bundle):
    rng = np.random.default_rng(SEED)
    calib = calibrate_leak_threshold(bundle.answer_fn, bundle.calib_queries, bundle.calib_contexts)
    retr_floor = derive_retrieval_floor(bundle.retriever, bundle.calib_queries)
    verdicts = {}
    for q, ctx in bundle.test_cases:
        verdicts[q.id] = analyze(bundle.answer_fn, q, ctx, calib,
                                 retr_floor=retr_floor, rng=np.random.default_rng(SEED))
    return calib, retr_floor, verdicts


def main() -> int:
    rng = np.random.default_rng(SEED)
    b = fixtures.build(rng, leaky_calib=0)          # well-grounded corpus
    calib, retr_floor, v = _run(b)

    g = v[b.grounded_query.id]
    lk = v[b.leak_query.id]
    rt = v[b.retriever_query.id]
    gn = v[b.generator_query.id]

    checks = {}

    # -- GREEN: grounded query is clean --
    checks["green_grounded_not_leaked"] = (not g.leak_flag)
    checks["green_grounded_attr_ok"] = (g.attribution == "ok")
    checks["green_grounded_answer_from_docs"] = (g.answer == b.grounded_truth)

    # -- RED: parametric leak caught by real invariance mechanism --
    checks["red_leak_flagged"] = lk.leak_flag
    checks["red_leak_invariant"] = (lk.invariance >= calib.threshold)
    checks["red_leak_answer_is_parametric"] = (lk.answer == b.leak_parametric)
    checks["red_leak_disagrees_with_docs"] = (lk.answer != b.leak_doc_truth)

    # -- RED: retrieval miss attributed to the retriever --
    checks["red_retriever_attr"] = (rt.attribution == "retriever")
    checks["red_retriever_below_floor"] = (rt.retrieval_quality < retr_floor)

    # -- RED: unfaithful generation attributed to the generator, NOT a leak --
    checks["red_generator_attr"] = (gn.attribution == "generator")
    checks["red_generator_not_leak"] = (not gn.leak_flag)
    checks["red_generator_unsupported"] = (not gn.supported)

    # -- CONTROL (anti-rig): remove the parametric knowledge -> same query/context,
    #    flag disappears. Proves the verdict is behavioral, not id-keyed. --
    rng2 = np.random.default_rng(SEED)
    ctrl = fixtures.build(rng2, leaky_calib=0, inject_leak=False)
    ctrl_calib, ctrl_floor, cv = _run(ctrl)
    cl = cv[ctrl.leak_query.id]
    checks["control_same_query_not_flagged"] = (not cl.leak_flag)
    checks["control_answer_now_from_docs"] = (cl.answer == fixtures._TOPICS[7][2])
    checks["control_invariance_dropped"] = (cl.invariance < lk.invariance)

    # -- CALIB: auto-calibrated threshold is genuinely data-derived --
    thr = {}
    for n in (0, 1, 6):
        bb = fixtures.build(np.random.default_rng(SEED), leaky_calib=n)
        thr[n] = calibrate_leak_threshold(bb.answer_fn, bb.calib_queries, bb.calib_contexts).threshold
    checks["calib_monotonic"] = (thr[0] <= thr[1] <= thr[6])
    checks["calib_mixed_strictly_inside"] = (calib.floor < thr[1] < calib.ceil)
    checks["calib_reacts_to_corpus"] = (thr[0] != thr[6])

    # -- DETERMINISM --
    calib_b, floor_b, v_b = _run(fixtures.build(np.random.default_rng(SEED), leaky_calib=0))
    checks["determinism_threshold"] = (calib_b.threshold == calib.threshold)
    checks["determinism_verdicts"] = all(
        v_b[k] == v[k] for k in v
    )

    # -- ABLATION (the headline portfolio proof): the auto-calibrated leak
    #    threshold vs the OBVIOUS naive baseline (a fixed global 0.5 cutoff), scored
    #    on a held-out labelled mix over a legitimately-context-independent corpus.
    #    Both gates read the SAME real invariance signal; only the threshold differs.
    fixed_g, auto_g, auto_thr, _inv = ablation.run_ablation(np.random.default_rng(ablation.SEED))
    # the wedge, MEASURED (not asserted): the adaptive bar false-flags far less...
    checks["ablation_auto_fewer_false_positives"] = (auto_g.fp < fixed_g.fp)
    checks["ablation_auto_zero_false_positives"] = (auto_g.fp == 0)
    checks["ablation_auto_higher_precision"] = (auto_g.precision > fixed_g.precision)
    # ...at a net-better F1, i.e. the FP win is not bought by wrecking recall...
    checks["ablation_auto_higher_f1"] = (auto_g.f1 > fixed_g.f1)
    checks["ablation_auto_still_catches_leaks"] = (auto_g.recall >= 0.8)
    # ...and the winning threshold is DATA-DERIVED (strictly inside the clamp band
    #    and above the naive constant), i.e. tuned by the corpus, not hand-picked.
    checks["ablation_threshold_data_derived"] = (0.35 < auto_thr < 0.95)
    checks["ablation_threshold_above_naive"] = (auto_thr > ablation.FIXED_CUTOFF)
    # ...deterministic across runs.
    fixed_g2, auto_g2, auto_thr2, _ = ablation.run_ablation(np.random.default_rng(ablation.SEED))
    checks["ablation_determinism"] = (
        (fixed_g2.fp, fixed_g2.fn, auto_g2.fp, auto_g2.fn, auto_thr2)
        == (fixed_g.fp, fixed_g.fn, auto_g.fp, auto_g.fn, auto_thr)
    )

    # ---- report ----
    print("=== RAGForensics eval (measured, label-free) ===")
    print(f"auto-calibrated leak threshold = {calib.threshold:.4f} "
          f"(q90={calib.q90:.4f}, floor={calib.floor}, ceil={calib.ceil})")
    print(f"corpus-derived retrieval floor = {retr_floor:.4f}")
    print(f"threshold vs corpus baseline   : leaky0={thr[0]:.2f} leaky1={thr[1]:.2f} leaky6={thr[6]:.2f}")
    print()
    print(f"  grounded  ({b.grounded_query.id:9s}) ans={g.answer:11s} inv={g.invariance:.2f} "
          f"leak={g.leak_flag!s:5s} attr={g.attribution}")
    print(f"  leak      ({b.leak_query.id:9s}) ans={lk.answer:11s} inv={lk.invariance:.2f} "
          f"leak={lk.leak_flag!s:5s} attr={lk.attribution}  (docs say '{b.leak_doc_truth}')")
    print(f"  retriever ({b.retriever_query.id:9s}) ans={rt.answer:11s} rq={rt.retrieval_quality:.2f} "
          f"(<floor {retr_floor:.2f}) attr={rt.attribution}")
    print(f"  generator ({b.generator_query.id:9s}) ans={gn.answer:11s} inv={gn.invariance:.2f} "
          f"supported={gn.supported!s:5s} leak={gn.leak_flag!s:5s} attr={gn.attribution}")
    print(f"  CONTROL   ({ctrl.leak_query.id:9s}) ans={cl.answer:11s} inv={cl.invariance:.2f} "
          f"leak={cl.leak_flag!s:5s}  (parametric knowledge removed)")
    print()
    print("--- ABLATION: auto-calibrated vs fixed-0.5 (labelled held-out mix) ---")
    print(f"  auto-calibrated leak threshold = {auto_thr:.3f} (data-derived) vs "
          f"naive fixed = {ablation.FIXED_CUTOFF}")
    print(f"  {'gate':16s}  TP FP FN TN   prec   rec    F1")
    for gg in (fixed_g, auto_g):
        print(f"  {gg.name:16s}  {gg.tp:2d} {gg.fp:2d} {gg.fn:2d} {gg.tn:2d}  "
              f"{gg.precision:.3f} {gg.recall:.3f} {gg.f1:.3f}")
    print(f"  => false positives {fixed_g.fp} -> {auto_g.fp}, F1 {fixed_g.f1:.3f} -> {auto_g.f1:.3f}")
    print()
    for k, ok in checks.items():
        print(f"{'OK  ' if ok else 'FAIL'} {k}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
