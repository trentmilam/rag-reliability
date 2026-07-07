"""ReindexGate FIRST-MILESTONE red/green self-test (deterministic, offline).

Runs the REAL gate mechanism (no hard-coded verdicts, no rigging) on three
scenarios and asserts the gate behaves correctly:

  GREEN  no-op reindex (identical config)      -> gate PASSES, CI straddles 0.
  RED    dim-truncation reindex (256 -> 16)    -> gate FAILS, CI separated < 0.
  DEBIAS weak OLD (dim 24) vs strong NEW (256) -> the pooling-bias correction
         materially changes the number (corrected delta >> incumbent-pool
         "biased" delta), proving the debiasing is a live mechanism, not inert.

exit 0 iff every assertion holds.
"""

from __future__ import annotations

import sys

import numpy as np

from index import IndexConfig
from embedder import REF_DIM, REF_SEED, REF_ORACLE_SEED, embed_batch
from corpus import build_corpus
from gate import FAIL_MARGIN, REL_THRESH, reference_simmatrix, run_gate
from concordance import (
    GOLD_TAU, RBO_TAU, measure_scenario, cohen_kappa, pearson, kendall_tau,
    agreement_rate,
)

SEED = 1234


def _check(label: str, cond: bool) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def _cfg(dim: int = 256, chunker: str = "sentence", seed: int | None = None,
         name: str = "idx") -> IndexConfig:
    if seed is None:
        return IndexConfig(embed_dim=dim, chunker=chunker, name=name)
    return IndexConfig(embed_dim=dim, chunker=chunker, embed_seed=seed, name=name)


# A reindex scenario matrix spanning no-ops, real regressions, genuine
# improvements, a worse chunker, and a benign same-quality reseed. GOLD (real
# nDCG on structural qrels) is the ground-truth verdict for each; we measure how
# often ReindexGate's LABEL-FREE verdict matches it, versus the naive incumbent
# "alert on RBO churn".
_SCENARIOS = [
    ("no-op",              _cfg(256), _cfg(256)),
    ("dim 256->128",       _cfg(256), _cfg(128)),
    ("dim 256->64",        _cfg(256), _cfg(64)),
    ("dim 256->16",        _cfg(256), _cfg(16)),
    ("dim 256->8",         _cfg(256), _cfg(8)),
    ("whole chunker",      _cfg(256, "sentence"), _cfg(256, "whole")),
    ("IMPROVE 16->256",    _cfg(16), _cfg(256)),
    ("IMPROVE 64->256",    _cfg(64), _cfg(256)),
    ("IMPROVE whole->sent", _cfg(256, "whole"), _cfg(256, "sentence")),
    ("BENIGN reseed",      _cfg(256, seed=REF_SEED), _cfg(256, seed=0x1234ABCD)),
]


def concordance_section() -> bool:
    """Measure agreement of the LABEL-FREE gate with REAL nDCG on GOLD labels,
    and prove it beats the naive RBO-churn baseline. This is the portfolio
    hero measurement: it converts "plausible heuristic" into a number."""
    print("[CONCORDANCE] label-free gate vs REAL nDCG on GOLD labels "
          "(+ naive RBO baseline)")
    ms = [measure_scenario(n, o, nw, seed=SEED) for (n, o, nw) in _SCENARIOS]

    print(f"  {'scenario':22} {'gold_d':>8} {'gate_d':>8} {'biased_d':>9} "
          f"{'rbo':>6} | {'gold?':>5} {'gate?':>5} {'rbo?':>5}")
    for m in ms:
        print(f"  {m.name:22} {m.gold_delta:+8.4f} "
              f"{m.gate.mean_delta_corrected:+8.4f} "
              f"{m.gate.mean_delta_biased:+9.4f} {m.mean_rbo:6.3f} | "
              f"{str(m.gold_regression):>5} {str(m.gate_regression):>5} "
              f"{str(m.rbo_regression):>5}")

    gold = [m.gold_regression for m in ms]
    gate = [m.gate_regression for m in ms]
    rbo = [m.rbo_regression for m in ms]
    gold_d = [m.gold_delta for m in ms]
    gate_d = [m.gate.mean_delta_corrected for m in ms]
    rbo_sig = [1.0 - m.mean_rbo for m in ms]

    gate_acc = agreement_rate(gate, gold)
    rbo_acc = agreement_rate(rbo, gold)
    gate_kappa = cohen_kappa(gate, gold)
    rbo_kappa = cohen_kappa(rbo, gold)
    r_pearson = pearson(gate_d, gold_d)
    r_kendall = kendall_tau(gate_d, gold_d)
    r_rbo_pearson = pearson(rbo_sig, gold_d)
    rbo_false_alarms = sum(1 for m in ms if m.rbo_regression and not m.gold_regression)

    print()
    print(f"  gate vs gold : accuracy={gate_acc:.3f}  Cohen kappa={gate_kappa:+.3f}")
    print(f"  RBO  vs gold : accuracy={rbo_acc:.3f}  Cohen kappa={rbo_kappa:+.3f}  "
          f"(naive incumbent)")
    print(f"  signed-delta agreement: Pearson(gate,gold)={r_pearson:+.3f}  "
          f"Kendall={r_kendall:+.3f}")
    print(f"  RBO churn cannot give direction: Pearson(1-RBO, gold_delta)="
          f"{r_rbo_pearson:+.3f}")
    print(f"  RBO false alarms (flagged, but gold shows no regression): "
          f"{rbo_false_alarms}/{len(ms)}")

    ok = True
    ok &= _check("label-free gate matches GOLD verdict on every scenario "
                 f"(acc={gate_acc:.3f})", gate_acc == 1.0)
    ok &= _check(f"gate-vs-gold Cohen kappa is near-perfect ({gate_kappa:+.3f} >= 0.99)",
                 gate_kappa >= 0.99)
    ok &= _check(f"gate signed delta tracks gold (Pearson {r_pearson:+.3f} >= 0.99)",
                 r_pearson >= 0.99)
    ok &= _check(f"gate signed delta rank-tracks gold (Kendall {r_kendall:+.3f} >= 0.50)",
                 r_kendall >= 0.50)
    ok &= _check("gate BEATS the naive RBO baseline on gold accuracy "
                 f"({gate_acc:.3f} > {rbo_acc:.3f})", gate_acc > rbo_acc)
    ok &= _check("gate BEATS the naive RBO baseline on gold kappa "
                 f"({gate_kappa:+.3f} > {rbo_kappa:+.3f})", gate_kappa > rbo_kappa)
    ok &= _check("naive RBO baseline false-alarms on >=2 benign/improved reindexes "
                 f"({rbo_false_alarms})", rbo_false_alarms >= 2)

    # The corrected==gold identity is EARNED by pooling-bias correction, not a
    # definitional tautology: on a shallow-pool improvement the SAME oracle judged
    # on the incumbent pool (biased) diverges sharply from gold, while the
    # union-pool corrected delta reproduces gold to the last digit.
    d = measure_scenario("shallow-pool improve 16->256",
                         _cfg(16), _cfg(256), seed=SEED, k=2)
    print()
    print(f"  [earned-concordance] shallow improve 16->256 @k=2: "
          f"gold={d.gold_delta:+.4f} corrected={d.gate.mean_delta_corrected:+.4f} "
          f"biased={d.gate.mean_delta_biased:+.4f}")
    ok &= _check("union-pool CORRECTED delta reproduces GOLD exactly",
                 abs(d.gate.mean_delta_corrected - d.gold_delta) < 1e-9)
    ok &= _check("incumbent-pool BIASED delta diverges from GOLD by > 0.10 "
                 "(so the match is earned by debiasing, not tautological)",
                 abs(d.gate.mean_delta_biased - d.gold_delta) > 0.10)
    print()
    return ok


def main() -> int:
    print("ReindexGate red/green self-test (seed=%d)\n" % SEED)
    ok = True

    old = IndexConfig(embed_dim=256, chunker="sentence", name="OLD")

    # ---------------------------------------------------------- GREEN: no-op ---
    print("[GREEN] no-op reindex (NEW config == OLD config)")
    g = run_gate(old, IndexConfig(embed_dim=256, chunker="sentence", name="NEW"),
                 seed=SEED)
    print(g.report())
    ok &= _check("gate PASSES the no-op reindex", g.passed)
    ok &= _check("CI straddles zero (no false regression)",
                 g.ci_low <= 0.0 <= g.ci_high)
    ok &= _check("ranking unchanged (mean RBO == 1.0)", abs(g.mean_rbo - 1.0) < 1e-9)
    print()

    # ------------------------------------------------------ RED: truncation ---
    print("[RED] dim-truncation reindex (embed_dim 256 -> 16)")
    r = run_gate(old, IndexConfig(embed_dim=16, chunker="sentence", name="NEW"),
                 seed=SEED)
    print(r.report())
    ok &= _check("gate FAILS the known-bad reindex", not r.passed)
    ok &= _check("regression is real (mean corrected delta < 0)",
                 r.mean_delta_corrected < 0.0)
    ok &= _check("CI separated from zero below the fail margin",
                 r.ci_high < -FAIL_MARGIN)
    ok &= _check("ranking churned (mean RBO < 1.0)", r.mean_rbo < 1.0)
    print()

    # --------------------------------------------- DEBIAS: correction is live ---
    print("[DEBIAS] weak OLD (dim 24) vs strong NEW (dim 256), shallow pool k=2")
    d = run_gate(IndexConfig(embed_dim=24, name="OLD"),
                 IndexConfig(embed_dim=256, name="NEW"), seed=SEED, k=2)
    print(d.report())
    gap = d.mean_delta_corrected - d.mean_delta_biased
    print(f"  pooling-bias correction gap (corrected - biased): {gap:+.4f}")
    ok &= _check("incumbent-pool 'biased' eval under-credits the better NEW",
                 d.mean_delta_biased < d.mean_delta_corrected)
    ok &= _check("correction materially changes the number (gap > 0.05)",
                 gap > 0.05)
    ok &= _check("debiased delta correctly says NEW is not worse (PASS)", d.passed)
    print()

    # ------------------------------------- INDEPENDENCE: oracle != OLD/NEW ---
    # The README claims the pseudo-relevance oracle is "never derived from OLD
    # or NEW." OLD and NEW are built with the default index seed REF_SEED; a
    # genuinely independent oracle must therefore live in a DIFFERENT hash
    # family. Prior to the fix the oracle used REF_SEED too -> its sim-matrix was
    # BYTE-IDENTICAL to OLD's own doc-embedding similarity, and check (1) below
    # would FAIL (catching the overclaim). It passes only because the oracle now
    # uses REF_ORACLE_SEED, a distinct near-orthogonal family.
    print("[INDEPENDENCE] oracle is a third signal, not OLD/NEW's embedder")
    docs = build_corpus(seed=SEED)
    oracle_sim = reference_simmatrix(docs)
    # OLD/NEW's embedder family (default index seed) over the same docs:
    old_family = embed_batch([d.text for d in docs], dim=REF_DIM, weights_seed=REF_SEED)
    old_family_sim = old_family @ old_family.T
    ok &= _check("oracle seed differs from OLD/NEW index seed",
                 REF_ORACLE_SEED != REF_SEED)
    ok &= _check("oracle sim-matrix is NOT byte-identical to OLD/NEW's",
                 not np.array_equal(oracle_sim, old_family_sim))
    sim_absdiff = np.abs(oracle_sim - old_family_sim)
    ok &= _check("oracle is materially distinct (some cell diff > 0.10)",
                 float(sim_absdiff.max()) > 0.10)
    # The independent oracle must still be a VALID signal, not noise: paraphrase
    # partners separate cleanly from non-partners across REL_THRESH.
    partner_min, nonpartner_max = 1.0, -1.0
    for a in docs:
        for b in docs:
            if a.doc_id == b.doc_id:
                continue
            v = float(oracle_sim[a.doc_id, b.doc_id])
            if b.pair_id == a.pair_id:
                partner_min = min(partner_min, v)
            else:
                nonpartner_max = max(nonpartner_max, v)
    ok &= _check("independent oracle still separates partners over REL_THRESH",
                 nonpartner_max < REL_THRESH <= partner_min)
    print(f"  partner_min={partner_min:+.3f}  nonpartner_max={nonpartner_max:+.3f}  "
          f"REL_THRESH={REL_THRESH}")
    print()

    ok &= concordance_section()

    print("RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
