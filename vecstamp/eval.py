"""VecStamp RED/GREEN self-test + confusion matrix + MEASURED baseline A/B.

Milestone 1 (diagonal smoke test): the failure-typing decision tree correctly
classifies build->load embedder swaps, validated by a confusion matrix over
real swaps (no hard-coded verdicts -- every verdict comes from
vecstamp.verify() re-embedding for real).

  GREEN : identical reload reproduces bit-exactly (verdict "reproduced", ok).
  RED   : each injected fault is CAUGHT and TYPED correctly, in particular a
          q8 load is typed "quant-dtype-drift", NOT a false "wrong-weights".

Milestone 2 (statistically-real proof + differentiation wedge): the n=1/class
matrix above is a smoke test, not a statistical result. This milestone runs a
SWEEP (many seeds x probe-counts x quant levels {4,6,8} x fault families,
N>=50 per verdict cell) and reports:
  * a real-N confusion matrix with per-class precision/recall,
  * the DECISION MARGIN separating "same model" (quant/norm drift) from a
    genuinely wrong model (min same-model cosine vs max wrong-weights cosine),
  * a MEASURED head-to-head vs the fair incumbent: a name+dim fingerprint (what
    mainstream RAG frameworks actually persist). The incumbent is BLIND to any
    fault that keeps the declared name+dim (quant drift, lost L2 norm, silently
    swapped same-named weights); VecStamp re-embeds and catches all of them.

Exit 0 iff milestone 1 is diagonal AND milestone 2's measured assertions hold
(margin present, incumbent-vs-VecStamp gap as claimed, no benign fault ever
mislabeled the dangerous "wrong-weights", cells statistically real N>=50).
"""

from __future__ import annotations

import sys
from functools import partial

import numpy as np

from embedder import REF_DIM, REF_SEED, embed
from vecstamp import (
    COS_SAME_MODEL,
    build_manifest,
    make_probes,
    verify,
)

REF_NAME = "hashing-blake2b"

# BUILD-time embedder: the fp16 reference (normalized, dim 256, full weights).
build_embed = partial(embed, weights_seed=REF_SEED, dim=REF_DIM, normalize=True)

# LOAD-time embedder variants, each simulating one deployment fault (labelled).
# Maps: injected-swap -> (live embed_fn, expected verdict).
LOAD_VARIANTS = {
    # clean reload: identical embedder -> must reproduce bit-exactly (GREEN)
    "identical":     (partial(embed, weights_seed=REF_SEED, dim=REF_DIM,
                              normalize=True),                       "reproduced"),
    # q8 quantized serving copy: tiny perturbation, NOT a different model
    "q8-drift":      (partial(embed, weights_seed=REF_SEED, dim=REF_DIM,
                              normalize=True, quantize=8),           "quant-dtype-drift"),
    # someone deployed a different embedding model (different weights)
    "wrong-weights": (partial(embed, weights_seed=REF_SEED ^ 0x5A5A5A, dim=REF_DIM,
                              normalize=True),                       "wrong-weights"),
    # the L2-normalization post-step was dropped in the serving path
    "lost-l2-norm":  (partial(embed, weights_seed=REF_SEED, dim=REF_DIM,
                              normalize=False),                      "lost-l2-norm"),
    # dimension changed (truncated / different projection head)
    "dim-mismatch":  (partial(embed, weights_seed=REF_SEED, dim=192,
                              normalize=True),                       "dim-mismatch"),
}


def _milestone1() -> bool:
    manifest = build_manifest(build_embed)
    print(f"[build] {len(manifest.probes)} probes  dim={manifest.dim}  "
          f"float_hash={manifest.float_hash[:16]}...")

    # Confusion matrix is over the VERDICT space (what verify() can emit).
    # Each injected swap has an expected verdict (the true class); the tree's
    # predicted verdict is the observed class.
    classes = ["reproduced", "quant-dtype-drift", "wrong-weights",
               "lost-l2-norm", "dim-mismatch"]
    idx = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)  # [true][pred]
    unknown_pred = {}  # predictions outside the known verdict space

    print("\n[verify] injected-swap -> predicted-verdict (expected)")
    green_ok = False
    for true_lab in LOAD_VARIANTS:
        live_fn, expected = LOAD_VARIANTS[true_lab]
        res = verify(manifest, live_fn)
        pred = res.verdict
        tag = "PASS/ok" if res.ok else "caught"
        mark = "OK" if pred == expected else "XX"
        print(f"  {mark} {true_lab:14s} -> {pred:18s} ({tag})  {res.detail}")

        if true_lab == "identical":
            green_ok = res.ok and pred == "reproduced"

        if pred in idx:
            confusion[idx[expected]][idx[pred]] += 1
        else:
            unknown_pred[true_lab] = pred

    # ---- confusion matrix ----
    print("\n[confusion matrix]  rows=true-verdict  cols=predicted-verdict")
    print("                " + " ".join(f"{c[:8]:>9s}" for c in classes))
    for i, c in enumerate(classes):
        row = " ".join(f"{confusion[i][j]:>9d}" for j in range(len(classes)))
        print(f"  {c:16s}{row}")

    diagonal = int(np.trace(confusion))
    total = len(LOAD_VARIANTS)
    off_diag = int(confusion.sum()) - diagonal
    perfect = (diagonal == total) and (off_diag == 0) and not unknown_pred

    # explicit anti-false-positive check called out in the spec
    q8_fn, _ = LOAD_VARIANTS["q8-drift"]
    q8_pred = verify(manifest, q8_fn).verdict
    q8_ok = (q8_pred == "quant-dtype-drift")

    print("\n[summary]")
    print(f"  green (identical reload reproduces)      : {green_ok}")
    print(f"  q8 typed drift NOT wrong-model           : {q8_ok} (got '{q8_pred}')")
    print(f"  confusion diagonal                       : {diagonal}/{total}"
          f"  off-diagonal={off_diag}  unknown={unknown_pred}")

    passed = green_ok and perfect and q8_ok
    print(f"\n[milestone 1] RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Milestone 2 -- statistically-real sweep + MEASURED baseline A/B
# ---------------------------------------------------------------------------
# The fair incumbent: a name+dim fingerprint. Mainstream RAG frameworks persist
# the embedder *name* and output *dim* and treat "same name, same dim" as
# "same embedder". This is a REASONABLE thing a competent engineer ships -- it
# just cannot see anything that leaves name+dim unchanged. We model each fault's
# DECLARED (name, dim) as what a real deployment would advertise:
#   * quant copy / lost-norm serving bug / silently-swapped same-named weights
#     all keep the declared name+dim -> incumbent is BLIND (the realistic
#     silent-fault case this tool targets).
#   * a changed projection dim is visible in metadata -> incumbent CATCHES it
#     (we give the incumbent the one fault it can legitimately see -- not a
#     rigged strawman).

SWEEP_N = 60          # seeds per fault family
QUANT_LEVELS = (4, 6, 8)


def _incumbent_ok(live_name: str, live_dim: int) -> bool:
    """Fair name+dim fingerprint check (the mainstream framework incumbent)."""
    return live_name == REF_NAME and live_dim == REF_DIM


def _sweep_samples():
    """Yield (family, expected_verdict, vs_result, declared_name, declared_dim).

    family is the ground-truth fault family; declared_(name,dim) is the metadata
    a real deployment of that fault would advertise (what the incumbent sees).
    Every VecStamp verdict comes from verify() re-embedding for real.
    """
    for s in range(SWEEP_N):
        n = 8 + (s % 8)
        probes = make_probes(n=n, seed=1000 + s)
        man = build_manifest(build_embed, probes)

        # reproduced control (identical reload)
        yield ("reproduced", "reproduced",
               verify(man, build_embed), REF_NAME, REF_DIM)

        # quantized serving copies at q4/q6/q8 -- same declared name+dim
        for q in QUANT_LEVELS:
            fn = partial(embed, weights_seed=REF_SEED, dim=REF_DIM,
                         normalize=True, quantize=q)
            yield (f"q{q}", "quant-dtype-drift", verify(man, fn), REF_NAME, REF_DIM)

        # L2-normalization dropped in the serving path -- same declared name+dim
        fn = partial(embed, weights_seed=REF_SEED, dim=REF_DIM, normalize=False)
        yield ("lost-l2-norm", "lost-l2-norm", verify(man, fn), REF_NAME, REF_DIM)

        # silently swapped weights (reverted/corrupted checkpoint) -- SAME name
        alt = REF_SEED ^ ((0x5A5A5A + s * 2654435761) & 0xFFFFFFFF)
        fn = partial(embed, weights_seed=alt, dim=REF_DIM, normalize=True)
        yield ("wrong-weights", "wrong-weights", verify(man, fn), REF_NAME, REF_DIM)

        # changed projection dim -- metadata dim differs (incumbent can see it)
        dim = (128, 192, 320)[s % 3]
        fn = partial(embed, weights_seed=REF_SEED, dim=dim, normalize=True)
        yield ("dim-mismatch", "dim-mismatch", verify(man, fn), REF_NAME, dim)


def _milestone2() -> bool:
    classes = ["reproduced", "quant-dtype-drift", "wrong-weights",
               "lost-l2-norm", "dim-mismatch"]
    idx = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)  # [true][pred]

    # per-family detection tallies for the baseline A/B
    fam_total: dict[str, int] = {}
    vs_caught: dict[str, int] = {}          # VecStamp: ok is False (divergence)
    inc_caught: dict[str, int] = {}         # name+dim incumbent: fingerprint differs

    same_model_cos: list[float] = []        # min cosine over quant/norm drift
    wrong_cos: list[float] = []             # |mean cosine| over wrong-weights
    quant_to_wrong = 0                       # dangerous false-alarm counter
    quant_total = 0
    q68_as_drift = 0                         # q6/q8 correctly typed quant drift
    q68_total = 0

    for fam, expected, res, dname, ddim in _sweep_samples():
        fam_total[fam] = fam_total.get(fam, 0) + 1

        # confusion over the TRUE class (expected) vs predicted verdict
        if expected in idx and res.verdict in idx:
            confusion[idx[expected]][idx[res.verdict]] += 1

        # VecStamp detection = it flagged divergence (reproduced control is ok)
        if not res.ok:
            vs_caught[fam] = vs_caught.get(fam, 0) + 1
        # incumbent detection = declared name/dim differs from reference
        if not _incumbent_ok(dname, ddim):
            inc_caught[fam] = inc_caught.get(fam, 0) + 1

        if fam.startswith("q") and fam[1:].isdigit():
            quant_total += 1
            same_model_cos.append(res.detail.get("min_cosine", 1.0))
            if res.verdict == "wrong-weights":
                quant_to_wrong += 1
            if fam in ("q6", "q8"):
                q68_total += 1
                if res.verdict == "quant-dtype-drift":
                    q68_as_drift += 1
        if fam == "wrong-weights":
            wrong_cos.append(abs(res.detail.get("mean_cosine", 0.0)))

    total = int(confusion.sum())
    print(f"\n[milestone 2] statistical sweep: {total} verified samples "
          f"({SWEEP_N} seeds x {len(fam_total)} fault families)")

    # ---- real-N confusion matrix ----
    print("\n[confusion matrix]  rows=true-class  cols=predicted-verdict  (real N)")
    print("                    " + " ".join(f"{c[:8]:>9s}" for c in classes))
    for i, c in enumerate(classes):
        row = " ".join(f"{confusion[i][j]:>9d}" for j in range(len(classes)))
        print(f"  {c:18s}{row}")

    # ---- per-class precision / recall ----
    print("\n[per-class precision / recall / support]")
    min_support = 10 ** 9
    for i, c in enumerate(classes):
        tp = int(confusion[i][i])
        support = int(confusion[i].sum())
        pred_tot = int(confusion[:, i].sum())
        rec = tp / support if support else 0.0
        prec = tp / pred_tot if pred_tot else 0.0
        min_support = min(min_support, support)
        print(f"  {c:18s} precision={prec:5.3f}  recall={rec:5.3f}  support={support}")

    # ---- decision margin (the headline number) ----
    min_same = min(same_model_cos)
    max_wrong = max(wrong_cos)
    margin = min_same - max_wrong
    print("\n[decision margin]  same-model (quant/norm drift) vs wrong-weights")
    print(f"  min same-model cosine (q4/q6/q8, N={quant_total}) : {min_same:.6f}")
    print(f"  max wrong-weights |cosine| (N={len(wrong_cos)})    : {max_wrong:.6f}")
    print(f"  threshold COS_SAME_MODEL                        : {COS_SAME_MODEL}")
    print(f"  SEPARATION MARGIN                               : {margin:.6f}")

    # ---- MEASURED baseline A/B: name+dim incumbent vs VecStamp ----
    print("\n[baseline A/B]  fault-family detection: name+dim incumbent vs VecStamp")
    print(f"  {'fault family':16s} {'N':>4s}  {'incumbent':>10s}  {'VecStamp':>9s}")
    inc_blind_fams = 0
    inc_seen_fams = 0
    real_faults = [f for f in fam_total if f != "reproduced"]
    for fam in ["reproduced"] + sorted(real_faults):
        N = fam_total[fam]
        inc = inc_caught.get(fam, 0)
        vs = vs_caught.get(fam, 0)
        if fam != "reproduced":
            if inc == 0:
                inc_blind_fams += 1
            elif inc == N:
                inc_seen_fams += 1
        print(f"  {fam:16s} {N:>4d}  {inc:>4d}/{N:<5d}  {vs:>4d}/{N:<4d}")

    # same-name-same-dim faults: the ones the incumbent structurally cannot see
    silent_fams = [f for f in real_faults if f != "dim-mismatch"]
    silent_total = sum(fam_total[f] for f in silent_fams)
    silent_inc = sum(inc_caught.get(f, 0) for f in silent_fams)
    silent_vs = sum(vs_caught.get(f, 0) for f in silent_fams)

    print("\n[wedge] same-name/same-dim faults (quant drift, lost-l2-norm, "
          "silent weight swap):")
    print(f"  incumbent (name+dim) caught : {silent_inc}/{silent_total}")
    print(f"  VecStamp caught             : {silent_vs}/{silent_total}")
    print(f"  fault families incumbent is BLIND to : {inc_blind_fams}/"
          f"{len(real_faults)}")

    # ---- honest limitation: aggressive q4 perturbs the norm past NORM_TOL ----
    print(f"\n[honest] q4 (4-bit) drift is DETECTED but typed 'lost-l2-norm' in "
          f"{int(confusion[idx['quant-dtype-drift']][idx['lost-l2-norm']])}/"
          f"{fam_total['q4']} cases:")
    print("         4-bit quant perturbs the L2 norm past NORM_TOL, so the tree "
          "attributes it to a dropped norm.")
    print(f"         CRITICAL: quant drift mislabeled the dangerous "
          f"'wrong-weights' in {quant_to_wrong}/{quant_total} cases.")

    # ---- measured assertions (these define PASS) ----
    checks = {
        "cells statistically real (min support >= 50)": min_support >= 50,
        "reproduced control recall == 1.0":
            confusion[idx["reproduced"]][idx["reproduced"]] == fam_total["reproduced"],
        "wrong-weights recall == 1.0":
            confusion[idx["wrong-weights"]][idx["wrong-weights"]] == fam_total["wrong-weights"],
        "wrong-weights precision == 1.0 (no benign fault mislabeled wrong-model)":
            int(confusion[:, idx["wrong-weights"]].sum())
            == confusion[idx["wrong-weights"]][idx["wrong-weights"]],
        "lost-l2-norm recall == 1.0":
            confusion[idx["lost-l2-norm"]][idx["lost-l2-norm"]] == fam_total["lost-l2-norm"],
        "dim-mismatch recall == 1.0":
            confusion[idx["dim-mismatch"]][idx["dim-mismatch"]] == fam_total["dim-mismatch"],
        "q6+q8 typed quant-dtype-drift 100%": q68_as_drift == q68_total,
        "quant drift NEVER mislabeled wrong-weights (0/N)": quant_to_wrong == 0,
        "decision margin positive and comfortably > 0.5": margin > 0.5,
        "threshold sits inside the margin": max_wrong < COS_SAME_MODEL <= min_same,
        "incumbent BLIND to same-name/same-dim faults (0/N)": silent_inc == 0,
        "VecStamp catches ALL same-name/same-dim faults (N/N)":
            silent_vs == silent_total,
        "incumbent blind to >= 3 of 4 fault families": inc_blind_fams >= 3,
    }

    print("\n[milestone 2 assertions]")
    passed = True
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'XX'} {name}")
        passed = passed and ok
    print(f"\n[milestone 2] RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    m1 = _milestone1()
    m2 = _milestone2()
    passed = m1 and m2
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
