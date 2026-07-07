# -*- coding: utf-8 -*-
"""Plumbline MEASURED A/B benchmark: fuzzy provenance resolver vs the naive
exact-substring (content-hash-style) baseline, over a large realistic OCR corpus.

Why this exists
---------------
`eval.py` proves the wedge EXISTS on 5 hand-built citations. That reads as "it
works on my example." This benchmark turns the claim into a defensible NUMBER: it
builds a ~1000-chunk synthetic financial corpus, degrades the citations with a
mix of MODELED OCR normalization (what Plumbline canonicalizes) and UNMODELED OCR
noise (random insert/delete/transpose that Plumbline does NOT model), and MEASURES
recall + false-positive rate for BOTH resolvers on the SAME corpus.

Fairness of the baseline
------------------------
The comparator is `naive_resolve` — exact-substring lookup, i.e. exactly what a
content-hash / exact-match provenance check does. That is the reasonable incumbent
a competent engineer ships, not a crippled strawman: it is given the FULL, correct
source and the FULL citation text, and it succeeds on every clean (un-degraded)
citation (measured below). It only loses where the stored chunk was legitimately
OCR/whitespace-normalized relative to the source — which is the wedge.

Honesty
-------
This benchmark does NOT flatter Plumbline. It MEASURES and REPORTS two things the
tool does badly, and asserts them so they can't silently regress into a rosy claim:
  * Plumbline recall on UNMODELED OCR noise is < 1.0 (it is not a magic aligner;
    the fixture is not tautologically winnable — cf. eval.py which only uses
    reversible transforms).
  * Plumbline manufactures FALSE provenance for short numeric citations because
    the OCR confusion table folds digits to letters (0->o, 1->l, 5->s): "100"
    canonicalizes to "loo" and matches the word "look". The naive baseline has a
    0% false-positive rate on the exact same numeric negatives. This is a REAL
    precision hole, measured here and disclosed in the README.

Deterministic: numpy default_rng(SEED); numpy + stdlib only; offline; no GPU.
Exit 0 iff every measured assertion holds.
"""

from __future__ import annotations

import sys

import numpy as np

from plumbline import build_manifest, naive_resolve, resolve

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 20260704
N_CHUNKS = 1000  # present (legitimate) citations

# --- corpus vocabulary -------------------------------------------------------
# Deliberately includes "collision-bait" prose tokens (look-through, balloon,
# rollover) so the digit-fold precision hole can be MEASURED, not just asserted.
_SUBJECTS = [
    "The board", "The audit committee", "Management", "The finance team",
    "The risk committee", "The treasury desk", "The investment committee",
    "The valuation group", "The compliance office", "The portfolio team",
]
_VERBS = [
    "confirmed", "reported", "disclosed", "reaffirmed", "flagged",
    "projected", "reviewed", "approved", "restated", "acknowledged",
]
_OBJECTS = [
    "the look-through exposure", "the balloon maturity schedule",
    "the rollover assumption", "the operating margin trend",
    "the reconciliation gap", "the collateral coverage ratio",
    "the liquidity buffer", "the fee reconciliation", "the hedge effectiveness",
    "the impairment allowance",
]
_REGIONS = [
    "the northern region", "the coastal division", "the offshore book",
    "the legacy portfolio", "the growth segment", "the core mandate",
    "the reserve account", "the syndicated tranche",
]


def _alpha_tag(i: int) -> str:
    """Digit-free unique tag (base-26 letters) so the corpus contains NO digits;
    that keeps the numeric-negative false-positive test clean for the baseline."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def _build_source_and_citations(rng: np.random.Generator):
    """Return (source_text, {cite_id: clean_sentence}). Each sentence is unique."""
    sentences = []
    for i in range(N_CHUNKS):
        s = (f"{_SUBJECTS[rng.integers(len(_SUBJECTS))]} "
             f"{_VERBS[rng.integers(len(_VERBS))]} "
             f"{_OBJECTS[rng.integers(len(_OBJECTS))]} across "
             f"{_REGIONS[rng.integers(len(_REGIONS))]} in reporting unit "
             f"{_alpha_tag(i)}.")
        sentences.append(s)
    source = "\n\n".join(sentences)
    cites = {f"C{i:04d}": sentences[i] for i in range(N_CHUNKS)}
    return source, cites


# --- degradation -------------------------------------------------------------
def _modeled_ocr(text: str, rng: np.random.Generator) -> str:
    """Apply ONLY transforms Plumbline canonicalizes (case flip, o->0, l->1,
    fi->ligature, whitespace doubling). A legitimate chunk that survives Plumbline
    but breaks exact-substring."""
    s = text.swapcase()                       # case alone already breaks naive
    s = s.replace("O", "0", 2).replace("o", "0", 2)
    s = s.replace("L", "1", 1).replace("l", "1", 1)
    s = s.replace("fi", "ﬁ", 1).replace("FI", "ﬁ", 1)
    sp = s.find(" ", 1)
    if sp != -1:
        s = s[:sp] + "  " + s[sp + 1:]        # double one internal space
    return s


def _unmodeled_ocr(text: str, rng: np.random.Generator) -> str:
    """Real OCR corruption Plumbline does NOT model: a random insert / delete /
    transpose of alphabetic chars inside the span. Expected to defeat exact-
    substring alignment (both resolvers should mostly miss)."""
    chars = list(text)
    n = len(chars)
    op = int(rng.integers(3))
    pos = int(rng.integers(1, n - 1))
    if op == 0:  # insert a stray letter
        chars.insert(pos, chr(ord("a") + int(rng.integers(26))))
    elif op == 1:  # delete a letter
        del chars[pos]
    else:  # transpose adjacent
        chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
    return "".join(chars)


def _novel_sentence(rng: np.random.Generator) -> str:
    """A plausible sentence that is NOT in the source (true negative)."""
    return (f"An external counterparty questioned the settlement window for "
            f"unit {_alpha_tag(int(rng.integers(10_000, 20_000)))} offshore.")


def _resolves_correct(span, true_start, true_end) -> bool:
    """A resolution is a true positive only if it maps to the correct source
    region (unique sentences => correct == exact start match)."""
    return span is not None and span[0] == true_start


def _recall(resolver, source, cites, truth) -> float:
    m = build_manifest(source, cites, resolver=resolver)
    hits = sum(1 for cid, span in m.items()
               if _resolves_correct(span, *truth[cid]))
    return hits / len(cites) if cites else 0.0


def _fp_rate(resolver, source, negatives) -> float:
    """Fraction of NEGATIVE citations (text absent from source) that spuriously
    resolve to SOME span."""
    n = len(negatives)
    if not n:
        return 0.0
    fp = sum(1 for q in negatives if resolver(source, q) is not None)
    return fp / n


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    rng = np.random.default_rng(SEED)
    source, clean_cites = _build_source_and_citations(rng)

    # Ground-truth source offset for every (unique) sentence.
    truth = {cid: (source.index(txt), source.index(txt) + len(txt))
             for cid, txt in clean_cites.items()}

    # Partition the present citations: 60% modeled OCR, 20% unmodeled OCR, 20% clean.
    ids = sorted(clean_cites)
    n = len(ids)
    n_mod = int(n * 0.6)
    n_unm = int(n * 0.2)
    mod_ids = ids[:n_mod]
    unm_ids = ids[n_mod:n_mod + n_unm]
    clean_ids = ids[n_mod + n_unm:]

    modeled = {cid: _modeled_ocr(clean_cites[cid], rng) for cid in mod_ids}
    unmodeled = {cid: _unmodeled_ocr(clean_cites[cid], rng) for cid in unm_ids}
    clean = {cid: clean_cites[cid] for cid in clean_ids}
    truth_mod = {cid: truth[cid] for cid in mod_ids}
    truth_unm = {cid: truth[cid] for cid in unm_ids}
    truth_clean = {cid: truth[cid] for cid in clean_ids}
    all_present = {**modeled, **unmodeled, **clean}

    # Negatives: novel sentences (both should reject) + short numeric citations
    # (digit-fold collision => Plumbline false-positive; naive rejects).
    novel_neg = [_novel_sentence(rng) for _ in range(100)]
    numeric_neg = ["100", "1", "5", "50", "1005", "515", "10", "55", "500", "105"]

    print("=" * 74)
    print(f"Plumbline A/B benchmark  (N={n} present citations, "
          f"{len(novel_neg) + len(numeric_neg)} negatives)")
    print("=" * 74)

    # ---- RECALL (per degradation class + overall) --------------------------
    r_mod_p = _recall(resolve, source, modeled, truth_mod)
    r_mod_n = _recall(naive_resolve, source, modeled, truth_mod)
    r_unm_p = _recall(resolve, source, unmodeled, truth_unm)
    r_unm_n = _recall(naive_resolve, source, unmodeled, truth_unm)
    r_cln_p = _recall(resolve, source, clean, truth_clean)
    r_cln_n = _recall(naive_resolve, source, clean, truth_clean)
    r_all_p = _recall(resolve, source, all_present, truth)
    r_all_n = _recall(naive_resolve, source, all_present, truth)

    print("\nRECALL (correct-span resolutions / citations)  [measured]")
    print(f"  {'class':<22}{'Plumbline':>12}{'naive':>12}{'gain':>10}")
    for label, p, nv in (
        ("modeled OCR (60%)", r_mod_p, r_mod_n),
        ("unmodeled OCR (20%)", r_unm_p, r_unm_n),
        ("clean control (20%)", r_cln_p, r_cln_n),
        ("ALL present", r_all_p, r_all_n),
    ):
        print(f"  {label:<22}{p:>12.3f}{nv:>12.3f}{p - nv:>+10.3f}")

    # ---- FALSE POSITIVES ---------------------------------------------------
    fp_novel_p = _fp_rate(resolve, source, novel_neg)
    fp_novel_n = _fp_rate(naive_resolve, source, novel_neg)
    fp_num_p = _fp_rate(resolve, source, numeric_neg)
    fp_num_n = _fp_rate(naive_resolve, source, numeric_neg)

    print("\nFALSE-POSITIVE RATE (spurious resolutions / negatives)  [measured]")
    print(f"  {'negatives':<22}{'Plumbline':>12}{'naive':>12}")
    print(f"  {'novel sentences':<22}{fp_novel_p:>12.3f}{fp_novel_n:>12.3f}")
    print(f"  {'numeric (digit-fold)':<22}{fp_num_p:>12.3f}{fp_num_n:>12.3f}")

    # ===================== MEASURED ASSERTIONS ==============================
    # 1) THE WEDGE: on legitimately OCR-normalized chunks, Plumbline resolves
    #    (near-)all, naive false-fails on (near-)all -> large measured gap.
    if r_mod_p < 0.99:
        _fail(f"Plumbline recall on modeled OCR too low: {r_mod_p:.3f}")
    if r_mod_n > 0.05:
        _fail(f"naive recall on modeled OCR unexpectedly high: {r_mod_n:.3f}")
    if (r_mod_p - r_mod_n) < 0.90:
        _fail(f"recall gap on modeled OCR too small: {r_mod_p - r_mod_n:.3f}")

    # 2) FAIR BASELINE: naive is NOT crippled — it resolves EVERY clean citation.
    if r_cln_n < 0.999:
        _fail(f"baseline unfair: naive missed clean citations (recall {r_cln_n:.3f})")
    if r_cln_p < 0.999:
        _fail(f"Plumbline regressed on clean citations (recall {r_cln_p:.3f})")

    # 3) OVERALL measured recall gain must be substantial.
    if (r_all_p - r_all_n) < 0.45:
        _fail(f"overall recall gain too small: {r_all_p - r_all_n:.3f}")

    # 4) HONESTY — not tautological: Plumbline does NOT solve unmodeled OCR noise.
    if r_unm_p >= 1.0:
        _fail("Plumbline recall on unmodeled OCR == 1.0 -> fixture is tautological")

    # 5) HONESTY — measured precision hole: digit-fold manufactures false
    #    provenance for Plumbline; the naive baseline does not.
    if fp_num_p <= 0.0:
        _fail("expected Plumbline digit-fold false positives were not observed")
    if fp_num_n != 0.0:
        _fail(f"naive baseline unexpectedly false-positive on numerics: {fp_num_n}")
    if fp_novel_p != 0.0 or fp_novel_n != 0.0:
        _fail(f"unexpected FP on novel negatives: plumb={fp_novel_p} naive={fp_novel_n}")

    print("\n" + "=" * 74)
    print("MEASURED SUMMARY")
    print(f"  wedge: modeled-OCR recall  Plumbline {r_mod_p:.3f}  vs  "
          f"naive {r_mod_n:.3f}   (+{r_mod_p - r_mod_n:.3f})")
    print(f"  overall recall             Plumbline {r_all_p:.3f}  vs  "
          f"naive {r_all_n:.3f}   (+{r_all_p - r_all_n:.3f})")
    print(f"  honest limit: unmodeled-OCR recall (Plumbline) = {r_unm_p:.3f} (<1.0)")
    print(f"  honest cost:  numeric-citation FP rate  Plumbline {fp_num_p:.3f}  "
          f"vs naive {fp_num_n:.3f}")
    print("=" * 74)
    print("PASS: measured recall wedge holds; honest limits measured + asserted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
