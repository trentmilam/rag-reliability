"""Leakprobe: corpus-scale MEASURED head-to-head vs the Presidio-style
blanket-redaction baseline.

Why this exists
---------------
The first-milestone `eval.py` proves the claim on ONE hand-built utility query
(recall 1.000 -> 0.000, chunk 400 padded to overtake a mask-diluted chunk 200).
A skeptic rightly discounts a single tuned point. This eval instead builds a
*distribution* of N topics/queries with a single uniform template (no per-query
rigging) and MEASURES, over the whole distribution:

  * recall(no-redaction): the unsafe upper bound
  * recall(minimal): Leakprobe, redact only reachable PII
  * recall(blanket): the incumbent, Presidio-style, redact ALL detected PII
  * over-redaction rate: (blanket_n - minimal_n)/blanket_n, spans the
                             incumbent masks that reachability shows to be needless

...as a CURVE over k in {1,3,5,10} (k is both the reachability top-k that
defines the minimal set and the recall@k of the utility measurement).

The fair baseline
-----------------
"Blanket" == what a competent engineer using Microsoft Presidio actually does:
detect every PII entity and anonymize each occurrence. It is NOT a crippled
strawman: it uses the exact same detectors and mask as Leakprobe; the ONLY
difference is that Leakprobe first asks "can a retriever even reach this span?"
and spares the unreachable ones.

The corpus
----------
`baseline_corpus()` emits N topics under one template:

  * half the topics carry an email in DISTINCTIVE context  -> reachable  (a real
    leak both strategies must redact);
  * half carry a name in BOILERPLATE context, drowned by shared decoys
    -> UNREACHABLE (redacting it is pure over-redaction);
  * every topic also has a longer, PII-free same-topic "appendix" competitor and
    a utility query (the topic's own tokens, gold == the topic's primary chunk).

Masking dilutes the gold chunk's vector (the mask token adds mass to the norm),
so blanket masking of an *unreachable*-PII gold can knock it below its unmasked
competitor, recall lost for no safety benefit. Minimal leaves those intact.
The reachable-PII golds are masked by BOTH strategies, so any recall they lose
is charged to both equally; the measured gap comes purely from over-redaction.

Deterministic: filler is drawn from a fixed vocabulary via a fixed-seed
`np.random.default_rng`; everything else is pure logic. exit 0 on pass.
"""

from __future__ import annotations

import sys

import numpy as np

from leakprobe import (
    Chunk,
    apply_redaction,
    audit,
    over_redaction_rate,
    recall_at_k,
)

KS = (1, 3, 5, 10)
N_TOPICS = 24  # 12 reachable + 12 unreachable
SEED = 0xC0FFEE

DEBUG = "--debug" in sys.argv

_BOILER = "confidential document prepared on behalf of {who} per standard policy footer"

# A neutral filler vocabulary for the same-topic "appendix" competitor chunks.
_FILLER_VOCAB = (
    "appendix annex exhibit schedule addendum supplement overview highlights "
    "tables figures notes references glossary summary section paragraph clause "
    "provision recital preamble narrative discussion analysis commentary "
    "background context rationale methodology framework guidance"
).split()

_NAMES = ("jane doe", "john roe", "mary major", "richard miles")


def baseline_corpus() -> tuple[list[Chunk], list[tuple[str, int]], dict]:
    """Return (chunks, utility_queries, meta).

    One uniform template across all topics, NOT per-query tuning. The two
    free parameters (gold "other"-token count, competitor filler length) were
    picked ONCE by measuring margins against the real hash embedder (not
    per-topic hand-tuning) so the unredacted baseline is a genuinely easy
    retrieval task (gold beats its same-topic competitor almost everywhere),
    a prerequisite for a fair redaction-cost comparison. A short, compact gold
    chunk keeps cosine high (fewer non-topic tokens diluting the norm); the
    longer competitor is a genuine same-topic rival that only overtakes gold
    once gold's own vector is diluted by mask tokens."""
    rng = np.random.default_rng(SEED)
    chunks: list[Chunk] = []
    utility: list[tuple[str, int]] = []
    reachable_topics = 0

    # shared boilerplate decoys: drown the boilerplate-context names so those
    # name spans are UNREACHABLE (exactly the Presidio over-redaction target).
    decoy_id = 90000
    for grp in ("operations", "audit", "vendor", "review", "finance", "legal"):
        chunks.append(Chunk(decoy_id, _BOILER.format(who=f"the {grp} group") + f" notice {grp}"))
        decoy_id += 1

    for i in range(N_TOPICS):
        t0, t1, t2, t3 = (f"tk{i}a", f"tk{i}b", f"tk{i}c", f"tk{i}d")
        gold_id = 1000 + i
        comp_id = 5000 + i
        query = f"{t0} {t1} {t2} {t3}"

        if i % 2 == 0:
            # REACHABLE: email sits amid the topic's own distinctive tokens, so
            # its elicited query is unique -> it surfaces -> a real leak.
            reachable_topics += 1
            gold_text = f"{t0} {t1} {t2} {t3} contact email pii{i}user@corp.example"
        else:
            # UNREACHABLE: a gazetteer name buried in boilerplate, drowned by the
            # shared decoys above -> reachability=False -> redacting it is waste.
            name = _NAMES[i % len(_NAMES)]
            gold_text = f"{t0} {t1} {t2} {t3} quarterly review summary " + _BOILER.format(who=name)

        chunks.append(Chunk(gold_id, gold_text))

        # Same-topic competitor: a longer, PII-free appendix. Shares all 4 topic
        # tokens, so it is a genuine retrieval rival; longer => lower baseline
        # cosine than the compact gold, i.e. gold wins UNTIL gold is diluted.
        filler = " ".join(rng.choice(_FILLER_VOCAB, size=18, replace=False))
        chunks.append(Chunk(comp_id, f"{t0} {t1} {t2} {t3} {filler}"))

        utility.append((query, gold_id))

    meta = {"reachable_topics": reachable_topics, "unreachable_topics": N_TOPICS - reachable_topics}
    return chunks, utility, meta


def measure() -> dict:
    """Run the three strategies over the corpus for every k; return a results
    table keyed by k."""
    chunks, utility, meta = baseline_corpus()
    table: dict[int, dict] = {}
    for k in KS:
        res = audit(chunks, k=k)
        redacted_min = apply_redaction(chunks, res.minimal_redaction)
        redacted_blanket = apply_redaction(chunks, res.blanket_redaction)
        table[k] = {
            "recall_none": recall_at_k(chunks, utility, k),
            "recall_minimal": recall_at_k(redacted_min, utility, k),
            "recall_blanket": recall_at_k(redacted_blanket, utility, k),
            "over_redaction": over_redaction_rate(res),
            "blanket_n": len(res.blanket_redaction),
            "minimal_n": len(res.minimal_redaction),
            "reachable_n": len(res.reachable_spans),
        }
    return {"table": table, "meta": meta, "n_queries": len(utility)}


def run() -> int:
    failures: list[str] = []
    out = measure()
    table, meta, n_q = out["table"], out["meta"], out["n_queries"]

    print("Leakprobe -- MEASURED head-to-head vs Presidio-style blanket mask")
    print(f"  corpus: {N_TOPICS} topics ({meta['reachable_topics']} reachable-PII, "
          f"{meta['unreachable_topics']} unreachable-PII), {n_q} utility queries")
    print(f"  fair baseline = blanket = redact EVERY detected span (same detectors + mask)")
    print()
    print("   k | recall none | recall minimal | recall blanket | over-redaction | spans(min/blanket)")
    print("  ---+-------------+----------------+----------------+----------------+-------------------")
    for k in KS:
        r = table[k]
        print(f"  {k:2d} |    {r['recall_none']:.3f}    |     {r['recall_minimal']:.3f}      "
              f"|     {r['recall_blanket']:.3f}      |     {r['over_redaction']:.3f}      "
              f"|   {r['minimal_n']:2d} / {r['blanket_n']:2d}")

    # ---- MEASURED assertions (the portfolio proof) -----------------------
    for k in KS:
        r = table[k]
        # (1) monotonic ordering: fewer masks can never help less than more masks.
        if not (r["recall_none"] + 1e-9 >= r["recall_minimal"] >= r["recall_blanket"] - 1e-9):
            failures.append(
                f"C-ord@k={k}: expected none>=minimal>=blanket, got "
                f"{r['recall_none']:.3f}/{r['recall_minimal']:.3f}/{r['recall_blanket']:.3f}")

    # (2) the gap, MEASURED: at the strict k=1 the incumbent blanket mask
    #     loses recall that Leakprobe's minimal set preserves.
    r1 = table[1]
    if not (r1["recall_minimal"] > r1["recall_blanket"]):
        failures.append(
            f"C-gap@k=1: minimal ({r1['recall_minimal']:.3f}) not > blanket "
            f"({r1['recall_blanket']:.3f}) -- no measured recall advantage")

    # (3) cost claim: minimal is not free. It still redacts every
    #     genuinely reachable span, which costs some real recall (those golds
    #     are masked identically under minimal AND blanket). What minimal
    #     avoids is the WASTED cost of masking unreachable PII too. So the
    #     defensible claim is comparative, not absolute: minimal's recall loss
    #     vs the unsafe no-redaction baseline must be strictly SMALLER than
    #     blanket's loss, not that minimal has zero cost.
    loss_min = r1["recall_none"] - r1["recall_minimal"]
    loss_blank = r1["recall_none"] - r1["recall_blanket"]
    if not (loss_min < loss_blank):
        failures.append(
            f"C-cost@k=1: minimal's recall loss ({loss_min:.3f}) not < blanket's "
            f"loss ({loss_blank:.3f}) vs the no-redaction baseline")

    # (4) over-redaction is a real, positive fraction (incumbent masks needlessly).
    if not (r1["over_redaction"] > 0.0):
        failures.append(f"C-overredact@k=1: over-redaction rate {r1['over_redaction']:.3f} not > 0")

    # (5) determinism: re-measure, expect byte-identical table.
    out2 = measure()
    if out2["table"] != table:
        failures.append("C-determinism: re-measured table differs (non-deterministic)")

    print()
    print(f"  MEASURED wedge @k=1: minimal recall {r1['recall_minimal']:.3f} vs "
          f"blanket {r1['recall_blanket']:.3f}  (+{r1['recall_minimal'] - r1['recall_blanket']:.3f})")
    print(f"  recall LOST vs no-redaction baseline @k=1: minimal={loss_min:.3f}  blanket={loss_blank:.3f}")
    print(f"  over-redaction rate @k=1: {r1['over_redaction']:.3f} "
          f"({r1['blanket_n'] - r1['minimal_n']} of {r1['blanket_n']} detected spans spared)")

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS (measured recall advantage + real over-redaction, deterministic)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
