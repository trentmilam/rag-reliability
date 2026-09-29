"""Legigate eval: RED/GREEN self-test of the FIRST MILESTONE.

    python legigate/eval.py

GREEN: clean single-column prose + a well-formed table both PASS the gate.
RED:   the same content after real fault injection (two-column interleave
       and a delimiter-collapsed table) is QUARANTINED in both cases, each by
       the real mechanism (reading-order margin inversion / table structure
       loss), not by any hard-coded verdict.

Also asserts determinism: scoring is byte-identical across two runs, including
on a batch of RNG-generated chunks.

Exit 0 iff every check passes.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import fixtures  # noqa: E402
from baseline import IncumbentQualityFilter  # noqa: E402
from corpus import labeled_corpus  # noqa: E402
from legigate import (  # noqa: E402
    ORDER_THRESHOLD,
    SEED,
    TABLE_THRESHOLD,
    score_chunk,
)


def _prf(preds, labels):
    """precision, recall over binary preds (1 == quarantine == positive)."""
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return prec, rec, tp, fp, fn, tn


def measure_ab(checks):
    """MEASURED head-to-head: Legigate vs a FAIR incumbent quality filter.

    Builds a labeled legibility corpus (clean vs mechanically-corrupted chunks,
    incl. the HARD same-topic interleave and uniform legal boilerplate) and
    reports precision/recall for BOTH the gate and a competent commodity
    perplexity+heuristics filter. The wedge is proven, not asserted: the
    incumbent is calibrated to keep ALL clean chunks and it still misses the
    structural corruption (interleave / collapsed tables) that Legigate catches,
    while (control) it DOES catch genuine gibberish/mojibake, so it is a real
    filter, not a strawman.
    """
    items = labeled_corpus()
    labels = [y for _, _, y in items]
    clean_texts = [t for _, t, y in items if y == 0]

    # Legigate predictions + clean/corrupt score distributions.
    leg_preds, clean_scores, corrupt_scores = [], [], []
    for cid, text, y in items:
        v = score_chunk(text, cid)
        pred = 0 if v.legible else 1
        leg_preds.append(pred)
        applied = min(v.order_score if v.order_applicable else 1.0,
                      v.table_score if v.table_applicable else 1.0)
        (corrupt_scores if y == 1 else clean_scores).append(applied)

    # Incumbent: trained + calibrated on the clean half (most favorable to it).
    inc = IncumbentQualityFilter().fit(clean_texts)
    inc_preds = [1 if inc.quarantine(t) else 0 for _, t, _ in items]

    leg_p, leg_r, ltp, lfp, lfn, ltn = _prf(leg_preds, labels)
    inc_p, inc_r, itp, ifp, ifn, itn = _prf(inc_preds, labels)

    # Control: the incumbent must genuinely catch gibberish (proves it is fair).
    junk = [
        "The cafÃ© Ã©tÃ© rÃ©sumÃ© garbled text.\n"
        "More Ã¼ mojibake lines Ã± follow Ã§ along here now.",
        "xkq zwvbn qptlm zxcvbn plmqwe rtyuio\nasdfgh jklzxc vbnmqw ertyui opasdf ghjklz\n"
        "qwzxpl mkvbnr tygbhu jicder swaqzx cvbnml",
        "### @@@ %%% ^^^ &&& *** ((( ))) ??? !!!\n<<< >>> {{{ }}} [[[ ]]] +++ === ~~~ |||",
        "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii\nkkkk llll mmmm nnnn oooo pppp qqqq",
    ]
    junk_caught = sum(1 for t in junk if inc.quarantine(t))

    clean_min = min(clean_scores)
    corrupt_max = max(corrupt_scores)
    separation = clean_min - corrupt_max  # >0 => clean & corrupt fully separated

    # --- assertions: the gate wins, and it wins on a FAIR baseline ---------
    checks["ab_legigate_recall_high"] = leg_r >= 0.90
    checks["ab_legigate_precision_high"] = leg_p >= 0.90
    checks["ab_incumbent_blind_to_structure"] = inc_r <= 0.10
    checks["ab_recall_gap_large"] = (leg_r - inc_r) >= 0.80
    checks["ab_incumbent_is_fair_catches_gibberish"] = junk_caught >= 3
    checks["ab_incumbent_no_clean_false_positive"] = ifp == 0
    checks["ab_scores_separated"] = separation > 0.0

    print("\n=== MEASURED A/B: Legigate vs fair incumbent quality filter ===")
    print(f"labeled corpus: {len(items)} chunks "
          f"({labels.count(0)} clean, {labels.count(1)} corrupted; "
          f"incl. same-topic interleave + legal boilerplate)")
    print(f"  Legigate : precision {leg_p:.3f}  recall {leg_r:.3f}  "
          f"(tp={ltp} fp={lfp} fn={lfn} tn={ltn})")
    print(f"  Incumbent: precision {inc_p:.3f}  recall {inc_r:.3f}  "
          f"(tp={itp} fp={ifp} fn={ifn} tn={itn})")
    print(f"  -> recall gap on structural corruption: "
          f"{leg_r:.3f} vs {inc_r:.3f}  (Legigate catches {ltp}/{ltp + lfn}, "
          f"incumbent {itp}/{itp + ifn})")
    print(f"  control: incumbent catches {junk_caught}/4 genuine gibberish/"
          f"mojibake chunks (=> it is a real filter, not a strawman)")
    print(f"  score separation: clean_min={clean_min:.3f} > "
          f"corrupt_max={corrupt_max:.3f}  (margin {separation:+.3f})")
    return checks


def main() -> int:
    checks = {}

    # --- GREEN: clean content passes ---------------------------------------
    clean_prose = score_chunk(fixtures.clean_prose(), "clean_prose")
    clean_table = score_chunk(fixtures.clean_table(), "clean_table")
    checks["green_clean_prose_legible"] = clean_prose.legible is True
    checks["green_clean_table_legible"] = clean_table.legible is True

    # --- RED: injected faults are caught -----------------------------------
    interleaved = score_chunk(fixtures.interleaved_prose(), "interleaved_prose")
    collapsed = score_chunk(fixtures.collapsed_table(), "collapsed_table")
    checks["red_interleave_quarantined"] = interleaved.legible is False
    checks["red_collapsed_quarantined"] = collapsed.legible is False

    # --- the RED must be caught by the RIGHT mechanism ---------------------
    checks["red_interleave_via_order"] = (
        interleaved.order_applicable
        and interleaved.order_score < ORDER_THRESHOLD
        and interleaved.order_margin < 0.0        # genuine coherence inversion
    )
    checks["red_collapsed_via_table"] = (
        collapsed.table_applicable
        and collapsed.table_score < TABLE_THRESHOLD
    )

    # --- guard against rigging: clean scores are actually ABOVE threshold
    checks["green_clean_prose_order_ok"] = (
        clean_prose.order_applicable
        and clean_prose.order_score >= ORDER_THRESHOLD
        and clean_prose.order_margin > 0.0
    )
    checks["green_clean_table_struct_ok"] = (
        clean_table.table_applicable
        and clean_table.table_score >= TABLE_THRESHOLD
    )

    # --- signals stay in their lane (no cross-penalty) ---------------------
    checks["prose_not_flagged_as_table"] = clean_prose.table_applicable is False
    checks["table_not_flagged_as_order"] = clean_table.order_applicable is False

    # --- determinism: re-run identical --------------------------------------
    again = score_chunk(fixtures.interleaved_prose(), "interleaved_prose")
    checks["deterministic_repeat"] = (
        abs(again.order_score - interleaved.order_score) < 1e-12
        and abs(again.order_margin - interleaved.order_margin) < 1e-12
    )

    # determinism on RNG-generated inputs (uses the fixed SEED)
    rng = np.random.default_rng(SEED)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    batch = []
    for _ in range(20):
        nlines = int(rng.integers(4, 8))
        lines = []
        for _ in range(nlines):
            k = int(rng.integers(4, 9))
            lines.append(" ".join(rng.choice(vocab, size=k)))
        batch.append("\n".join(lines))
    r1 = [score_chunk(t).order_score for t in batch]
    r2 = [score_chunk(t).order_score for t in batch]
    checks["deterministic_batch"] = all(abs(a - b) < 1e-12 for a, b in zip(r1, r2))

    # --- MEASURED A/B vs a fair incumbent (the credibility lever) -----------
    measure_ab(checks)

    # --- report -------------------------------------------------------------
    print("=== Legigate eval (measured) ===")
    print(f"GREEN clean_prose : order={clean_prose.order_score:.3f} "
          f"(margin {clean_prose.order_margin:+.3f})  table=n/a  -> "
          f"{'KEEP' if clean_prose.legible else 'QUARANTINE'}")
    print(f"GREEN clean_table : order=n/a  table={clean_table.table_score:.3f}  -> "
          f"{'KEEP' if clean_table.legible else 'QUARANTINE'}")
    print(f"RED   interleave  : order={interleaved.order_score:.3f} "
          f"(margin {interleaved.order_margin:+.3f})  -> "
          f"{'KEEP' if interleaved.legible else 'QUARANTINE'}")
    for r in interleaved.reasons:
        print(f"                   reason: {r}")
    print(f"RED   collapsed   : table={collapsed.table_score:.3f} "
          f"(multicell {collapsed.table_detail.get('frac_multicell')})  -> "
          f"{'KEEP' if collapsed.legible else 'QUARANTINE'}")
    for r in collapsed.reasons:
        print(f"                   reason: {r}")
    print(f"\nthresholds: order>={ORDER_THRESHOLD}  table>={TABLE_THRESHOLD}")
    print("\n--- checks ---")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")

    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
