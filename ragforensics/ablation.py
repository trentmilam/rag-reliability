"""MEASURED ablation: auto-calibrated leak threshold vs a fixed global cutoff.

The ONE net-new wedge in RAGForensics is `calibrate_leak_threshold`: instead of a
magic global leak cutoff, it derives the bar from the corpus's own held-out
context-removal-invariance distribution. The README/eval already prove that bar is
*data-derived* (monotonic in the corpus baseline, strictly interior). They do NOT
prove it is *better* than the obvious thing it replaces. This module measures that.

Fair baseline (NOT a strawman): a fixed 0.5 invariance cutoff is exactly the
competent default a reasonable engineer ships absent any calibration -- the natural
midpoint of the [0,1] invariance signal. Both gates see the SAME label-free
invariance computed by the REAL `context_removal_invariance` over the SAME opaque
`answer_fn`. Only the threshold differs; nothing about the baseline is crippled.

The scenario is the exact one the wedge claims to handle: a corpus whose *normal*
traffic is LEGITIMATELY context-independent (many queries the model can answer
without the docs, and does so consistently *while still agreeing with the docs*).
On such a corpus a fixed cutoff false-flags that legitimate traffic as "leaks",
because it has no idea the corpus baseline is high. The auto-calibrated bar rises
to the corpus's own baseline and only flags queries that are ABNORMALLY invariant
relative to it -- the genuine parametric leaks.

Everything is SIMULATED and clearly labelled (offline, no network, no LLM, no GPU).
The multi-token answerer produces GRADED invariance (unlike the near-binary
single-token demo answerer), which is what a real multi-sample invariance estimate
looks like. The forensics code treats it as an opaque callable and never sees the
hidden ground-truth categories below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from answerer import IDK, Passage, Query
from forensics import calibrate_leak_threshold, context_removal_invariance

SEED = 20260704
FIXED_CUTOFF = 0.5  # the naive global baseline the wedge is measured against

# hidden ground-truth categories -- the forensics gates NEVER see these.
LEAK = "leak"          # answers from parametric memory, ignores context -> IS a leak
LEGIT_CI = "legit_ci"  # legitimately answerable w/o docs, agrees w/ docs -> NOT a leak
GROUNDED = "grounded"  # context-dependent grounded answer            -> NOT a leak


@dataclass
class _Spec:
    qid: str
    cat: str
    full: tuple[str, ...]   # answer WITH retrieved context
    empty: tuple[str, ...]  # answer WITH context stripped


class AblationAnswerer:
    """Opaque (to forensics) multi-token simulated RAG stack.

    Returns the pre-baked `full` answer when given context and the `empty` answer
    when context is stripped -- so `context_removal_invariance` measures a genuine
    graded token overlap. The gates read only this string; the category is hidden.
    """

    def __init__(self, specs: dict[str, _Spec]):
        self._specs = specs

    def answer(self, query: Query, passages: list[Passage]) -> str:
        spec = self._specs[query.id]
        return " ".join(spec.full if passages else spec.empty)


def _leak_clean(rng, qid: str) -> _Spec:
    # a true parametric leak: long verbatim answer, (almost) unchanged when context
    # is removed -> invariance >= 0.9. The defining behavior of a leak.
    n = int(rng.integers(9, 13))
    toks = tuple(f"{qid}_p{j}" for j in range(n))
    extra = int(rng.integers(0, 2))
    full = toks + ((f"{qid}_c0",) if extra else ())
    return _Spec(qid, LEAK, full, toks)


def _leak_hard(rng, qid: str) -> _Spec:
    # a genuinely BORDERLINE leak: short answer that partly incorporates context ->
    # invariance ~0.67-0.83, deliberately in the overlap zone. Honest hard case.
    s = int(rng.integers(4, 6))
    d = int(rng.integers(1, 3))
    toks = tuple(f"{qid}_p{j}" for j in range(s))
    full = toks + tuple(f"{qid}_c{j}" for j in range(d))
    return _Spec(qid, LEAK, full, toks)


def _legit_ci(rng, qid: str) -> _Spec:
    # legitimately answerable without docs, but the model STILL folds in some
    # retrieved detail when context is present -> invariance ~0.43-0.71 (below a true
    # leak). Agrees with the docs; flagging it would be a false positive.
    s = int(rng.integers(3, 6))
    d = int(rng.integers(2, 5))
    shared = tuple(f"{qid}_s{j}" for j in range(s))
    full = shared + tuple(f"{qid}_d{j}" for j in range(d))
    return _Spec(qid, LEGIT_CI, full, shared)


def _grounded(rng, qid: str) -> _Spec:
    # context-dependent: answers from the docs with context, abstains without ->
    # invariance 0.
    n = int(rng.integers(3, 6))
    doc = tuple(f"{qid}_doc{j}" for j in range(n))
    return _Spec(qid, GROUNDED, doc, (IDK,))


@dataclass
class AblationBundle:
    answer_fn: object
    calib_queries: list[Query]
    calib_contexts: dict[str, list[Passage]]
    test_specs: dict[str, _Spec]


def build(rng: np.random.Generator) -> AblationBundle:
    """Context-independent corpus: normal (calibration) traffic is dominated by
    legitimately context-independent queries, so the corpus baseline invariance is
    high. Held-out test set carries hidden leak labels."""
    specs: dict[str, _Spec] = {}
    ctx1 = [Passage(id="ctx", text="retrieved context passage", answer_token=None)]

    # --- calibration corpus (the gate's view of "normal" traffic) ---
    # mostly legitimately context-independent + a few grounded => HIGH baseline.
    calib_queries: list[Query] = []
    calib_contexts: dict[str, list[Passage]] = {}
    for i in range(15):
        qid = f"calib_ci_{i}"
        specs[qid] = _legit_ci(rng, qid)
        calib_queries.append(Query(id=qid, text=qid))
        calib_contexts[qid] = ctx1
    for i in range(5):
        qid = f"calib_gr_{i}"
        specs[qid] = _grounded(rng, qid)
        calib_queries.append(Query(id=qid, text=qid))
        calib_contexts[qid] = ctx1

    # --- held-out test set (hidden labels) ---
    test_specs: dict[str, _Spec] = {}
    for i in range(12):
        qid = f"test_leak_{i}"
        test_specs[qid] = _leak_clean(rng, qid)
    for i in range(3):
        qid = f"test_leakhard_{i}"
        test_specs[qid] = _leak_hard(rng, qid)
    for i in range(15):
        qid = f"test_ci_{i}"
        test_specs[qid] = _legit_ci(rng, qid)
    for i in range(10):
        qid = f"test_gr_{i}"
        test_specs[qid] = _grounded(rng, qid)

    specs.update(test_specs)
    return AblationBundle(
        answer_fn=AblationAnswerer(specs).answer,
        calib_queries=calib_queries,
        calib_contexts=calib_contexts,
        test_specs=test_specs,
    )


@dataclass(frozen=True)
class GateScore:
    name: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _score_gate(name: str, threshold: float, invariances: dict[str, float],
                test_specs: dict[str, _Spec]) -> GateScore:
    tp = fp = fn = tn = 0
    for qid, spec in test_specs.items():
        flagged = invariances[qid] >= threshold
        is_leak = spec.cat == LEAK
        if is_leak and flagged:
            tp += 1
        elif is_leak and not flagged:
            fn += 1
        elif not is_leak and flagged:
            fp += 1
        else:
            tn += 1
    return GateScore(name, threshold, tp, fp, fn, tn)


def run_ablation(rng: np.random.Generator | None = None):
    """Returns (fixed_gate, auto_gate, auto_threshold, invariances)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    b = build(rng)
    ctx1 = [Passage(id="ctx", text="retrieved context passage", answer_token=None)]

    # SAME invariance signal for both gates -- the REAL forensics mechanism.
    invariances = {
        qid: context_removal_invariance(b.answer_fn, Query(id=qid, text=qid), ctx1)
        for qid in b.test_specs
    }

    auto_thr = calibrate_leak_threshold(b.answer_fn, b.calib_queries, b.calib_contexts).threshold

    fixed = _score_gate("fixed-0.5", FIXED_CUTOFF, invariances, b.test_specs)
    auto = _score_gate("auto-calibrated", auto_thr, invariances, b.test_specs)
    return fixed, auto, auto_thr, invariances


def _fmt(g: GateScore) -> str:
    return (f"  {g.name:16s} thr={g.threshold:.3f}  "
            f"TP={g.tp:2d} FP={g.fp:2d} FN={g.fn:2d} TN={g.tn:2d}  "
            f"prec={g.precision:.3f} rec={g.recall:.3f} F1={g.f1:.3f}")


def main() -> int:
    fixed, auto, auto_thr, _ = run_ablation()
    print("=== RAGForensics ablation: auto-calibrated vs fixed global cutoff ===")
    print("corpus: normal traffic is LEGITIMATELY context-independent (high baseline)")
    print(f"auto-calibrated threshold (data-derived) = {auto_thr:.3f}  "
          f"vs naive fixed = {FIXED_CUTOFF}")
    print()
    print(_fmt(fixed))
    print(_fmt(auto))
    print()
    print(f"false-positive reduction: {fixed.fp} -> {auto.fp}  "
          f"(fixed over-flags legitimate context-independent traffic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
