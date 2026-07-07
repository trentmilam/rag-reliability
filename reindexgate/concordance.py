"""Concordance-with-gold experiment for ReindexGate.

Does the LABEL-FREE gate verdict agree with the REAL nDCG delta computed from
GOLD relevance labels -- and does it beat the naive incumbent ("just alert on
RBO churn")? This module supplies the machinery; ``eval.py``'s ``[CONCORDANCE]``
section runs it and asserts the measured agreement.

GOLD labels are the STRUCTURAL ground truth of the fixture: a probe mined from
document ``d`` is relevant to ``d`` and to ``d``'s paraphrase partner (the other
doc with the same ``pair_id``) -- and to nothing else. These labels are
independent of BOTH the indexes under test AND the gate's own pseudo-relevance
oracle, which must RECOVER them without ever seeing them. Real nDCG on gold is
exactly what you could only compute WITH qrels; ReindexGate claims to reach the
same PASS/FAIL verdict WITHOUT them. This experiment measures how often that
claim holds, and contrasts it with RBO-alone.

Everything is deterministic, offline, numpy + stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corpus import Doc, Probe, build_corpus, mine_probes
from index import IndexConfig, build_index, search_docs
from gate import _ndcg, _rbo, reference_simmatrix, _relevant_docs, run_gate, GateResult

# A regression is "real" (on gold) / "flagged" (by a detector) when the signed
# gold delta drops below this margin. Same magnitude as the gate's FAIL_MARGIN
# so the two detectors are compared on an equal footing.
GOLD_TAU = 0.01
# Naive incumbent: flag a regression when rankings churned past this RBO floor.
# A no-op reindex has RBO == 1.0; a competent engineer picks a floor a little
# below 1.0 (here 10% churn) so genuinely-identical reindexes are not flagged.
# This is a REASONABLE RBO threshold, not a strawman -- it is exactly what "just
# diff the rankings and alarm on a big drop" looks like in practice.
RBO_TAU = 0.90


def gold_qrels(docs: list[Doc]) -> dict[int, set[int]]:
    """Structural GOLD relevance: doc_id -> {doc_id, paraphrase-partner id}.

    Ground truth of the fixture, independent of any embedder or the gate oracle.
    """
    by_pair: dict[int, list[int]] = {}
    for d in docs:
        by_pair.setdefault(d.pair_id, []).append(d.doc_id)
    qrels: dict[int, set[int]] = {}
    for d in docs:
        qrels[d.doc_id] = set(by_pair[d.pair_id])  # {self, partner(s)}
    return qrels


@dataclass
class ScenarioMeasurement:
    name: str
    gate: GateResult          # label-free ReindexGate result (official verdict)
    gold_delta: float         # mean signed REAL nDCG delta on GOLD labels
    mean_rbo: float           # ranking churn (naive-baseline signal)

    # --- derived binary regression decisions (True == "regression") ----------
    @property
    def gold_regression(self) -> bool:
        return self.gold_delta < -GOLD_TAU

    @property
    def gate_regression(self) -> bool:
        return not self.gate.passed

    @property
    def rbo_regression(self) -> bool:
        return self.mean_rbo < RBO_TAU


def measure_scenario(
    name: str,
    old_cfg: IndexConfig,
    new_cfg: IndexConfig,
    *,
    seed: int = 1234,
    k: int = 10,
) -> ScenarioMeasurement:
    """Run one reindex scenario and measure gate verdict, GOLD delta, and RBO."""
    # Official label-free gate verdict (bootstrap CI + PASS/FAIL).
    gate = run_gate(old_cfg, new_cfg, seed=seed, k=k)

    # REAL nDCG delta on GOLD labels over the same mined probes.
    docs = build_corpus(seed=seed)
    probes: list[Probe] = mine_probes(docs)
    qrels = gold_qrels(docs)
    old = build_index(docs, old_cfg)
    new = build_index(docs, new_cfg)

    deltas = np.empty(len(probes), dtype=np.float64)
    for i, pr in enumerate(probes):
        rel = qrels[pr.source_doc]
        old_ranked = search_docs(old, pr.query, k)
        new_ranked = search_docs(new, pr.query, k)
        deltas[i] = _ndcg(new_ranked, rel, k) - _ndcg(old_ranked, rel, k)

    return ScenarioMeasurement(
        name=name,
        gate=gate,
        gold_delta=float(np.mean(deltas)),
        mean_rbo=gate.mean_rbo,
    )


# --------------------------------------------------------------- statistics ---

def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0  # both raters constant and agree -> perfect
    return float((po - pe) / (1 - pe))


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall tau-b over paired samples (small n, O(n^2), no scipy)."""
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return 0.0
    return float((conc - disc) / denom)


def agreement_rate(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / n
