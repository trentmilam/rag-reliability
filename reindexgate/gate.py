"""ReindexGate: label-free, judge-free retrieval-regression CI gate.

Given an OLD and a NEW index built over the same corpus, ReindexGate:

  1. mines probe queries from the corpus (label-free, known-item convention);
  2. judges relevance with a fixed, SYSTEM-INDEPENDENT pseudo-relevance signal
     (reference hash-embedding similarity of a candidate doc to the query's
     source doc), with no human qrels and no LLM judge;
  3. computes a POOLING-BIAS-CORRECTED index-vs-index nDCG delta: relevance is
     judged on the UNION pool of both systems' results (Buttcher-2007 style), so
     a doc found only by NEW is judged fairly instead of assumed non-relevant;
  4. wraps the mean delta in a bootstrap confidence interval over queries;
  5. FAILS the gate when the CI is separated below zero by a margin (NEW is
     meaningfully worse). Also reports RBO churn as a diagnostic.

Scope: this is a COARSE-BUT-ROBUST regression DETECTOR. The relevance
signal is a heuristic, not truth; a green gate means "no regression this crude
detector can see," not "NEW is proven better."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedder import REF_DIM, REF_ORACLE_SEED, embed_batch
from corpus import Doc, Probe, build_corpus, mine_probes
from index import Index, IndexConfig, build_index, search_docs

REL_THRESH = 0.45  # reference-space cosine >= this => pseudo-relevant
# (measured separation on this corpus in the ORACLE hash family REF_ORACLE_SEED:
#  partner sims in [0.53,0.67], non-partner sims in [-0.31,0.38]; 0.45 cleanly
#  separates the two populations.)
TOP_K = 10
N_BOOT = 2000
FAIL_MARGIN = 0.01  # NEW must be worse by more than this, CI-separated, to FAIL


# ---------------------------------------------------------------- relevance ---

def reference_simmatrix(docs: list[Doc]) -> np.ndarray:
    """Fixed, system-independent doc-vs-doc similarity in the REF_DIM space.

    This is the pseudo-relevance oracle. It is built with REF_ORACLE_SEED, a
    hash family that is DISTINCT from the seed OLD and NEW use, so it is a
    genuinely third, near-orthogonal signal that depends only on the corpus,
    never on OLD or NEW, and cannot favour either index. It is a deterministic
    heuristic standing in for (absent) qrels.
    """
    ref = embed_batch([d.text for d in docs], dim=REF_DIM, weights_seed=REF_ORACLE_SEED)
    return ref @ ref.T


def _relevant_docs(simmat: np.ndarray, source_doc: int) -> set[int]:
    col = simmat[:, source_doc]
    return {int(i) for i in np.where(col >= REL_THRESH)[0]}


# --------------------------------------------------------------------- nDCG ---

def _ndcg(ranked: list[int], judged_relevant: set[int], k: int) -> float:
    """nDCG@k where only docs in `judged_relevant` count as gain 1.

    The ideal DCG uses the number of judged-relevant docs (capped at k), so both
    systems are scored against the same ideal for a given query.
    """
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in judged_relevant:
            dcg += 1.0 / np.log2(i + 2)
    r = min(len(judged_relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(r))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------- RBO ---

def _rbo(s: list[int], t: list[int], p: float = 0.9) -> float:
    """Truncated rank-biased overlap (churn diagnostic, not the gate signal)."""
    k = max(len(s), len(t))
    if k == 0:
        return 1.0
    ss: set[int] = set()
    tt: set[int] = set()
    total = 0.0
    weight = 0.0
    for d in range(1, k + 1):
        if d <= len(s):
            ss.add(s[d - 1])
        if d <= len(t):
            tt.add(t[d - 1])
        agreement = len(ss & tt) / d
        w = p ** (d - 1)
        total += w * agreement
        weight += w
    return float(total / weight) if weight > 0 else 1.0


# ------------------------------------------------------------------- verdict ---

@dataclass
class GateResult:
    mean_delta_corrected: float
    mean_delta_biased: float
    ci_low: float
    ci_high: float
    mean_rbo: float
    n_queries: int
    passed: bool

    def report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"ReindexGate verdict: {verdict}\n"
            f"  queries mined         : {self.n_queries}\n"
            f"  delta nDCG (corrected): {self.mean_delta_corrected:+.4f}\n"
            f"  delta nDCG (biased)   : {self.mean_delta_biased:+.4f}  "
            f"(uncorrected, for contrast)\n"
            f"  bootstrap 95% CI      : [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"  mean RBO (old vs new) : {self.mean_rbo:.4f}  (1.0 = identical ranking)\n"
            f"  fail margin           : delta CI-high must be < {-FAIL_MARGIN:+.4f} to FAIL"
        )


def run_gate(
    old_cfg: IndexConfig,
    new_cfg: IndexConfig,
    *,
    seed: int = 1234,
    k: int = TOP_K,
    n_boot: int = N_BOOT,
) -> GateResult:
    docs = build_corpus(seed=seed)
    probes: list[Probe] = mine_probes(docs)
    simmat = reference_simmatrix(docs)

    old = build_index(docs, old_cfg)
    new = build_index(docs, new_cfg)

    delta_c = np.empty(len(probes), dtype=np.float64)
    delta_b = np.empty(len(probes), dtype=np.float64)
    rbos = np.empty(len(probes), dtype=np.float64)

    for i, pr in enumerate(probes):
        old_ranked = search_docs(old, pr.query, k)
        new_ranked = search_docs(new, pr.query, k)
        true_rel = _relevant_docs(simmat, pr.source_doc)

        # Corrected: judge relevance on the UNION pool (debiased).
        union_pool = set(old_ranked) | set(new_ranked)
        judged_c = true_rel & union_pool
        ndcg_old_c = _ndcg(old_ranked, judged_c, k)
        ndcg_new_c = _ndcg(new_ranked, judged_c, k)
        delta_c[i] = ndcg_new_c - ndcg_old_c

        # Biased: judgments built ONLY from the incumbent (OLD) pool. NEW-only
        # relevant docs are therefore invisible, which penalises NEW unfairly. Kept
        # to demonstrate what the pooling-bias correction removes.
        biased_pool = set(old_ranked)
        judged_b = true_rel & biased_pool
        ndcg_old_b = _ndcg(old_ranked, judged_b, k)
        ndcg_new_b = _ndcg(new_ranked, judged_b, k)
        delta_b[i] = ndcg_new_b - ndcg_old_b

        rbos[i] = _rbo(old_ranked, new_ranked)

    # Bootstrap CI over queries on the corrected delta.
    rng = np.random.default_rng(seed)
    n = len(probes)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(delta_c[idx]))
    ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    passed = not (ci_high < -FAIL_MARGIN)

    return GateResult(
        mean_delta_corrected=float(np.mean(delta_c)),
        mean_delta_biased=float(np.mean(delta_b)),
        ci_low=ci_low,
        ci_high=ci_high,
        mean_rbo=float(np.mean(rbos)),
        n_queries=n,
        passed=passed,
    )


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ReindexGate: label-free reindex CI gate")
    ap.add_argument("--old-dim", type=int, default=REF_DIM)
    ap.add_argument("--new-dim", type=int, default=REF_DIM)
    ap.add_argument("--old-chunker", choices=["sentence", "whole"], default="sentence")
    ap.add_argument("--new-chunker", choices=["sentence", "whole"], default="sentence")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    old_cfg = IndexConfig(embed_dim=args.old_dim, chunker=args.old_chunker, name="OLD")
    new_cfg = IndexConfig(embed_dim=args.new_dim, chunker=args.new_chunker, name="NEW")
    res = run_gate(old_cfg, new_cfg, seed=args.seed)
    print(res.report())
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
