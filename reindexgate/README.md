# ReindexGate

Label-free retrieval-regression CI gate.

- Compares an OLD and a NEW retrieval index.
- No relevance labels (qrels), no LLM judge.
- Fails the build when the new index regresses. Passes a no-op reindex.

## Method

1. Probe queries mined from the corpus (known-item / inverse-cloze).
2. Pseudo-relevance oracle: reference-embedding similarity to the query's source document.
   Distinct hash family (`REF_ORACLE_SEED`, near-orthogonal to `REF_SEED`).
   Never derived from OLD or NEW.
   Checked by the `[INDEPENDENCE]` checks in `eval.py`.
3. Pooling-bias correction: relevance judged on the union of both systems' results.
4. Bootstrap CI over queries. Fails only when the CI is below zero by a margin.

```
corpus (synthetic)            OLD index          NEW index
      │                     (config A)         (config B)
      ├── mine probe queries (label-free, known-item)
      ├── reference sim-matrix  ─────────────►  pseudo-relevance oracle
      │                                         (system-independent, judge-free)
      └── per query:
            retrieve top-k docs from OLD and NEW
            pool = OLD∪NEW  ── judge on the pool ──►  debiased nDCG@k each side
            delta = nDCG(NEW) − nDCG(OLD)
      bootstrap CI over queries  ──►  FAIL iff CI-high < −margin
```

## Simulated

Self-contained. Vendored pure-python hashing embedder (`embedder.py`, BLAKE2b feature-hashing, dim 256, L2-normalized, cosine). No pip install, no network, no data files.

- Dim-truncation reindex: smaller hash dimensionality.
- Worse chunker: whole-document vs fine.
- Different embedding model: different hash seed.

## Red / green (`eval.py`)

| scenario | expectation | measured (seed 1234) |
|---|---|---|
| **GREEN** no-op reindex | gate PASSES, CI straddles 0 | delta `+0.0000`, CI `[+0.0000,+0.0000]`, RBO `1.000` → **PASS** |
| **RED** dim-truncation 256→16 | gate FAILS, CI separated < 0 | delta `−0.1363`, CI `[−0.1915,−0.0899]`, RBO `0.519` → **FAIL** |
| **DEBIAS** weak OLD(24) vs strong NEW(256), k=2 | correction is live | corrected `+0.1376` vs incumbent-pool `+0.0123`, **gap `+0.1253`** → **PASS** |

## Concordance with GOLD labels (`eval.py` `[CONCORDANCE]`)

- GOLD labels: a probe mined from doc `d` is relevant to `d` and its paraphrase partner (same `pair_id`).
- Independent of both indexes and the gate's oracle.
- 10 reindex scenarios: no-ops, dim-truncation regressions, improvements, a worse chunker, a benign reseed.
- Compared: GOLD nDCG delta, ReindexGate verdict, naive alarm (`RBO < 0.90`).

Measured (seed 1234, 10 scenarios):

| detector (label-free) | accuracy vs GOLD | Cohen κ vs GOLD |
|---|---|---|
| **ReindexGate** | **1.000 (10/10)** | **+1.000** |
| naive RBO-churn alarm | 0.600 (6/10) | +0.286 |

- Pearson(ReindexGate Δ, GOLD Δ) = **+1.000**, Kendall τ = +0.667.
- Pearson(1−RBO, GOLD Δ) = **−0.508**.
- RBO alarm false-fires on 4/10: every improvement and the benign reseed. ReindexGate passes all four.
- Shallow-pool improvement (16→256 @k=2): incumbent-pool delta `+0.0185`, GOLD `+0.2816`, union-pool corrected `+0.2816`.
- Corrected delta = GOLD delta holds on this clean synthetic fixture. On real corpora the oracle only approximates gold; expect high, not perfect, concordance.

## Run

```
# from repo root
python reindexgate/eval.py
# exit 0 on pass
```

```
python reindexgate/gate.py --old-dim 256 --new-dim 16   # -> FAIL
```

## Limits

- Coarse regression detector. Green means no regression it can see.
- Relevance signal is a heuristic. Catches large-capacity regressions (dim truncation) and paraphrase-partner relevance. Misses subtle semantic regressions the reference embedder cannot see.
- Reference embedder is the same algorithm in a distinct seed family. In production, use a frozen, trusted, different embedder.
- Judge independent of OLD and NEW; it cannot favour either index.
- Fixed seed via `numpy.random.default_rng(SEED)`. No wall-clock, no `random`.
- Synthetic corpus. Results transfer as far as the fixture resembles your data.

## Related

- `ranx`: IR-eval (nDCG/MAP/RBP/fusion). Needs qrels. RBP (rank-biased *precision*) is a different measure from the RBO used here. <https://github.com/AmenRa/ranx>
- Drift-Adapter: embedding/index drift across reindexing, arXiv 2509.23471.
- Büttcher, Clarke, Yeung, Soboroff (SIGIR 2007): *Reliable Information Retrieval Evaluation with Incomplete and Biased Judgements*. Pooling bias.
- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993). Bootstrap CI over queries.
- Webber, Moffat, Zobel (2010): *A Similarity Measure for Indefinite Rankings*. RBO, used as a churn diagnostic.

## Files

| file | purpose |
|---|---|
| `embedder.py` | vendored BLAKE2b hashing embedder (dim knob = capacity) |
| `corpus.py`   | deterministic synthetic corpus + label-free probe mining |
| `index.py`    | chunker + build/search a tiny vector index (`IndexConfig`) |
| `gate.py`     | relevance oracle, debiased nDCG delta, bootstrap CI, RBO, verdict, CLI |
| `concordance.py` | GOLD structural qrels + real-nDCG concordance machinery + kappa/Pearson/Kendall stats |
| `eval.py`     | red/green self-test + `[CONCORDANCE]` gold-agreement measurement (exit 0 on pass) |
