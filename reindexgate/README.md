# ReindexGate: label-free retrieval-regression CI gate

Rebuilding a search index (a "reindex," e.g. after swapping in a new embedding model, the piece
that turns text into searchable vectors, or a new chunking strategy) can quietly make retrieval
worse without anyone noticing until users complain. ReindexGate catches that regression
automatically in CI, without needing a human-labeled set of "correct" search results to compare
against.

More precisely: point ReindexGate at an OLD and a NEW retrieval index and it tells you,
in CI, whether the reindex made retrieval meaningfully worse, without any
relevance labels (qrels) and without an LLM judge. It fails the build when the
new index regresses, and passes a no-op reindex.

## Why existing eval paths don't cover this

Every existing offline IR-eval path needs something ReindexGate does without:

- `ranx` / `pytrec_eval` compute nDCG/MAP beautifully, but require
  qrels (human relevance judgments) you usually do not have in a product repo.
- LLM-as-judge reindex checks need a model, are non-deterministic, cost
  money, and cannot run air-gapped in CI.
- Raw rank-overlap (RBO alone) tells you rankings changed, not whether they
  changed for the worse.

ReindexGate's wedge is the combination, offline and deterministic:

1. Label-free: probe queries are auto-mined from the corpus (known-item /
   inverse-cloze convention); the relevant target is known structurally, not
   from a human label.
2. Judge-free: relevance is a fixed, system-independent pseudo-relevance
   signal (reference-embedding similarity to the query's source document), never
   an LLM and never derived from OLD or NEW. Independence is concrete, not
   asserted. The oracle is built in a distinct hash family (`REF_ORACLE_SEED`,
   near-orthogonal to the `REF_SEED` that OLD/NEW use), so its similarity matrix
   is not byte-identical to either index under test (enforced by the
   `[INDEPENDENCE]` checks in `eval.py`).
3. Pooling-bias-corrected index-vs-index delta: relevance is judged on the
   union pool of both systems' results, so a document found only by NEW is
   judged fairly instead of assumed non-relevant (the classic pooling bias).
4. Delta-with-CI plus a reindex CI-fail gate: a bootstrap confidence interval
   over queries turns "NEW looks a bit worse" into a **decision**, failing only when
   the CI is separated below zero by a margin.

## How it works

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

The demo is fully self-contained. Retrieval uses a **vendored pure-python
hashing embedder** (`embedder.py`, BLAKE2b feature-hashing, dim 256,
L2-normalized, cosine). No pip install, no network, no data files.

### Simulated pieces

Real embedding models and GPU indexes are **not** required to exercise the gate,
so they are simulated deterministically and labelled as such:

- a **dim-truncation reindex** is simulated by shrinking the hash embedder's
  dimensionality: fewer buckets mean more collisions, which lowers the index's capacity;
- a **worse chunker** is a coarse "whole-document" chunker vs the fine one;
- a **different embedding model** is a different hash seed (near-orthogonal).

## Red / green milestone (`eval.py`)

`eval.py` runs the **real** mechanism (no hard-coded verdicts) on three cases:

| scenario | expectation | measured (seed 1234) |
|---|---|---|
| **GREEN** no-op reindex | gate PASSES, CI straddles 0 | delta `+0.0000`, CI `[+0.0000,+0.0000]`, RBO `1.000` → **PASS** |
| **RED** dim-truncation 256→16 | gate FAILS, CI separated < 0 | delta `−0.1363`, CI `[−0.1915,−0.0899]`, RBO `0.519` → **FAIL** |
| **DEBIAS** weak OLD(24) vs strong NEW(256), k=2 | correction is live | corrected `+0.1376` vs incumbent-pool `+0.0123`, **gap `+0.1253`** → **PASS** |

The DEBIAS row demonstrates the pooling-bias correction is not inert: judging on
the incumbent (OLD) pool alone under-credits the genuinely-better NEW index by
`0.125` nDCG; the union-pool correction recovers it.

## Concordance with GOLD labels: the measured proof (`eval.py` `[CONCORDANCE]`)

The core credibility question is whether this label-free heuristic actually agrees
with a real, labelled evaluation. `eval.py`'s `[CONCORDANCE]` section answers it
with numbers, not prose. It attaches **GOLD relevance labels** to the fixture: a
probe mined from doc `d` is relevant to `d` and its paraphrase partner (same
`pair_id`), and to nothing else. These labels are **structural ground truth**,
independent of both the indexes under test *and* the gate's own pseudo-relevance
oracle (which must *recover* them, unseen). It then runs a 10-scenario reindex
matrix (no-ops, real dim-truncation regressions, genuine improvements, a worse
chunker, a benign same-quality reseed) and, for each, compares:

- GOLD: the real nDCG delta computed *with* the labels (what you could only
  get with qrels).
- ReindexGate: the label-free verdict.
- Naive incumbent baseline: "just alert on an RBO churn drop" (`RBO < 0.90`),
  a fair, standard "diff the rankings" alarm.

Measured (seed 1234, 10 scenarios):

| detector (label-free) | accuracy vs GOLD | Cohen κ vs GOLD |
|---|---|---|
| **ReindexGate** | **1.000 (10/10)** | **+1.000** |
| naive RBO-churn alarm | 0.600 (6/10) | +0.286 |

- Signed-delta agreement: Pearson(ReindexGate Δ, GOLD Δ) = **+1.000**,
  Kendall τ = +0.667. ReindexGate's number *is* the gold number here.
- Why RBO-alone fails: Pearson(1−RBO, GOLD Δ) = **−0.508**. Churn magnitude
  carries no direction, so the RBO alarm **false-fires on 4/10** scenarios:
  every genuine *improvement* and the benign reseed churn rankings hard enough
  that RBO drops and triggers a false regression alarm. ReindexGate passes all
  four correctly.
- The match is earned, not tautological. On a shallow-pool improvement
  (16→256 @k=2) the *same* oracle judged on the incumbent pool (the standard
  **biased** eval) gives Δ `+0.0185`, diverging from GOLD `+0.2816` by `0.26`
  nDCG; only the **union-pool corrected** delta reproduces GOLD (`+0.2816`) to
  the last digit. The pooling-bias correction is what buys the concordance.

Caveat: the corrected-delta = GOLD-delta *identity* is a property of
this clean synthetic fixture, where the oracle recovers the paraphrase-pair
structure exactly. On real corpora the oracle is a heuristic that only
*approximately* recovers gold, so expect high, not perfect, concordance. The
result here proves the mechanism is sound and strictly dominates the naive
RBO-only baseline; it does not promise κ=1.0 on arbitrary data.

Run it:

```
# from repo root
python reindexgate/eval.py
# exit 0 on pass
```

CLI for an ad-hoc comparison:

```
python reindexgate/gate.py --old-dim 256 --new-dim 16   # -> FAIL
```

## Limits of the mechanism

- ReindexGate is a **coarse-but-robust regression *detector***, not a proof of
  quality. A green gate means *"no regression this crude detector can see,"* not
  *"NEW is better."*
- The relevance signal is a **heuristic** (embedding similarity to the source
  doc), not ground truth. It reliably catches large-capacity regressions (dim
  truncation) and accurately captures paraphrase-partner relevance; it will **not**
  catch subtle semantic regressions that the reference embedder also cannot see.
- The reference embedder is the vendored hash embedder here, but in a **distinct
  seed family** (`REF_ORACLE_SEED`) from the indexes under test, so it is a
  genuinely separate signal rather than a copy of OLD. It is still the *same
  algorithm*; in production you would go further and point the reference signal
  at a *frozen, trusted, different* embedder. Independence of OLD/NEW guarantees
  the judge cannot favour either index; it does **not** guarantee the judge is
  correct.
- Determinism: fixed seed via `numpy.random.default_rng(SEED)`; no wall-clock, no
  `random`. The synthetic corpus stands in for a real one; results transfer only
  as far as the fixture resembles your data.

## Related tools

- `ranx`: fast Python IR-eval (nDCG/MAP/RBP/fusion; note RBP, rank-biased
  *precision*, is a different measure from the RBO used here). Requires qrels;
  ReindexGate targets the no-qrels case. <https://github.com/AmenRa/ranx>
- Drift-Adapter: embedding/index drift across reindexing, arXiv **2509.23471**.
- Büttcher, Clarke, Yeung, Soboroff (SIGIR 2007): *Reliable Information
  Retrieval Evaluation with Incomplete and Biased Judgements*; the pooling-bias
  problem this gate's union-pool judging addresses.
- Bootstrap confidence intervals: Efron & Tibshirani, *An Introduction to
  the Bootstrap* (1993); resampling over queries for the delta CI.
- RBO: Webber, Moffat, Zobel (2010), *A Similarity Measure for Indefinite
  Rankings* (rank-biased overlap); used here as a churn diagnostic.

## Files

| file | purpose |
|---|---|
| `embedder.py` | vendored BLAKE2b hashing embedder (dim knob = capacity) |
| `corpus.py`   | deterministic synthetic corpus + label-free probe mining |
| `index.py`    | chunker + build/search a tiny vector index (`IndexConfig`) |
| `gate.py`     | relevance oracle, debiased nDCG delta, bootstrap CI, RBO, verdict, CLI |
| `concordance.py` | GOLD structural qrels + real-nDCG concordance machinery + kappa/Pearson/Kendall stats |
| `eval.py`     | red/green self-test + `[CONCORDANCE]` gold-agreement measurement (exit 0 on pass) |
