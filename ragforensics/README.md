# RAGForensics

Label-free, offline diagnostics for a RAG stack: given only a black-box
`answer(query, passages) -> str` endpoint and its retriever, **attribute a failure to
the retriever vs. the generator, and raise a parametric-leak flag — without any gold
answers or relevance judgments.** Shipped as a deterministic CI gate.

## Honest scope

The individual mechanisms here are **commodity**. RAGForensics does not invent a new
detection primitive and does not reimplement anyone's method. It:

- **answer-invariance under context removal** — run the model with the retrieved
  context and again with it stripped; an answer that survives context removal is being
  produced from the model's parametric memory (a leak). Behavioral, black-box.
- **self-consistency** under context resampling (shuffle + drop-one) — a stability
  signal for the generator.
- **answer-in-context support** — is the emitted answer token actually present in the
  retrieved context? This is a faithfulness/attribution check computed **against the
  retrieved context, never against a gold answer** (the idea popularized by ContextCite;
  we use the concept, we do **not** lift its code or its gradient/ablation attribution).

### The one net-new wedge

An **auto-calibrated, corpus-specific leak threshold** (`calibrate_leak_threshold`).
Instead of a magic global cutoff, the gate tunes its threshold from a held-out set of
the corpus's own queries: it takes the 90th percentile of the observed
context-removal-invariance distribution plus a margin, clamped to a sane `[floor, ceil]`.
The result **adapts to how context-independent a given corpus legitimately is** — a
corpus whose normal traffic can be answered without the docs gets a higher bar so the
gate does not false-flag everything. `eval.py` proves this is genuinely data-derived:
the threshold is monotonic in the corpus baseline and lands **strictly inside**
`(floor, ceil)` for a mixed corpus (measured: `leaky0=0.35, leaky1=0.65, leaky6=0.95`).

The claim is narrow and deliberate: **label-free end-to-end + a packaged offline CI gate
+ local-first + the auto-calibrated threshold.** Nothing more.

### Measured: auto-calibrated vs. a fixed global cutoff (`ablation.py`)

The wedge claim above — "adapts so the gate does not false-flag everything" — used to be
*asserted*, not shown. `ablation.py` now measures it head-to-head against the **fair, obvious
baseline**: a fixed `0.5` invariance cutoff, the natural midpoint of the `[0, 1]` signal and
exactly what a competent engineer ships absent any calibration. Both gates read the *same*
real `context_removal_invariance` signal over the *same* opaque `answer_fn`; only the
threshold differs. The scenario is a corpus whose normal traffic is **legitimately
context-independent** (many queries the model can correctly answer without the docs), which is
precisely the case a fixed cutoff cannot tell apart from a genuine leak. Measured
(`eval.py`, 39 held-out queries, deterministic):

| gate | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| fixed-0.5 (naive) | 0.500 | 15 | 11 | 0 | 14 | 0.577 | 1.000 | 0.732 |
| auto-calibrated | 0.753 (data-derived) | 14 | 0 | 1 | 25 | 1.000 | 0.933 | 0.966 |

False positives `11 → 0`, F1 `0.732 → 0.966`, for one point of recall. The fixed cutoff
over-flags the corpus's legitimately context-independent traffic as "leaks"; the calibrated
threshold rises to the corpus's own baseline and only flags queries that are *abnormally*
invariant relative to it.

**Honest caveat (don't over-read this):** on this *one* corpus, a hindsight-picked fixed
constant in the `0.65–0.80` range ties or marginally beats the auto-calibrated F1 — the win
over `0.5` is not "no fixed number could ever work here." The real point is that picking
`0.65–0.80` requires a labelled sweep the label-free design explicitly avoids, and the *same*
fixed constant does not transfer: `eval.py`'s calibration check shows the auto threshold
sliding `0.35 → 0.65 → 0.95` as the corpus's own baseline shifts, so any single global
constant is only ever accidentally right for one baseline and wrong for the others. The
auto-calibrated bar gets into the high-performing region **without seeing a single label**,
which is the property a fixed constant cannot have by construction.

## Attribution logic (label-free)

| retrieval quality (max cosine, query↔retrieved) | answer supported by context | verdict |
|---|---|---|
| below corpus-derived floor | — | **retriever** (nothing relevant was fetched) |
| ≥ floor | no | **generator** (had good context, answered unfaithfully) |
| ≥ floor | yes | **ok** |

Leak flag is a separate axis: `invariance ≥ auto-calibrated threshold` while the model
actually produced an answer. A plain hallucination (context-dependent, unsupported) is
caught as a **generator** fault but is **not** a leak — the two are separable, and
`eval.py` asserts it.

## What is SIMULATED (and clearly labelled)

Fully offline, no network, no LLM, no GPU:

- **Embedder** (`embedder.py`): a vendored ~40-line pure-python BLAKE2b feature-hashing
  embedder, dim 256, L2-normalized, cosine. Stands in for a sentence embedder for
  retrieval; the `weights_seed` plays the role of "which model".
- **Answerer** (`answerer.py`): a `SimAnswerer` with a transparent blend rule (parametric
  memory vs. context support). It stands in for a real (retriever + generator) stack. The
  forensics code treats it as an opaque callable and **never** reads its internals or any
  label. The faults are emergent from the blend rule; verdicts are not hard-coded.

## Run

```
python ragforensics/eval.py     # red/green + ablation, exit 0 on pass
python ragforensics/forensics.py  # demo report
python ragforensics/ablation.py   # ablation table alone
```

`eval.py` is the first-milestone RED/GREEN self-test (28 checks): a grounded query is
clean (green); an injected parametric leak is caught by the real invariance mechanism
(red) and disagrees with the docs; an injected retrieval miss attributes to the
retriever (red); an injected unfaithful generation attributes to the generator and is
**not** mislabelled a leak (red). An anti-rig **control** removes the parametric
knowledge from the *same* query+context and shows the flag disappears — proving the
verdict is behavioral, not keyed on the query id. It also runs and asserts the
`ablation.py` head-to-head against a fixed global cutoff (see "Measured" above), and
checks the ablation is itself deterministic. Deterministic (`numpy.random.default_rng`,
fixed seed, no wall-clock/`random`).

## Prior art (cited; not reimplemented)

- **ContextCite** (Cohen-Wang et al., 2024) — context-attribution of a generation to
  its sources. We reuse the *concept* of answer-vs-context support; we do not lift its
  code or its ablation/gradient attribution method.
- **ReDeEP / parametric-leak analysis** — arXiv **2510.12668**; work on detecting when a
  RAG model answers from parametric knowledge rather than the retrieved context. Our
  context-removal-invariance signal is a black-box behavioral proxy for this.
- **RAGChecker / RAG-E** and similar evaluators — thorough, but **require labels**
  (gold answers / relevance). RAGForensics' wedge is being **label-free**.

## Limitations (honest)

- The answerer/embedder are **simulations**; against a real endpoint the same mechanisms
  apply but the numbers change and the invariance signal needs multi-sample answers.
- With **single-token** answers, invariance is near-binary, so on a perfectly-grounded or
  perfectly-leaky corpus the auto-calibrated threshold snaps to `floor`/`ceil`; the
  interpolated, strictly-interior value appears on mixed corpora (and, in production, with
  multi-token answers where Jaccard overlap is continuous).
- Attribution uses a cosine-similarity relevance proxy; a semantically-relevant passage
  with low lexical/embedding overlap could be under-scored (retriever false-positive).
- Not yet validated on a real RAG stack — that is the next milestone.
