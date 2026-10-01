# RAGForensics

Label-free RAG failure attribution: retriever or generator, plus a parametric-leak flag.

- Input: a black-box `answer(query, passages) -> str` endpoint and its retriever.
- No gold answers, no relevance judgments.
- Shipped as a deterministic CI gate.

## Mechanisms

- Answer invariance under context removal: answer survives with context stripped, flag a leak.
- Self-consistency under context resampling (shuffle, drop-one).
- Answer-in-context support, checked against the retrieved context, never a gold answer.
  Concept from ContextCite. No code reused.

## Auto-calibrated leak threshold

`calibrate_leak_threshold`

- Held-out queries from the corpus itself.
- 90th percentile of context-removal invariance, plus a margin, clamped to `[floor, ceil]`.
- Monotonic in the corpus baseline, strictly inside `(floor, ceil)` for a mixed corpus.
- Measured: `leaky0=0.35, leaky1=0.65, leaky6=0.95`.

## Auto-calibrated vs fixed cutoff

`ablation.py`. Same invariance signal, same `answer_fn`, only the threshold differs.
Corpus with legitimately context-independent traffic. `eval.py`, 39 held-out queries, deterministic.

| gate | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| fixed-0.5 (naive) | 0.500 | 15 | 11 | 0 | 14 | 0.577 | 1.000 | 0.732 |
| auto-calibrated | 0.753 (data-derived) | 14 | 0 | 1 | 25 | 1.000 | 0.933 | 0.966 |

- FP 11 to 0, F1 0.732 to 0.966, one point of recall lost.
- On this corpus a hindsight fixed constant in `0.65–0.80` ties or marginally beats the calibrated F1.

## Attribution

| retrieval quality (max cosine, query↔retrieved) | answer supported by context | verdict |
|---|---|---|
| below corpus-derived floor | n/a | **retriever** |
| ≥ floor | no | **generator** |
| ≥ floor | yes | **ok** |

- Leak flag is a separate axis: `invariance ≥ auto-calibrated threshold` with an answer produced.
- Plain hallucination is a **generator** fault, not a leak. `eval.py` asserts it.

## Simulated

Offline. No network, LLM or GPU.

- `embedder.py`: vendored ~40-line BLAKE2b feature-hashing embedder, dim 256, L2-normalized, cosine.
- `answerer.py`: `SimAnswerer`, a blend of parametric memory and context support. Treated as an opaque callable.

## Run

```
python ragforensics/eval.py     # red/green + ablation, exit 0 on pass
python ragforensics/forensics.py  # demo report
python ragforensics/ablation.py   # ablation table alone
```

`eval.py`: 28 checks.

- Grounded query is clean.
- Injected parametric leak is caught.
- Injected retrieval miss attributes to the retriever.
- Injected unfaithful generation attributes to the generator, not a leak.
- Control: removing the parametric knowledge from the same query and context clears the flag.
- Ablation head-to-head asserted, and checked deterministic.
- Fixed seed (`numpy.random.default_rng`), no wall-clock, no `random`.

## Related work

- ContextCite (Cohen-Wang et al., 2024): context attribution of a generation to its sources.
- ReDeEP (Sun et al., ICLR 2025), arXiv 2410.11414: separates context use from parametric knowledge.
  The invariance signal here is a black-box proxy, with no access to internals.
- RAGChecker, arXiv 2408.08067: requires labels.

## Limitations

- Answerer and embedder are simulations. A real endpoint changes the numbers and needs multi-sample answers.
- Single-token answers make invariance near-binary. The threshold snaps to `floor`/`ceil` on a fully grounded or fully leaky corpus.
- Strictly-interior values appear on mixed corpora and with multi-token answers (Jaccard overlap is continuous).
- Cosine relevance proxy (retriever false-positive) can under-score a relevant passage with low lexical/embedding overlap.
- Not yet validated on a real RAG stack.
