# Legigate

Reference-free, per-chunk OCR/parse legibility gate for RAG ingestion.

Scores already-extracted text. Quarantines chunks with structural damage (interleaved columns,
collapsed tables) before indexing. No source images, no ground truth.

## Signals

1. Reading-order scramble (column bleed, interleaved lines). Local-coherence scoring with a small
   hashing embedder: ordered prose has `mean cos(i, i+1) > mean cos(i, i+2)`. Interleaved columns
   (`A1,B1,A2,B2,...`) invert it. Signed margin is squashed to a `[0,1]` score.

2. Collapsed table structure (cell-boundary loss). Tabular-looking = short lines, regular token
   counts, numeric density. Such content is re-parsed for cells via multi-space / tab / pipe
   delimiters. A well-formed table yields at least 2 aligned, consistently-counted cells per row.
   A collapsed one yields one blob per row, so the score is about 0.

## Limitations

- Table check runs only on tabular-looking content, order check only on prose. A signal that does
  not apply scores `1.0` and is reported not-applicable.
- Order signal targets alternation-type corruption. Not a general shuffle detector.
- Embedder: signed BLAKE2b hashing bag-of-tokens (dim 256, cosine). Lexical overlap only. No
  learned weights, no network.
- Thresholds calibrated on the shipped fixtures, then frozen. Re-calibrate `ORDER_THRESHOLD` /
  `TABLE_THRESHOLD` on held-out data.
- Out of scope: mojibake repair, gibberish and language filtering.
- Text only. No image or layout geometry.

## Prior art

- `ftfy`: mojibake / broken Unicode repair
- NVIDIA NeMo Curator, datatrove (Hugging Face): gibberish filters, language ID, quality/dedup

## Usage

```python
from legigate import score_chunk, gate

v = score_chunk(text, "doc-42#chunk-3")
if not v.legible:
    print(v.reasons)          # human-readable why-quarantined

kept, quarantined = gate([("id1", text1), ("id2", text2)])
```

`ChunkVerdict` fields: `order_score`, `order_margin`, `order_applicable`, `table_score`,
`table_applicable`, `table_detail`, `legible`, `reasons`.

## Self-test

```
python legigate/eval.py
```

- GREEN: clean single-column prose and a well-formed table both `KEEP`.
- RED: two-column interleave and delimiter-collapsed table are `QUARANTINE`d (margin `< 0`; table
  structure `≈ 0`).
- Clean scores sit above threshold. Scoring is deterministic across repeated and RNG-generated
  runs.

Exit `0` on pass.

## Results

Labeled corpus: 23 chunks (13 clean, 10 corrupted), including same-topic interleave and uniform
legal boilerplate. Incumbent (`baseline.py`): character-trigram perplexity (CCNet/datatrove
style) plus Gopher/C4 heuristics, calibrated on the clean half. Catches genuine gibberish/mojibake
4/4.

| Filter | Precision | Recall (structural corruption) | Genuine gibberish caught |
|---|---|---|---|
| **Legigate** | **1.000** | **1.000** (10/10) | n/a (out of scope) |
| Incumbent perplexity+heuristics | 1.000 | **0.000** (0/10) | 4/4 |

Clean min `0.808`, corrupt max `0.155`, margin `+0.653`. Numbers are frozen to the shipped
fixtures.

## Determinism

No RNG, no wall-clock. `SEED = 1234` seeds `numpy.random.default_rng` for the eval's
determinism-on-random-inputs check only.

## Files

- `embed.py`: hashing embedder (BLAKE2b, dim 256, cosine)
- `legigate.py`: the two signals and the gate (`score_chunk`, `gate`)
- `fixtures.py`: deterministic fixtures and fault-injection transforms
- `corpus.py`: labeled legibility corpus
- `baseline.py`: incumbent quality filter
- `eval.py`: self-test and A/B
