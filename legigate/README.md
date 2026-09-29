# Legigate

When a scanned document gets OCR'd (optical character recognition, turning a scanned image into
text) or parsed for layout, the extraction can quietly scramble the structure: columns get
interleaved, or tables collapse into a blob of text, even though every individual word still reads
fine. Legigate catches chunks with that kind of structural damage before they get indexed.

More precisely, it is a reference-free, per-chunk OCR/parse legibility gate for RAG ingestion.
It scores already-extracted text (no source images, no ground-truth reference) and quarantines
chunks whose structure was corrupted during OCR/layout parsing, before they poison the index.

## The two signals it owns

Existing corpus-cleaning tools (see below) already handle the commodity text-quality pillars
well: encoding/mojibake repair and generic gibberish / language-ID filtering. Legigate drops
those pillars and owns only the two structural signals that those tools underserve, packaged as
a pre-index quarantine gate:

1. Reading-order scramble: column bleed / interleaved lines from
   multi-column pages. Detected by local-coherence transition scoring: with a
   tiny vendored hashing embedder, correctly-ordered prose has adjacent lines
   that are more lexically related than lines two apart
   (`mean cos(i, i+1) > mean cos(i, i+2)`). When two columns are interleaved
   (`A1,B1,A2,B2,…`) the true continuation is two lines away, so that inequality
   inverts. The signed margin is squashed to a `[0,1]` legibility score.

2. Collapsed / merged table structure: cell-boundary loss during
   extraction. Content that still *looks* tabular (short lines, regular token
   counts, numeric density) is re-parsed for cells via multi-space / tab / pipe
   delimiters. A well-formed table yields ≥2 aligned, consistently-counted cells
   per row; a collapsed table (delimiters flattened to single spaces, or rows
   merged) yields one blob per row, so the structure score is about 0.

That is the entire claim. Everything else is standard.

## Scope and limitations

- Signals only apply where they make sense: the table check runs only on
  tabular-looking content, and the order check runs only on non-tabular prose. A
  signal that does not apply scores `1.0` and is reported *not-applicable*, so
  the gate never penalizes the wrong content type. This is enforced by the eval.
- The order signal targets alternation-type corruption (two-column
  interleave / column bleed), the dominant multi-column OCR failure. It
  is not a general shuffle detector: an arbitrary line permutation may not
  invert the lag-1/lag-2 margin.
- The embedder is intentionally weak: a signed BLAKE2b hashing-trick
  bag-of-tokens (dim 256, L2-normalized, cosine) that captures lexical overlap
  only, which is all the order signal needs. No learned weights, no network.
- Thresholds are calibrated on the shipped fixtures, then frozen. They are
  reasonable defaults, not tuned on a broad corpus; real deployment should
  re-calibrate `ORDER_THRESHOLD` / `TABLE_THRESHOLD` on held-out data.
- Out of scope: mojibake/encoding repair and gibberish/language filtering are
  the prior art's job (see below). Legigate is meant to sit *alongside* those
  tools, not replace them.
- Text-only: no image or layout-geometry input. It works purely on the text
  the extractor already produced.

## What existing tools already cover

- `ftfy`: fixes mojibake / broken Unicode. Legigate does **not** attempt
  encoding repair and defers to ftfy for it.
- NVIDIA NeMo Curator and datatrove (Hugging Face): commodity corpus
  heuristics such as gibberish filters, language ID, and quality/dedup pipelines. Legigate
  drops these commodity pillars and owns only the structural signals above.

Legigate's contribution is not "another cleaner." It is the reference-free
structural signals (interleave and table-collapse) packaged as a quarantine gate,
which the tools above do not target.

## Usage

```python
from legigate import score_chunk, gate

v = score_chunk(text, "doc-42#chunk-3")
if not v.legible:
    print(v.reasons)          # human-readable why-quarantined

kept, quarantined = gate([("id1", text1), ("id2", text2)])
```

`ChunkVerdict` fields: `order_score`, `order_margin`, `order_applicable`,
`table_score`, `table_applicable`, `table_detail`, `legible`, `reasons`.

## Run the red/green self-test

```
python legigate/eval.py
```

- `GREEN`: clean single-column prose and a well-formed table both `KEEP`.
- `RED`: the same content after real fault injection (two-column
  interleave; delimiter-collapsed table) is `QUARANTINE`d, each by the real
  mechanism (margin inversion `< 0`; table structure `≈ 0`), not by a
  hard-coded verdict. The eval also asserts the clean scores are genuinely
  *above* threshold (anti-rigging) and that scoring is deterministic across
  repeated and RNG-generated runs.

Exit code `0` on pass.

## Measured: head-to-head against a fair incumbent

The eval builds a small **labeled legibility corpus** (23 deterministic chunks:
13 clean, 10 mechanically-corrupted) and runs both Legigate *and* a **fair
incumbent quality filter** over it, then reports precision/recall for each. The
corpus includes the hard cases: **same-topic interleave**
(columns share vocabulary, so the order margin is far smaller than the
topically-distinct case) and **uniform legal boilerplate** (a false-positive
risk for the order signal).

The incumbent (`baseline.py`) is **not a strawman**: it is a competent
implementation of what the cited prior art actually does, a character-trigram
**perplexity** model (CCNet/datatrove style) plus **Gopher/C4 quality
heuristics** (symbol/word ratio, non-alpha fraction, mean word length, stopword
ratio). It is trained and calibrated on the *clean* half of the corpus (the most
favorable setup for it, since it never false-flags clean content), and as a control
it is verified to **catch genuine gibberish/mojibake (4/4)**, so it is a real
filter being measured on the same data.

Measured on the shipped corpus (deterministic, reproducible via `eval.py`):

| Filter | Precision | Recall (structural corruption) | Genuine gibberish caught |
|---|---|---|---|
| **Legigate** | **1.000** | **1.000** (10/10) | n/a (out of scope) |
| Incumbent perplexity+heuristics | 1.000 | **0.000** (0/10) | 4/4 |

**Measured recall gap: 1.000 vs 0.000.** The incumbent keeps *every*
structurally-corrupted chunk (interleaved columns and delimiter-collapsed tables
are, character- and word-for-word, fluent English, invisible to a quality/
gibberish filter), while Legigate quarantines all 10. Clean vs corrupt applied
scores are fully separated (clean min `0.808` > corrupt max `0.155`, margin
`+0.653`). This is the underserved structural wedge, converted from *claimed* to
*measured*. (Numbers are frozen to the shipped fixtures; real deployment should
re-calibrate and re-measure on held-out data, per the scope notes above.)

## Determinism

Fully deterministic. The pipeline uses no RNG and no wall-clock; the hashing
embedder maps identical text to identical vectors. `SEED = 1234` seeds
`numpy.random.default_rng` only for the eval's *determinism-on-random-inputs*
check (per repo convention).

## Files

- `embed.py`: vendored hashing embedder (BLAKE2b, dim 256, cosine).
- `legigate.py`: the two signals + the gate (`score_chunk`, `gate`).
- `fixtures.py`: deterministic fixtures + genuine fault-injection transforms.
- `corpus.py`: the labeled legibility corpus (clean vs corrupted, incl. the
  hard same-topic interleave + legal boilerplate cases).
- `baseline.py`: the fair incumbent quality filter (perplexity + Gopher/C4
  heuristics) used for the measured A/B.
- `eval.py`: the red/green milestone self-test + the measured A/B.
