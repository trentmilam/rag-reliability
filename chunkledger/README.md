# ChunkLedger

ChunkLedger watches the ingestion stage of a RAG pipeline, where a document gets split into
chunks (the small pieces a RAG system actually stores and retrieves). It catches chunkers that
silently drop or duplicate part of a document, like an entire table disappearing, with nothing
raising an error to say so.

More precisely, it is a label-free ingestion conservation law for RAG pipelines. Most ingestion
stacks stay silent on whether a chunker's output silently dropped or duplicated structural
content, and on which bytes, when it did. ChunkLedger answers that directly.

ChunkLedger parses a source document into typed structural elements (tables,
code blocks, headings, list items, links, numeric spans), each with a byte
range, then verifies element-by-element that the content survived into the union
of the emitted chunks. It reports a per-type conservation ratio ("tables:
2/3 conserved"), a byte-level manifest of every dropped/duplicated span, and
a run-over-run drift gate for CI that trips on per-type regression.

## Where it differs from existing ingestion checks

Existing ingestion-quality checks compare against an external gold reference or
score an aggregate token ratio across the whole document. ChunkLedger takes a
different approach on three specific axes:

1. Per structural-element-type conservation, not one document-wide number.
   "≈90% of tokens survived" hides "a whole table vanished"; `tables: 2/3` does not.
   The head-to-head below backs that with real numbers: on the exact
   same dropping-chunker output the incumbent aggregate token-ratio reads **0.898**
   (looks fine) while ChunkLedger reports `tables 2/3` (0.667) + the dropped byte
   range `[298, 396]`; drop a whole list block and the aggregate reads **0.889**
   while `list_items` goes to **0/3**.
2. Self-referential and label-free. The parsed source is its own reference:
   no gold chunk set, no human labels, no "expected table count" to maintain.
   You can run it on any document you already have.
3. Run-over-run drift as a CI gate, keyed per type, so a chunker/config change
   that starts dropping list items trips the build even if the aggregate ratio
   barely moves.

Everything else here (character-shingle anchoring, LCS-style overlap, markdown
parsing) is standard and not claimed as novel.

## Related work

- Unstructured `SCORE-Bench`: reference-based document-parsing evaluation
  using an aggregate token-ratio style metric against gold references. ChunkLedger
  is *reference-free* (self-anchoring) and reports *per-element-type* conservation
  plus a *drift gate*, rather than a single aggregate ratio against a gold set.
- Standard **shingling / MinHash / LCS** text-overlap techniques, used here as
  the anchoring primitive and not invented here.

No existing tool found under a name close to this one covers the same ground.
The closest prior art is SCORE-Bench, and it differs on the three axes above.

## Scope and limits

ChunkLedger itself is a deterministic, offline `numpy`+stdlib proof-of-concept
of the conservation law, with a real red/green self-test and a working CLI.

- Input format: markdown. The structural parser is a pragmatic markdown
  parser (fenced code, pipe tables, ATX headings, list items, inline links,
  numeric spans), **not** a full CommonMark/GFM implementation. HTML, PDF, and
  nested/edge-case markdown are out of scope for this milestone.
- Anchoring is literal character-shingle coverage: full-length k-gram
  shingles are matched as substrings; a **short** element (a bare numeric span, a
  short heading) is grown to its enclosing alphanumeric token and matched on
  **token boundaries**, so a genuinely dropped short span is not falsely
  "conserved" by its characters occurring inside an unrelated larger token (e.g.
  a dropped `12` is not excused by a surviving `512`). It detects content that
  *survived vs. vanished*; it is intentionally **contiguity-blind**. An element
  split across two chunks is still "conserved", and that is correct: it wasn't
  lost, just partitioned.
- Duplication detection flags an element present at conservation strength in
  ≥2 chunks (e.g. from overlapping-window chunkers).
- The demo's two chunkers (lossless, dropping) and the GPU/LLM-free fixture are
  **synthetic and clearly labelled**: deterministic synthetic fixtures with a
  held-out labeled answer key, the same discipline used across every tool in
  this repo.
- Not tuned for adversarial near-duplicate content or very short documents; the
  shingle length (`k=12`) and thresholds (`tau=0.85`) are documented knobs.

## Files

| File | Purpose |
|------|---------|
| `chunkledger.py` | core: `parse_elements`, `build_ledger`, `drift_gate`, + CLI |
| `eval.py` | first-milestone **red/green self-test** (exit 0 on pass) |

## Run the self-test

```
python chunkledger/eval.py
```

It builds a markdown doc with 3 tables + 2 code blocks, then:
- **GREEN:** a lossless (block-split) chunker conserves every type (all ratios 1.0);
- **RED:** a dropping chunker that omits the block holding table #2 is caught by
  the *mechanism* (that table's element coverage falls to ~0.12, well below `tau`),
  reported as `tables: 2/3` with the missing byte range, and the **drift gate
  trips** vs. the baseline;
- **GREEN again:** re-running the lossless chunker vs. the baseline leaves the
  drift gate clean.

No hard-coded verdicts: the drop is detected purely by shingle coverage dropping
below threshold, and the gate fires purely from the per-type ratio comparison.

### Measured self-test output (real run, 2026-07-04)

```
parsed: 3 tables, 2 code blocks
[baseline / lossless] tables 3/3, code_blocks 2/2, headings 4/4,
                      list_items 3/3, links 2/2, numeric_spans 19/19  (all ratio 1.0)
[dropping chunker]    tables: 2/3 conserved  dropped spans: [[298, 396]]
                      dropped table element coverage = 0.1205 (tau=0.85)
  drift gate (bad vs baseline): tripped=True  (tables 1.0->0.667, numeric_spans 1.0->0.789)
[clean re-run]        drift gate: tripped=False
PASS  (exit 0)
```

### Measured head-to-head vs the incumbent aggregate metric (real run, 2026-07-04)

The self-test now implements the **incumbent baseline**: a single document-wide
**aggregate token-recall ratio** (the aggregate token-ratio style that
reference-based document-parse evals such as Unstructured's SCORE-Bench report:
fraction of source tokens, as a multiset, present in the union of the emitted
chunks). It is a *fair, working* metric, **not** a strawman: on a lossless
chunker it reads exactly `1.0`, and on a real off-the-shelf LangChain-style
`RecursiveCharacterTextSplitter` (a pure re-partition) **both** metrics agree the
ingest is clean. The gap between them shows up only on a real drop:

```
real off-the-shelf recursive splitter (lossless): incumbent 1.0  |  ChunkLedger min per-type 1.0   (both CLEAN)

scenario               incumbent agg   ChunkLedger type    localized bytes
--------------------------------------------------------------------------
drop 1 of 3 tables            0.8981       tables 0.667       [[298, 396]]
drop the list block           0.8889   list_items 0.000  [[591,618],[619,645],[646,674]]

table drop: incumbent 1.0->0.8981 (-10.2pt)  vs  ChunkLedger tables 1.0->0.667 (-33.3pt)  = 3.3x more sensitive, + byte range
list  drop: incumbent 1.0->0.8889 (-11.1pt)  vs  ChunkLedger list_items 1.0->0.000 (-100pt) = 9.0x more sensitive, entire type wiped
```

The incumbent scalar is not *useless*; it does
dip ~10 points. But (a) that dip is the same order as benign run-over-run token
churn (re-wrapping, boilerplate), so a tolerance tight enough to catch it invites
false positives; (b) the scalar can name **neither the type nor the bytes** that
went missing; and (c) an *entire structural type* can be wiped out (`list_items
0/3`) while the aggregate still reads a comfortable **0.889**. ChunkLedger's
per-type ratio is measured here at **3.3×–9.0× more sensitive** to the structural
loss and hands you the type + exact byte spans, with a drift gate that trips at
zero tolerance. This is the head-to-head asserted in `eval.py` (it fails the
build if the incumbent ever *dips below 0.85* or ChunkLedger is *not* at least 2×
more sensitive), so the claim stays accurate if the fixture changes.

## CLI

```
# from the repo root
python chunkledger/chunkledger.py --source chunkledger/examples/doc.md --chunks chunkledger/examples/chunks_ok.json --out chunkledger/prior.json
python chunkledger/chunkledger.py --source chunkledger/examples/doc.md --chunks chunkledger/examples/chunks_new.json --prior chunkledger/prior.json
```

`examples/doc.md` + `examples/chunks_ok.json` / `chunks_new.json` are tiny
runnable fixtures: the first command's lossless split conserves everything and
writes `chunkledger/prior.json` (gitignored; see `.gitignore`); the second
re-chunks with the table dropped, which the drift gate against `prior.json`
catches (exit 2).

`--chunks` is a JSON list of strings (your chunker's output). With `--prior`, the
process **exits non-zero (2) when the drift gate trips**, so it drops straight into CI.

## Determinism

Fixed seed via `numpy.random.default_rng(SEED)` for the fixture; no wall-clock,
no `random`, no network, no pip install. Identical inputs produce an identical
ledger (asserted in the self-test).
