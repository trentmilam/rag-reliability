# ChunkLedger

Label-free ingestion conservation law for RAG pipelines. Catches chunkers that drop or duplicate
document content, such as a whole table disappearing.

- Parses a source document into typed elements (tables, code blocks, headings, list items, links, numeric spans), each with a byte range
- Checks element by element that content survived into the union of the emitted chunks
- Reports per-type conservation ("tables: 2/3 conserved") and a byte-level manifest of dropped/duplicated spans
- Run-over-run drift gate for CI, tripping on per-type regression

## Differences from existing checks

1. Per-element-type conservation, not one document-wide number. On the same dropping-chunker output the aggregate token ratio reads **0.898**; ChunkLedger reports `tables 2/3` (0.667) and the dropped range `[298, 396]`. Dropping a list block: aggregate **0.889**, `list_items` **0/3**.
2. Label-free. The parsed source is its own reference: no gold chunks, no expected counts.
3. Drift gate keyed per type.

Character-shingle anchoring, LCS-style overlap and markdown parsing are standard.

## Related work

- Unstructured `SCORE-Bench`: reference-based parsing evaluation with an aggregate token-ratio metric. ChunkLedger is reference-free, per-type, with a drift gate.
- Shingling / MinHash / LCS text-overlap techniques: used as the anchoring primitive.

## Scope and limits

Deterministic, offline `numpy`+stdlib proof of concept with a red/green self-test and a CLI.

- Input: markdown. Pragmatic parser (fenced code, pipe tables, ATX headings, list items, inline links, numeric spans), not full CommonMark/GFM. HTML, PDF and nested/edge-case markdown out of scope.
- Anchoring: literal character-shingle coverage. Full-length k-gram shingles match as substrings. Short elements grow to their enclosing alphanumeric token and match on token boundaries (a dropped `12` is not excused by a surviving `512`).
- Contiguity-blind: an element split across two chunks counts as conserved.
- Duplication: flagged when an element is present at conservation strength in ≥2 chunks.
- Demo chunkers (lossless, dropping) and the fixture are synthetic, with a held-out labeled answer key.
- Not tuned for adversarial near-duplicate content or very short documents. Knobs: shingle length `k=12`, threshold `tau=0.85`.

## Files

| File | Purpose |
|------|---------|
| `chunkledger.py` | core: `parse_elements`, `build_ledger`, `drift_gate`, + CLI |
| `eval.py` | red/green self-test (exit 0 on pass) |

## Self-test

```
python chunkledger/eval.py
```

Builds a markdown doc with 3 tables + 2 code blocks, then:
- **GREEN:** lossless (block-split) chunker conserves every type (all ratios 1.0)
- **RED:** dropping chunker omits the block holding table #2; its element coverage falls to ~0.12 (below `tau`), reported as `tables: 2/3` with the missing byte range; drift gate trips vs. baseline
- **GREEN:** lossless re-run vs. baseline leaves the drift gate clean

### Self-test output (real run, 2026-07-04)

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

### Head-to-head vs aggregate token-recall (real run, 2026-07-04)

Baseline: document-wide aggregate token-recall ratio (the SCORE-Bench style metric). Reads exactly `1.0` on a lossless chunker. On an off-the-shelf LangChain-style `RecursiveCharacterTextSplitter` (pure re-partition), both metrics read clean.

```
real off-the-shelf recursive splitter (lossless): incumbent 1.0  |  ChunkLedger min per-type 1.0   (both CLEAN)

scenario               incumbent agg   ChunkLedger type    localized bytes
--------------------------------------------------------------------------
drop 1 of 3 tables            0.8981       tables 0.667       [[298, 396]]
drop the list block           0.8889   list_items 0.000  [[591,618],[619,645],[646,674]]

table drop: incumbent 1.0->0.8981 (-10.2pt)  vs  ChunkLedger tables 1.0->0.667 (-33.3pt)  = 3.3x more sensitive, + byte range
list  drop: incumbent 1.0->0.8889 (-11.1pt)  vs  ChunkLedger list_items 1.0->0.000 (-100pt) = 9.0x more sensitive, entire type wiped
```

- Aggregate names neither the type nor the bytes
- A whole type can vanish (`list_items 0/3`) while the aggregate reads **0.889**
- ChunkLedger: **3.3×–9.0×** more sensitive, with type and byte spans, drift gate at zero tolerance

`eval.py` asserts this: it fails if the incumbent dips below 0.85 or ChunkLedger is under 2x more sensitive.

## CLI

```
# from the repo root
python chunkledger/chunkledger.py --source chunkledger/examples/doc.md --chunks chunkledger/examples/chunks_ok.json --out chunkledger/prior.json
python chunkledger/chunkledger.py --source chunkledger/examples/doc.md --chunks chunkledger/examples/chunks_new.json --prior chunkledger/prior.json
```

- `examples/doc.md`, `chunks_ok.json`, `chunks_new.json`: runnable fixtures
- First command: lossless split, writes `chunkledger/prior.json` (gitignored)
- Second command: re-chunks with the table dropped; drift gate against `prior.json` catches it (exit 2)
- `--chunks`: JSON list of strings (chunker output)
- With `--prior`, exits non-zero (2) when the drift gate trips

## Determinism

Fixed seed via `numpy.random.default_rng(SEED)`. No wall-clock, `random`, network or pip install. Identical inputs give an identical ledger (asserted in the self-test).
