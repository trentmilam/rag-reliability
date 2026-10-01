# Plumbline

Deterministic chunk-to-source provenance gate for RAG indexes.

At index time, checks that each stored chunk traces to a real span in its source under
whitespace/OCR/normalization-aware alignment. Across a reindex, emits a diffable provenance
manifest and reports lost and drifted citations.

No model, no network, no GPU. numpy + stdlib only.

## Mechanisms

1. Canonicalization-aware span realignment. Source and chunk go through the same canonical
   transform (`o`→`0`, `l`→`1`, `fi`→ligature, collapsed spaces). Match is mapped back to original
   source offsets.

2. Cross-reindex coverage diff. Two manifests, per citation: `LOST` (no longer resolves),
   `DRIFTED` (moved span, with `from`/`to`/`delta`), `STABLE`, `GAINED` (unresolved in baseline,
   now resolves).

## Prior art

- Guardrails provenance validators (`ProvenanceLLM`, embedding-provenance): answer time,
  LLM/embedding-threshold based.
- Content hash + document versioning: detects changed bytes, not moved spans or drift delta.

## Limitations

- Repeated quotes return the `AMBIGUOUS` sentinel from `resolve` / `naive_resolve`.
  `coverage_diff` reports them in a separate `ambiguous` bucket.
- Alignment is exact-substring in canonical space, not edit-distance. Covers case, whitespace
  runs, a small one-to-one OCR confusion set, two ligatures. Inserts, deletes, transpositions and
  unmodeled confusions show up as `LOST`.
- OCR confusion table folds digits to letters (`0`→`o`, `1`→`l`, `5`→`s`). On the 1000-chunk
  benchmark this gives a 70% false-positive rate on numeric-only citations (naive: 0%).
  Config-gate it off for numeric-heavy corpora.
- Citation identity assumed stable across reindex (same `cite_id`, same quoted text).
- OCR/extraction is a fixture (`_ocr_degrade`). No real OCR engine, embedder or LLM. Portfolio
  MVP.

## Files

- `plumbline.py`: `canonicalize`, `resolve`, `naive_resolve`, `build_manifest`,
  `coverage_diff`, `render_manifest`, small `audit` CLI
- `eval.py`: red/green self-test
- `bench.py`: A/B benchmark vs naive baseline, 1000-chunk synthetic OCR corpus

## Run

```
# from the repo root
python plumbline/eval.py
```

Deterministic (`numpy.random.default_rng(SEED)`, no wall-clock).

- GREEN: all 5 OCR-normalized citations resolve to their true spans. Naive exact-substring
  baseline fails all 5.
- RED: reindex inserts a header and deletes the C3-cited region. `coverage_diff` reports
  `LOST=[C3]` and `DRIFTED=[C1,C2,C4,C5]` with destination spans and byte deltas.
- GREEN (control): identical reindex reports 0 lost, 0 drifted.
- Ambiguity regression: repeated quote returns `AMBIGUOUS` from both `resolve` and
  `naive_resolve`.

Exit `0` iff all checks hold.

## Benchmark

```
python plumbline/bench.py
```

1000-chunk synthetic financial corpus, `numpy.random.default_rng(SEED)`. Citations: 60% modeled
OCR (case, `o`→`0`, `l`→`1`, ligature, whitespace), 20% unmodeled OCR (random insert / delete /
transpose), 20% clean. Comparator: `naive_resolve` with the full correct source.

| class | Plumbline recall | naive recall | gain |
|---|---|---|---|
| modeled OCR (the wedge) | **1.000** | 0.000 | **+1.000** |
| unmodeled OCR | 0.025 | 0.025 | +0.000 |
| clean control | 1.000 | 1.000 | +0.000 |
| **all present** | **0.805** | 0.205 | **+0.600** |

| negatives | Plumbline FP rate | naive FP rate |
|---|---|---|
| novel sentences | 0.000 | 0.000 |
| numeric (digit-fold) | **0.700** | 0.000 |

Exit `0`. The benchmark asserts all four facts: wedge gain, clean recall, unmodeled miss,
digit-fold FP.
