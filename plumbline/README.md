# Plumbline

**Deterministic chunk→source provenance gate for RAG indexes.**

Plumbline proves, at **index time** and with a **fully deterministic** mechanism,
that every stored chunk's text traces back to a real span in its source document
under **whitespace/OCR/normalization-aware fuzzy alignment** — then, across a
reindex, it emits a **diffable provenance manifest** and reports exactly which
citations **lost coverage** and where the survivors **drifted**.

No model, no network, no GPU. numpy + stdlib only.

## The narrow wedge (the only thing claimed as novel)

Two capabilities, combined, at index time and deterministically:

1. **Canonicalization-aware span realignment.** A stored chunk that was OCR'd or
   whitespace-normalized (`o`→`0`, `l`→`1`, `fi`→ligature, collapsed spaces) no
   longer matches the source byte-for-byte. Plumbline canonicalizes source and
   chunk through the *same* transform, aligns in canonical space, and maps the
   match back to **original source offsets**. A naive exact-substring / content-
   hash check **false-fails** on exactly these legitimate chunks.

2. **Cross-reindex coverage-loss diff.** Given two provenance manifests (before
   and after a reindex), Plumbline reports per citation: `LOST` (no longer
   resolves), `DRIFTED` (resolves to a moved span — with `from`/`to`/`delta`),
   `STABLE` (unchanged), or `GAINED` (unresolved in the baseline, now resolves —
   reported separately rather than folded into `STABLE`). This answers "which
   citations did the reindex silently break, and by how much did the rest
   move?" — which a byte-diff / hash-changed signal cannot.

That is the whole claim. Everything below the wedge (chunking, embeddings,
answering) is out of scope or simulated.

## Prior art (cited; Plumbline is deliberately different)

- **Guardrails provenance validators** (`ProvenanceLLM`, embedding-provenance):
  run at **answer time** and are **non-deterministic** (LLM- or embedding-
  threshold based). Plumbline runs at **index time**, is **deterministic**, and
  uses **no model**.
- **Content-hash + document versioning** (standard index hygiene): tells you a
  byte changed. It does **not** tell you whether a citation still resolves to a
  now-moved span, nor report the drift delta. Plumbline does both.

Plumbline does not attempt semantic provenance, answer faithfulness, or hallucination
detection. It is a mechanical **traceability gate** for the retrieval index.

## Honest scope / limitations

- **Repeated quotes are AMBIGUOUS, not silently resolved to the first match.**
  `resolve`/`naive_resolve` return the `AMBIGUOUS` sentinel whenever a quote's
  canonical (or exact) form occurs more than once in the source — a naive
  first-match (e.g. Python's `str.find`) would otherwise silently pick a span
  that may be the wrong one, producing a false `STABLE` (masking real drift) or
  blaming the wrong occurrence for a `DRIFTED`/`LOST` verdict. `coverage_diff`
  reports these citations under a separate `ambiguous` bucket rather than
  folding them into `LOST`/`DRIFTED`/`STABLE`. This is a real limitation of a
  substring-based aligner: repeated boilerplate/headers/restated summary lines
  cannot be disambiguated without more context than a span match provides.
- Alignment is **exact-substring in canonical space**, not full edit-distance
  fuzzy matching. It handles the modeled normalization classes (case, whitespace
  runs, a small one-to-one OCR confusion set, two ligatures). Insertions/deletions
  *inside* a cited span, transpositions, or unmodeled confusions are **not**
  covered — those show up as `LOST`, not as a partial match. Extending the class
  table or moving to an anchored edit-distance aligner is the obvious next
  milestone.
- The OCR confusion table folds some digits to letters (`0`→`o`, `1`→`l`,
  `5`→`s`). That is a **defined tradeoff** appropriate for OCR'd prose, but it has
  a **measured cost**: it manufactures *false* provenance for short numeric
  citations (`"100"`→`"loo"` matches the word "look"). On the 1000-chunk
  benchmark below this is a **70% false-positive rate on numeric-only citations
  (measured), vs 0% for the naive baseline** — so digit-folding should be
  config-gated OFF for numeric-heavy corpora before real use.
- Citation identity is assumed **stable across reindex** (same `cite_id` → same
  quoted text). Re-deriving citation identity from answers is out of scope.
- All external pieces are **simulated deterministically and clearly labeled**:
  the OCR/extraction pipeline is a fixture (`_ocr_degrade`), there is no real
  OCR engine, embedder, or LLM in the loop. This is a portfolio MVP, not a
  production ingest.

## Files

- `plumbline.py` — the tool: `canonicalize`, `resolve` (fuzzy), `naive_resolve`
  (baseline), `build_manifest`, `coverage_diff`, `render_manifest`, plus a small
  `audit` CLI.
- `eval.py` — the first-milestone **red/green self-test**.
- `bench.py` — the **measured A/B benchmark** vs the naive baseline on a
  1000-chunk synthetic OCR corpus (recall + false-positive rate).

## Run

```
# from the repo root
python plumbline/eval.py
```

`eval.py` is a genuine red/green test (deterministic, `numpy.random.default_rng(SEED)`,
no wall-clock):

- **GREEN (wedge):** Plumbline resolves all 5 OCR-normalized citations to their
  true source spans; the naive exact-substring baseline **false-fails on all 5**
  (proving the tool is doing real work, not rigged).
- **RED (fault caught):** a reindex inserts a header (shifts every offset) and
  deletes the C3-cited region. `coverage_diff` reports `LOST=[C3]` and
  `DRIFTED=[C1,C2,C4,C5]` with destination spans and byte deltas — and the deltas
  match the drift the mutation actually induced (C4/C5 shift less than C1/C2
  because the deletion partially offsets the insertion). If the mechanism failed
  to catch this, eval exits nonzero.
- **GREEN (control):** an identical reindex reports **0 lost, 0 drifted** — no
  false alarms.
- **Ambiguity regression:** a source with a repeated quote surfaces `AMBIGUOUS`
  from both `resolve` and `naive_resolve`, not a silent (and possibly wrong)
  first-match span (see "Honest scope / limitations" above).

Exit code is `0` iff all checks hold. Measured: exit `0`.

## Measured A/B benchmark (`bench.py`)

`eval.py` proves the wedge exists on 5 hand-built citations; `bench.py` turns it
into a defensible number over a **1000-chunk** synthetic financial corpus
(deterministic, `numpy.random.default_rng(SEED)`, numpy + stdlib only). Citations
are degraded as **60% modeled OCR** (case / `o`→`0` / `l`→`1` / ligature /
whitespace — what Plumbline canonicalizes), **20% unmodeled OCR** (random
insert / delete / transpose — noise Plumbline does *not* model), and **20% clean**
(a control that a fair baseline must pass). The comparator is `naive_resolve`
(exact-substring / content-hash-style match), given the full correct source.

**Measured (exit `0`):**

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

Read honestly: the wedge is real and large **only** on OCR/whitespace
*normalization* (modeled) noise — a **+1.000 recall gain** where an exact-match
check false-fails on every legitimate chunk. On **unmodeled** OCR corruption
(insertions/transpositions) Plumbline is **no better than naive** (0.025 each) —
it is a canonical-substring aligner, not an edit-distance one. And digit-folding
costs a **70% false-positive rate on numeric citations** the naive baseline never
incurs. The benchmark **asserts** all four facts (wedge gain, fair-baseline clean
recall, the non-tautological unmodeled miss, and the digit-fold FP) so none can
silently regress into an over-claim.

```
python plumbline/bench.py
```
