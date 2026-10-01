# Leakprobe

PII audit ranked by retrievability, with minimal redaction.

Redacts only the PII a live vector index can surface for a PII-eliciting query.

## Pipeline

`leakprobe.py`:

1. Detect: regex for emails / phones / SSNs plus a closed name gazetteer (`detect_pii`).
   Non-overlapping spans, char offsets.
2. Synthesize: per span, a query from PII type keywords plus surrounding context, excluding the
   value (`synthesize_query`).
3. Score reachability: embed chunks into a live index (BLAKE2b hash embedder, dim 256, L2-normalized, cosine).
   Check whether the span's chunk lands in top-k (`score_retrievability`).
4. Recommend: minimal set is reachable spans only. Blanket set is all spans (`audit`).
5. Measure: redaction replaces the span with mask tokens. `recall_at_k` on legitimate queries.

## Prior art

- Microsoft Presidio: existence-based, anonymizes every detected entity.
- rag-corpus-profiler and similar: static corpus profiling, no retrieval reachability.

## Limitations

- Retrieval is simulated. Bag-of-words hash embedder, no semantics. Numbers are a demonstration
  on synthetic fixtures.
- Query synthesis is a heuristic. Reachability is a lower bound on exposure.
- Regex/gazetteer detectors are minimal. Pair with Presidio/NER.
- Chunk boundaries are fixed inputs.

## Usage

```python
from leakprobe import Chunk, audit

chunks = [
    Chunk(id=0, text="Contact zephyr.lead@corp.example for the Q3 budget details."),
    Chunk(id=1, text="Standard boilerplate footer text repeated in every document."),
]
result = audit(chunks, k=3)
print(result.leak_count)          # reachable (actually-leaking) spans
print(result.minimal_redaction)   # spans to redact: reachable-only, recall-preserving
print(result.blanket_redaction)   # every detected span (the over-redacting baseline)
```

## Self-test

```
python leakprobe/eval.py
```

- RED: email in distinctive context (reachable), name in boilerplate (unreachable). Audit catches
  the leak (`A1/A2`), minimal set is smaller than blanket (`A3`), redaction closes the leak
  (`A4`), recall is higher than blanket (`A5`).
- GREEN: corpus with only unreachable PII gives zero leaks and an empty redaction set (`B1/B2`).

Exit `0` on pass. `--debug` prints per-span reachability, ranks and queries.

```
Leakprobe self-test (retrievability-ranked PII audit)
  determinism         = pure hashing/logic, no RNG (re-run identical: True)
  PII spans detected  = 2 (fault corpus)
  reachable (leaks)   = 1  -> ["[email:'zephyr.lead@corp.example'@chunk100]"]
  minimal redaction   = 1 span(s)   blanket = 2 span(s)
  recall@1  baseline=1.000  minimal=1.000  blanket=0.000
  recall preserved    = +1.000 vs blanket mask
  leaks after minimal = 0
  clean corpus leaks  = 0  redactions = 0

RESULT: PASS (red caught, green clean, recall preserved)
```

## Corpus-scale comparison

```
python leakprobe/eval_baseline.py
```

Curve over k in {1, 3, 5, 10}. 24 topics, one template: 12 with a reachable email, 12 with an
unreachable boilerplate-buried name. Blanket baseline uses the same detectors and mask (Presidio-style).

```
   k | recall none | recall minimal | recall blanket | over-redaction | spans(min/blanket)
  ---+-------------+----------------+----------------+----------------+-------------------
   1 |    0.958    |     0.833      |     0.417      |     0.500      |   12 / 24
   3 |    1.000    |     1.000      |     1.000      |     0.500      |   12 / 24
   5 |    1.000    |     1.000      |     1.000      |     0.500      |   12 / 24
  10 |    1.000    |     1.000      |     1.000      |     0.333      |   16 / 24

MEASURED wedge @k=1: minimal recall 0.833 vs blanket 0.417  (+0.417)
recall LOST vs no-redaction baseline @k=1: minimal=0.125  blanket=0.542
over-redaction rate @k=1: 0.500 (12 of 24 detected spans spared)
```

- Recall lost vs no redaction at k=1: minimal 0.125, blanket 0.542.
- By k=3 both reach full recall.

## Determinism

No RNG, wall-clock, network or install. Fixed-key BLAKE2b hash (`0xC0FFEE`), stable `argsort`.
Output is byte-identical across runs. Eval check `A6` re-runs the audit and compares fingerprints.

## Files

- `embedder.py`: BLAKE2b hashing embedder (dim 256, L2-normalized)
- `leakprobe.py`: detection, query synthesis, index, reachability, redaction, recall,
  `over_redaction_rate`
- `eval.py`: RED/GREEN self-test
- `eval_baseline.py`: corpus-scale comparison vs blanket redaction
