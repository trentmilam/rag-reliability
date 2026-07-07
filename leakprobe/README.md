# Leakprobe — retrievability-ranked PII audit with minimal redaction

## What it is

Leakprobe audits a RAG corpus for PII, but instead of flagging *every*
occurrence it ranks each detected span by whether the PII is **actually
reachable** — i.e. whether a live vector index surfaces the span's chunk in
top-k for an auto-synthesized, PII-eliciting query. It then recommends the
**minimal redaction set**: redact only what is reachable. The payoff is
retrieval recall preserved versus a blanket mask, because unreachable PII
(boilerplate footers, long tails) is left intact.

Pipeline (`leakprobe.py`):

1. **Detect** — regex for emails / phones / SSNs + a closed name gazetteer
   (`detect_pii`). Non-overlapping spans with char offsets.
2. **Synthesize** — per span, build a query from the PII **type keywords + the
   surrounding context**, *excluding the secret value itself*
   (`synthesize_query`). This models an attacker who knows *what* a secret is
   about but not the value.
3. **Score reachability** — embed all chunks into a live index (vendored hash
   embedder, dim 256, L2-normalized, cosine) and check whether each span's own
   chunk lands in the index top-k for its elicited query (`score_retrievability`).
4. **Recommend** — minimal set = reachable spans only; blanket set = all spans
   (`audit`).
5. **Measure** — redaction replaces a span with mask tokens (diluting the chunk
   vector, the real retrieval cost), and `recall_at_k` quantifies utility on a
   held set of legitimate queries.

## The narrow novelty wedge (and only this)

**Per-span, live-index top-k retrievability scoring driving a minimal,
recall-preserving redaction set.** That is the whole claim. Everything else
(the detectors, the embedder, the recall metric) is standard.

## Prior art (cited)

- **Microsoft Presidio** — strong PII detection/anonymization, but
  **existence-based**: it anonymizes every detected entity regardless of whether
  retrieval can ever surface it, so it over-redacts and silently degrades
  retrieval quality. Leakprobe reuses that *kind* of detection but gates
  redaction on reachability.
- **rag-corpus-profiler** (and similar corpus-profiling tools) — **static**:
  they characterize a corpus's contents/quality but do not model what a
  *retriever* can reach. Leakprobe's contribution is exactly the missing
  retrieval-reachability signal.

## Honest scope / limitations

- **Retrieval is simulated deterministically.** The "embedding model" is a
  BLAKE2b feature-hashing embedder (`embedder.py`), a bag-of-words proxy — it
  has no semantics, synonymy, or subword handling. Reachability numbers are only
  as meaningful as that proxy; a real deployment must swap in the production
  embedder + index. The *method* is model-agnostic; the *numbers here are a
  demonstration on synthetic fixtures*, not a benchmark.
- **Query synthesis is a heuristic** (type keywords + a fixed context window).
  A determined adversary can craft better queries; reachability is a lower bound
  on exposure, not a guarantee of safety.
- **Detection is not the contribution** — the regex/gazetteer detectors are
  minimal and will miss real-world PII; pair with Presidio/NER in practice.
- **No LLM answerer / no GPU telemetry** is involved; nothing to simulate there.
- Chunk boundaries are treated as fixed inputs; re-chunking would change
  reachability.

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

`audit()` runs the full pipeline (detect → synthesize → score reachability →
recommend); `Chunk` is a plain `(id, text)` dataclass.

## Run the red/green self-test

```
python leakprobe/eval.py
```

`eval.py` is the first-milestone RED/GREEN test:

- **RED** — a fault corpus seeds an email in *distinctive* context (reachable →
  a real leak) and a name in *boilerplate* context (unreachable, and the gold
  doc for a utility query). The audit **catches the reachable leak via the real
  retrieval mechanism** (`A1/A2`), the minimal set is strictly smaller than
  blanket (`A3`), applying it **closes the leak** (`A4`), and it preserves
  **measurably more recall** than the blanket mask (`A5`).
- **GREEN** — a corpus whose only PII is unreachable yields **zero** flagged
  leaks and an **empty** redaction set (`B1/B2`).

No verdict is hard-coded; every assertion runs the real detect→synthesize→
retrieve→redact→re-measure path. Exit code `0` on pass.

### Measured output (observed)

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

Run with `--debug` to print per-span reachability, ranks, and the synthesized
queries.

## Corpus-scale head-to-head vs the Presidio-style blanket baseline

`eval.py`'s single hand-built query proves the mechanism but is one tuned
point, which a skeptic can (rightly) discount. `eval_baseline.py` instead
builds a *distribution* of 24 topics under one uniform template (12 with a
reachable email leak, 12 with an unreachable boilerplate-buried name) and
measures recall(none) / recall(minimal) / recall(blanket) plus the
over-redaction rate as a curve over `k ∈ {1, 3, 5, 10}`. The "blanket"
comparator is the fair incumbent baseline: it uses the **exact same
detectors and mask** as Leakprobe — a competent engineer's actual Presidio-style
approach (redact every detected span) — not a crippled strawman; the only
difference is Leakprobe's reachability gate.

```
python leakprobe/eval_baseline.py
```

### Measured output (observed)

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

**Honest framing of the wedge:** minimal redaction is not free — it still
masks every genuinely reachable span (the real leaks), which costs some
recall (0.125 here) because those golds are diluted identically under both
strategies. What minimal avoids is the *wasted* cost of also masking PII
that was never reachable in the first place. The measured, defensible claim
is therefore **comparative, not absolute**: minimal loses 0.125 recall vs the
unsafe no-redaction baseline while blanket loses 0.542 — over 4x more, and
purely to redact spans (the buried names) that were never exposed to
retrieval. That gap is real only at strict k=1: by k=3 both strategies
saturate to full recall on this corpus (there's enough headroom in the
ranking for the diluted gold to still make top-k), which is itself an honest
finding — over-redaction's utility cost concentrates where retrieval is
least forgiving (small k), not everywhere.

## Determinism

Deterministic **by construction, not by a seed** — there is no randomness
anywhere in the pipeline to seed. The embedder is a fixed-key BLAKE2b feature
hash (`embedder.py`, key `0xC0FFEE`), retrieval is a stable `argsort`, and
everything else is pure logic. No RNG, no wall-clock, no `random`, no network,
no pip install. Output is therefore byte-identical across runs, and the eval's
`A6` check **proves** it by re-running the audit and asserting an identical
result fingerprint (rather than merely asserting reproducibility).

## Files

- `embedder.py` — vendored BLAKE2b hashing embedder (dim 256, L2-normalized).
- `leakprobe.py` — detection, query synthesis, live index, reachability scoring,
  redaction, recall, and `over_redaction_rate` (the blanket-vs-minimal metric).
- `eval.py` — the RED/GREEN first-milestone self-test (one hand-built query).
- `eval_baseline.py` — the corpus-scale MEASURED head-to-head vs the
  Presidio-style blanket-redaction baseline (24 topics, curve over k).
