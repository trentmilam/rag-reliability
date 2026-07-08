# Deadstage

Deadstage checks whether a RAG pipeline is actually alive end to end, and if one stage of it has
quietly died, it names that exact stage instead of just reporting that the final answer looks
wrong.

## What it is

A judge-free RAG liveness gate that names the single dead pipeline stage. It is
a read-only probe over a RAG pipeline's artifacts (not its source). It
asserts one structural invariant per stage across

```
ingest -> index -> embed -> retrieve -> score
```

(ingest loads the documents in, index stores them for search, embed turns text into the numeric
vectors search runs on, retrieve pulls candidates for a query, score ranks them), and, walking
the stages in order, prints the **single earliest** stage whose
invariant fails, with a short reason and the measured number. It returns a
non-zero exit code so it can gate CI.

Upstream death produces downstream symptoms (a dead embedder makes retrieval
and scoring look broken too), so naming the *first* dead stage points at the
root cause instead of the last thing that threw an exception.

## The wedge (positioning, not invention)

Fallback / failover layers keep a broken pipeline **answering** by routing
around the dead component — the pipeline stays "green" while silently serving
degraded results. Deadstage takes the opposite posture: **assert and name, do
not mask.** When a stage is structurally dead, the gate says which one and
fails loudly.

The only part claimed as new is that assert-and-name posture, plus the
first-dead-stage attribution. The invariants themselves are standard and are
not claimed as new (see Prior art).

## Invariants (per stage)

| Stage    | Invariant asserted                                                        | Example dead reason        |
|----------|---------------------------------------------------------------------------|----------------------------|
| ingest   | ≥1 doc; not all texts blank                                               | `empty/no-text`            |
| index    | cardinality matches ingest; no duplicate / dangling ids                   | `count-mismatch`           |
| embed    | shape + width match; finite; per-vector L2 norm > eps; anisotropy floor   | `collapsed/zero-norm`      |
| retrieve | every query returns ≥1 candidate; all ids exist in the index             | `empty-result`             |
| score    | scores finite, sorted descending, and discriminative (not all tied)       | `degenerate/tied`          |

The **embed** stage carries two representational-collapse checks:

- `collapsed/zero-norm` — vectors with ~zero L2 norm (embedder deprecated / not
  loaded / all-zero output).
- `anisotropic-collapse` — vectors present but crowding into one direction
  (mean off-diagonal cosine above a floor); grounded in the embedding
  **anisotropy** literature.

## Usage

```bash
# from the repo root
python deadstage/deadstage.py --pipeline deadstage/examples/pipeline.json           # build a tiny RAG, then probe it
python deadstage/deadstage.py --artifacts deadstage/examples/artifacts.json         # probe artifacts CAPTURED from a real pipeline
python deadstage/deadstage.py --pipeline deadstage/examples/pipeline.json --json    # full report as JSON
```

`examples/pipeline.json` and `examples/artifacts.json` are small, runnable
fixtures matching the schemas below — both are healthy (exit 0) out of the box.

Exactly one of `--pipeline` / `--artifacts` is required. `--pipeline` manufactures
a self-consistent pipeline from the vendored embedder (a demo/self-test path);
`--artifacts` is the real ingestion path — it probes `index_ids` / `embeddings` /
`retrieval` a live pipeline already emitted, so the retrieve and score stages are
reachable on real (possibly inconsistent) data. Exit 0 healthy / 2 dead.

`pipeline.json` schema (the tool builds a real tiny cosine-retrieval RAG from
the vendored hashing embedder, then probes the artifacts):

```json
{
  "docs":    [{"id": "d0", "text": "..."}],
  "queries": ["..."],
  "k": 3,
  "seed": 1234,
  "embed_kwargs": {"dead": true}
}
```

`artifacts.json` schema (externally-captured — the tool probes these directly,
it does NOT re-embed):

```json
{
  "docs":       [{"id": "d0", "text": "..."}],
  "index_ids":  ["d0", "d1"],
  "embeddings": [[0.01, ...], [0.02, ...]],
  "queries":    ["..."],
  "retrieval":  [[["d0", 0.91], ["d1", 0.40]]],
  "dim": 256,
  "k": 3
}
```

Programmatic API:

```python
from deadstage import build_pipeline, from_artifacts, check

# (a) manufacture a self-consistent pipeline from the vendored embedder:
state = build_pipeline(docs, queries, k=3)

# (b) OR ingest artifacts a REAL pipeline emitted (no re-embedding, no repair):
state = from_artifacts(docs, index_ids, embeddings, queries, retrieval, k=3)

rep = check(state)          # read-only
rep.dead_stage              # None when healthy, else "embed" etc.
rep.healthy                 # bool
```

`from_artifacts` is what makes the retrieve/score stages reachable on real data:
`build_pipeline` always recomputes a self-consistent retrieval from the
embeddings, so it can never emit a state where (say) the embedder is dead *and*
the retriever independently returns an out-of-index id — the exact
multi-symptom states that occur in production.

## Red/green self-test

`eval.py` is the first-milestone red/green gate. It builds **real** pipeline
artifacts and runs the **real** probe — no hard-coded verdicts:

- **RED** — deprecate the embedder to zero-vectors. The `embed` norm invariant
  fires; the probe must name exactly `embed` / `collapsed/zero-norm`, keep
  ingest+index live, and return the dead verdict.
- **GREEN** — same corpus with a healthy embedder passes every stage.

```bash
python deadstage/eval.py    # exit 0 iff RED catches the fault AND GREEN passes
```

Measured output (this repo, deterministic):

```
=== GREEN (healthy embedder) ===
verdict: healthy: all stages live (ingest->index->embed->retrieve->score)
stages : {'ingest': 'ok', 'index': 'ok', 'embed': 'ok', 'retrieve': 'ok', 'score': 'ok'}
PASS -> expected all stages live

=== RED (embedder deprecated to zero-vectors) ===
verdict: embed: collapsed/zero-norm
detail : 6/6 embedding vectors have ~zero L2 norm (min=0.000e+00)
stages : {'ingest': 'ok', 'index': 'ok', 'embed': 'dead', 'retrieve': 'skipped', 'score': 'skipped'}
PASS -> expected embed named as the single dead stage (collapsed/zero-norm)

=== RESULT ===
green_pass=True  red_catches_fault=True
SELF-TEST PASS
```

Real exit code: **0**.

## Measured wedge: root-cause attribution vs symptom monitoring (A/B)

The wedge ("name the *first* dead stage, not the downstream symptom") is not just
asserted — `eval.py` measures it against a **fair incumbent baseline**:
symptom-based monitoring that runs the **identical** per-stage invariants (same
detection power, not a crippled strawman) but reports the **deepest** stage where
a problem is observed — the way real alerting fires on the visible symptom
(empty retrieval / tied scores / a garbage answer). The only isolated variable is
attribution **direction** (first-failing vs last-failing).

The A/B runs over five externally-captured fixtures (via `from_artifacts`), each
with a known injected root cause plus its realistic downstream symptom(s):

| fixture        | true root | Deadstage (first) | symptom baseline (last) |
|----------------|-----------|-------------------|-------------------------|
| healthy        | —         | — (correct)              | — (correct)                    |
| embed_root     | embed     | embed (correct)          | score (wrong)                |
| index_root     | index     | index (correct)          | retrieve (wrong)             |
| retrieve_root  | retrieve  | retrieve (correct)       | score (wrong)                |
| score_root     | score     | score (correct)          | score (correct)                |

**Measured (this repo, deterministic):** Deadstage attributes the root cause
**5/5 = 100%**; the symptom baseline **2/5 = 40%** (a **+60-point gap**). The
baseline is right exactly when the fault *is* at the last stage (`score_root`,
`healthy`) — confirming it is a fair comparator, not a rigged one — and wrong
whenever a live pipeline shows a downstream symptom, which is the common case a
dead embedder or a dangling index id produces. A dead embedder in particular
makes the baseline blame the **scorer** while Deadstage names the **embedder**.

## Honest scope

- This is an **MVP / portfolio piece**, not a production library. It probes
  structural liveness, not answer quality — a live pipeline can still be wrong.
- The pipeline is a **minimal cosine-retrieval RAG** built from a vendored
  pure-python hashing embedder (BLAKE2b hashing trick, dim 256, L2-normalized).
  It is **not** a trained model; it stands in for one deterministically. The
  `dead` / `constant` embedder modes **simulate** load-time failures and are
  clearly labelled as simulation.
- No GPU telemetry, no LLM answerer, no network, no pip install. numpy + stdlib.
- The invariant thresholds (`NORM_EPS`, `ANISO_MAX`, `SCORE_EPS`) are documented
  defaults, not tuned to the fixture; the anisotropy floor in particular is a
  heuristic and would need calibration per embedding model in real use.
- Determinism: fixed `SEED` via `numpy.random.default_rng`; no wall-clock, no
  `random`.

## Prior art (cited)

- **ragfallback** — a failover/fallback approach that keeps RAG answering by
  routing around failures. Deadstage is deliberately the inverse posture
  (assert-and-name vs. mask-via-failover). *Not read; cited for contrast only.*
- **Embedding anisotropy literature** — e.g. Ethayarajh, "How Contextual are
  Contextualized Word Representations?" (2019); Gao et al., "Representation
  Degeneration Problem in Training Neural Language Models" (2019); Li et al.,
  BERT-flow (2020). Motivates the `anisotropic-collapse` embed invariant.

## Files

- `deadstage.py` — probe + `check()` + `build_pipeline` / `from_artifacts` + CLI.
- `embedder.py` — vendored deterministic hashing embedder (self-contained copy).
- `fixtures.py` — externally-captured artifact fixtures for the A/B (deterministic).
- `eval.py` — red/green self-test + measured root-cause-attribution A/B (exit 0 on pass).
