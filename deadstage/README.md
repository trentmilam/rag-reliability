# Deadstage

Judge-free RAG liveness gate. Names the single dead pipeline stage.

## What it is

- Read-only probe over a RAG pipeline's artifacts, not its source
- Asserts one structural invariant per stage:

```
ingest -> index -> embed -> retrieve -> score
```

- Walks the stages in order and prints the **earliest** stage whose invariant fails, with a short reason and the measured number
- Non-zero exit code, so it gates CI

## Positioning

- Fallback / failover layers route around a dead component; the pipeline stays green while serving degraded results
- Deadstage: assert and name, do not mask
- New: the assert-and-name posture and first-dead-stage attribution. The invariants are standard (see Prior art).

## Invariants

| Stage    | Invariant asserted                                                        | Example dead reason        |
|----------|---------------------------------------------------------------------------|----------------------------|
| ingest   | ≥1 doc; not all texts blank                                               | `empty/no-text`            |
| index    | cardinality matches ingest; no duplicate / dangling ids                   | `count-mismatch`           |
| embed    | shape + width match; finite; per-vector L2 norm > eps; anisotropy floor   | `collapsed/zero-norm`      |
| retrieve | every query returns ≥1 candidate; all ids exist in the index             | `empty-result`             |
| score    | scores finite, sorted descending, and discriminative (not all tied)       | `degenerate/tied`          |

Embed stage collapse checks:

- `collapsed/zero-norm`: vectors with ~zero L2 norm (embedder deprecated, not loaded, all-zero output)
- `anisotropic-collapse`: vectors crowding into one direction (mean off-diagonal cosine above a floor); from the embedding anisotropy literature

## Usage

```bash
# from the repo root
python deadstage/deadstage.py --pipeline deadstage/examples/pipeline.json           # build a tiny RAG, then probe it
python deadstage/deadstage.py --artifacts deadstage/examples/artifacts.json         # probe artifacts CAPTURED from a real pipeline
python deadstage/deadstage.py --pipeline deadstage/examples/pipeline.json --json    # full report as JSON
```

- `examples/pipeline.json`, `examples/artifacts.json`: runnable fixtures, healthy (exit 0)
- Exactly one of `--pipeline` / `--artifacts` is required
- `--pipeline`: builds a self-consistent pipeline from the vendored embedder (demo / self-test path)
- `--artifacts`: probes `index_ids` / `embeddings` / `retrieval` a live pipeline emitted; retrieve and score stages are reachable on real, possibly inconsistent data
- Exit 0 healthy / 2 dead

`pipeline.json` schema:

```json
{
  "docs":    [{"id": "d0", "text": "..."}],
  "queries": ["..."],
  "k": 3,
  "seed": 1234,
  "embed_kwargs": {"dead": true}
}
```

`artifacts.json` schema (externally captured; probed directly, not re-embedded):

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

`build_pipeline` recomputes retrieval from the embeddings: no dead-embedder plus out-of-index-id states. `from_artifacts` allows them.

## Red/green self-test

`eval.py` builds real pipeline artifacts and runs the real probe. No hard-coded verdicts.

- **RED:** embedder deprecated to zero-vectors. The `embed` norm invariant fires: `embed` / `collapsed/zero-norm`, ingest+index stay live, dead verdict.
- **GREEN:** same corpus, healthy embedder, every stage passes.

```bash
python deadstage/eval.py    # exit 0 iff RED catches the fault AND GREEN passes
```

Output (deterministic):

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

Exit code: **0**.

## Root-cause attribution vs symptom monitoring (A/B)

Baseline: symptom-based monitoring running the identical per-stage invariants, reporting the **deepest** failing stage. Only the attribution direction differs (first-failing vs last-failing).

Five externally captured fixtures (via `from_artifacts`), each with a known injected root cause and its downstream symptoms:

| fixture        | true root | Deadstage (first) | symptom baseline (last) |
|----------------|-----------|-------------------|-------------------------|
| healthy        | n/a       | n/a (correct)             | n/a (correct)                   |
| embed_root     | embed     | embed (correct)          | score (wrong)                |
| index_root     | index     | index (correct)          | retrieve (wrong)             |
| retrieve_root  | retrieve  | retrieve (correct)       | score (wrong)                |
| score_root     | score     | score (correct)          | score (correct)                |

Result (deterministic): Deadstage **5/5 = 100%**, symptom baseline **2/5 = 40%** (**+60-point gap**). The baseline is right only when the fault is at the last stage (`score_root`, `healthy`).

## Limits

- MVP / portfolio piece, not a production library
- Probes structural liveness, not answer quality
- Pipeline: minimal cosine-retrieval RAG with a vendored pure-python hashing embedder (BLAKE2b hashing trick, dim 256, L2-normalized). Not a trained model. `dead` / `constant` embedder modes simulate load-time failures.
- No GPU telemetry, LLM answerer, network or pip install. numpy + stdlib.
- Thresholds (`NORM_EPS`, `ANISO_MAX`, `SCORE_EPS`) are untuned defaults; the anisotropy floor is a heuristic needing per-model calibration
- Determinism: fixed `SEED` via `numpy.random.default_rng`; no wall-clock, no `random`

## Prior art

- ragfallback: failover approach that keeps RAG answering by routing around failures. Not read; cited for contrast only.
- Embedding anisotropy literature: Ethayarajh, "How Contextual are Contextualized Word Representations?" (2019); Gao et al., "Representation Degeneration Problem in Training Neural Language Models" (2019); Li et al., BERT-flow (2020). Basis for the `anisotropic-collapse` invariant.

## Files

- `deadstage.py`: probe + `check()` + `build_pipeline` / `from_artifacts` + CLI
- `embedder.py`: vendored deterministic hashing embedder
- `fixtures.py`: externally captured artifact fixtures for the A/B
- `eval.py`: red/green self-test + root-cause-attribution A/B (exit 0 on pass)
