# RAG Reliability & Verification Toolkit

[![test](https://github.com/trentmilam/rag-reliability/actions/workflows/test.yml/badge.svg)](https://github.com/trentmilam/rag-reliability/actions/workflows/test.yml)

RAG (retrieval-augmented generation) is what it's called when an AI answers a question using
documents you hand it, instead of only what it already knows from training. Building one means
chaining together several pipeline stages: chunking documents into pieces, embedding them into
searchable vectors, indexing them, retrieving the right pieces for a query, and tracing an answer
back to its source. Any one of those stages can fail quietly and corrupt the final answer without
ever raising an error.

This repo is twelve small, offline, deterministic tools, one per pipeline stage, that turn those
silent failures into loud, named, verifiable signals. Each tool ships a red/green self-test: it
reproduces a real failure, catches it, and also passes a clean case so the catch isn't rigged.

> RAG tooling is a crowded field, so each tool here claims just one narrow new idea and cites
> prior art for the rest. Several win on the specific combination, packaging, and a rigorous
> demo, not on a new algorithm.

## The twelve tools (one per pipeline stage)

| tool | stage | what it catches | eval |
|---|---|---|---|
| [**ReindexGate**](reindexgate/README.md) | eval / regression | label-free index-vs-index quality delta with a bootstrap CI; fails CI on a bad reindex | red/green |
| [**Deadstage**](deadstage/README.md) | liveness | names the *single* dead pipeline stage (assert, don't mask) | red/green |
| [**VecStamp**](vecstamp/README.md) | embedding integrity | re-embedding reproduction certificate; types quant-drift vs wrong-weights vs lost-norm vs dim-mismatch | red/green |
| [**Plumbline**](plumbline/README.md) | provenance | deterministic chunk→source coverage under normalization; reports drifted/lost/ambiguous citations on reindex | red/green |
| [**ChunkLedger**](chunkledger/README.md) | ingestion | per-structural-element conservation law; catches dropped tables / code | red/green |
| [**GraphRx**](graphrx/README.md) | graph retrieval | GraphRAG structural linter scored by answer-poisoning risk; fixes validated by a poisoning-delta | red/green |
| [**Headroom**](headroom/README.md) | agentic cost | hardware-aware budget governor (VRAM / thermal / latency) → allow / deny / defer before a hop | red/green |
| [**Leakprobe**](leakprobe/README.md) | PII / security | retrievability-ranked PII audit + minimal, recall-preserving redaction | red/green |
| [**Legigate**](legigate/README.md) | OCR / ingestion | reference-free reading-order + table-collapse legibility gate | red/green |
| [**RAGForensics**](ragforensics/README.md) | generation diagnostic | label-free retriever-vs-generator attribution + auto-calibrated leak threshold | red/green |
| [**SyncGate**](syncgate/README.md) | incremental sync / reference integrity | flags only genuinely-changed sources (content-hash diff, measured 6x reduction vs. reprocess-everything) + a three-way reference fallback (LIVE / STALE_CACHED / DANGLING) that names the exact broken edge, never fabricates a target | red/green |
| [**CovGate**](covgate/README.md) | coverage / gap-hunting | corpus-wide external-registry corroboration audit (orphan/thin/adequate + run-over-run drift); catches the self-referential "well covered" trap a naive keyword-frequency count misses | red/green |

## Requirements

```
pip install -r requirements.txt
```

Python 3.10–3.12 (tested with numpy==2.5.0 on Python 3.12.13). No other
dependencies. See "Design principles" below.

## Run every tool's self-test in one command

```
python run_all_evals.py
```

Discovers every tool's `eval.py`, runs it, and prints one PASS/FAIL for the
whole repo, exiting `0` iff all twelve pass. CI (badge above) runs this same
command on every push/PR across Python 3.10/3.11/3.12, so the claim stays
continuously verified rather than a point-in-time assertion.

To run one tool in isolation:

```
python <tool>/eval.py
```

Each exits 0 on pass, reproduces the injected failure (RED) and passes the clean case (GREEN), and is
**deterministic** (fixed seed) and **offline**: no network, no GPU required (GPU/model pieces are
simulated deterministically, clearly labeled).

## Design principles

1. Each tool ships a worked demonstration of a real failure mode: a reproduction of the failure turned into a RED gate plus a GREEN pass for the clean case. These are targeted demonstrations of the tool's core mechanism, not a claim of exhaustive test coverage; each tool's own README says what is and isn't exercised.
2. Each tool claims only one narrow new idea and cites prior art for the rest (ranx / Drift-Adapter for ReindexGate, ragfallback for Deadstage, SCORE-Bench for ChunkLedger, ContextCite / parametric-leak papers for RAGForensics, and so on).
3. Everything is reimplemented from public methods, clean-room; no incumbent source is vendored.
4. Each tool is self-contained: numpy plus the standard library only, each in its own directory.
   There is no shared package and no `pyproject.toml`/`setup.py`, so no tool is pip-installable
   or importable outside this repo. Run each one as a script from the repo root, as shown above.

## Versioning

No tagged releases yet. Once a commit passes CI it will be tagged `vX.Y.Z`
following semantic versioning. See `CHANGELOG.md`.

## Status

Portfolio project demonstrating the pattern end-to-end, not a hardened production library.
See each tool's own limitations section for what is and isn't proven.
