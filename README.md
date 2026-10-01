# RAG Reliability & Verification Toolkit

[![test](https://github.com/trentmilam/rag-reliability/actions/workflows/test.yml/badge.svg)](https://github.com/trentmilam/rag-reliability/actions/workflows/test.yml)

Twelve small, offline, deterministic tools, one per RAG pipeline stage.

Each has a red/green self-test: reproduces a failure, catches it, passes a clean case.

## Tools

| tool | stage | what it catches | eval |
|---|---|---|---|
| [**ReindexGate**](reindexgate/README.md) | eval / regression | label-free index-vs-index quality delta with a bootstrap CI; fails CI on a bad reindex | red/green |
| [**Deadstage**](deadstage/README.md) | liveness | names the single dead pipeline stage | red/green |
| [**VecStamp**](vecstamp/README.md) | embedding integrity | re-embedding reproduction certificate; types quant-drift, wrong-weights, lost-norm, dim-mismatch | red/green |
| [**Plumbline**](plumbline/README.md) | provenance | chunk-to-source coverage under normalization; reports drifted, lost, ambiguous citations on reindex | red/green |
| [**ChunkLedger**](chunkledger/README.md) | ingestion | per-structural-element conservation; catches dropped tables and code | red/green |
| [**GraphRx**](graphrx/README.md) | graph retrieval | GraphRAG structural linter scored by answer-poisoning risk; fixes validated by a poisoning delta | red/green |
| [**Headroom**](headroom/README.md) | agentic cost | hardware-aware budget governor (VRAM / thermal / latency): allow / deny / defer before a hop | red/green |
| [**Leakprobe**](leakprobe/README.md) | PII / security | retrievability-ranked PII audit; minimal, recall-preserving redaction | red/green |
| [**Legigate**](legigate/README.md) | OCR / ingestion | reference-free reading-order and table-collapse legibility gate | red/green |
| [**RAGForensics**](ragforensics/README.md) | generation diagnostic | label-free retriever-vs-generator attribution; auto-calibrated leak threshold | red/green |
| [**SyncGate**](syncgate/README.md) | incremental sync / reference integrity | flags only changed sources (content-hash diff, measured 6x reduction vs. reprocess-everything); three-way reference fallback (LIVE / STALE_CACHED / DANGLING); names the broken edge, never fabricates a target | red/green |
| [**CovGate**](covgate/README.md) | coverage / gap-hunting | corpus-wide external-registry corroboration audit (orphan / thin / adequate, run-over-run drift); catches the self-referential well-covered case | red/green |

## Requirements

```
pip install -r requirements.txt
```

Python 3.10–3.12 (tested with numpy==2.5.0 on Python 3.12.13).

## Run every self-test

```
python run_all_evals.py
```

Exits `0` iff all twelve pass. CI runs it on every push/PR across Python 3.10/3.11/3.12.

One tool:

```
python <tool>/eval.py
```

Deterministic (fixed seed) and offline. GPU/model pieces are simulated and labeled.

## Design principles

1. Each tool ships one worked failure demo: RED gate plus GREEN clean case. Not exhaustive coverage; each tool's README lists what is exercised.
2. Each tool claims one narrow new idea and cites prior art for the rest (ranx / Drift-Adapter for ReindexGate, ragfallback for Deadstage, SCORE-Bench for ChunkLedger, ContextCite / parametric-leak papers for RAGForensics, and so on).
3. Clean-room reimplementation from public methods. No incumbent source vendored.
4. Self-contained: numpy plus the standard library, one directory per tool. No shared package, no `pyproject.toml`/`setup.py`. Run each as a script from the repo root.

## Versioning

No tagged releases yet. Tags will be `vX.Y.Z` (semantic versioning). See `CHANGELOG.md`.

## Status

Portfolio project, not a production library.
