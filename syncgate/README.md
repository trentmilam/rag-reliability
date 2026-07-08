# SyncGate

When a corpus updates incrementally, only the changed documents should get reprocessed, and any
document that points at another one by ID should still resolve to something real. SyncGate checks
both: it flags exactly what changed since the last sync, and it tells apart a reference that's
gone stale (the target moved but an older cached copy still exists) from one that's genuinely
broken (the target never existed at all).

More precisely, it is a label-free incremental-sync and cross-document reference-integrity gate
for RAG pipelines. It answers two questions most ingestion pipelines answer
with silence or a shrug:

> Did my incremental sync reprocess exactly what changed — no more, no less?
> And when a document points at another entity by ID, does that pointer still
> resolve — and if not, is it *recoverably* stale or *genuinely* broken?

SyncGate operates on the same `{relative_file_path: content_sha256}` watermark
shape a real ingest pipeline already maintains (see the sibling `agentic-rag`
project's `ingest/state.py`), plus a small set of typed reference edges
(`{"from": ..., "to": ..., "reason": ...}`, e.g. an RFC's `Obsoletes` pointer
or an erratum's "corrects" pointer). It reports exactly which files need
reprocessing and a three-way verdict — `LIVE` / `STALE_CACHED` / `DANGLING` —
for every reference.

## What's new here (and what isn't)

Nothing here is a new hashing or graph algorithm. The wedge is in what gets
reported and how the failure modes are separated:

1. Three named sync/reference states, not one boolean. Most "does this
   reference still work" checks collapse to `dict.get(ref) is None`: a single
   True/False. That conflates two structurally different situations — a
   target that was renamed or superseded (recoverable: serve a
   last-known-good cached snapshot, flagged stale) and a target that never
   existed at all (a hard failure — never fabricate a target). SyncGate's
   `resolve_reference` keeps these distinct: `LIVE` / `STALE_CACHED` /
   `DANGLING`. This is measured, not asserted — see the head-to-head in
   `eval.py`: on the exact same fixture the naive presence check flags a
   renamed entity and a never-existed entity identically (`True`/`True`,
   "missing"), while SyncGate reports `STALE_CACHED` for one and `DANGLING`
   for the other, naming the exact broken edge (`from`/`to`/`reason`).
2. Self-referential incremental diff. `changed_files` needs no external
   "what should have changed" oracle — the hash comparison against the prior
   watermark IS the ground truth. `eval.py` measures the reduction directly
   against the naive incumbent ("reprocess everything every run"): editing 1
   of 6 files reprocesses 1/6, a **6.0× reduction**, not an assumed one.
3. `reference_closure` as the explicit naive-equivalent primitive. Rather
   than hiding the naive check inside a strawman, SyncGate's own strict
   (cache-blind) closure function computes the same thing a `dict.get`
   check would — so the three-way fallback's improvement over it is visible
   in the API surface, not just in the demo.

Everything else here (sha256 content hashing, dict/set lookups) is standard
and **not** claimed as novel.

## Prior art (cited, and how this differs)

- Content-addressed incremental build systems (Make-style mtime/hash
  watermarks, `dvc`, LangChain's `Indexing API` dedup) already do
  hash-based incremental sync. SyncGate does not reinvent that; it packages
  the SAME watermark shape with an explicit, measured naive-vs-real
  comparison and adds the reference-integrity half, which those tools don't
  cover.
- Link-checkers (`linkchecker`, broken-link CI actions) already flag `404`s.
  They are binary (works / doesn't) and web-URL-shaped. SyncGate's targets
  are corpus-internal entity IDs, and the three-way fallback (with an
  explicit cached-snapshot state) is the part a generic link-checker doesn't
  model.
- Sibling tools in this repo: **ChunkLedger** (`chunkledger/`) is
  self-referential *within one document's own chunking* — it proves nothing
  was dropped between a source and its chunks. SyncGate is a different axis
  entirely: corpus-wide, cross-run (sync) and cross-document (reference
  integrity) bookkeeping, not per-document conservation.

No existing tool found under a name close to this one covers the same
ground — the closest prior art is the incremental-build and link-checker
tooling cited above, and it differs on the axes described there.

## Honest scope (what this MVP is and is NOT)

- **Is:** a deterministic, offline, stdlib-only proof-of-concept of the two
  mechanisms, with a real red/green self-test and a working CLI.
- **No numpy needed.** Unlike several sibling tools, SyncGate's core is
  dict/set bookkeeping, not numeric scoring — pure standard library.
- **Incremental sync** operates on hashes you supply; it does not compute
  file hashes itself (that's the ingest pipeline's job — see
  `agentic-rag/ingest/state.py`) and does not touch the filesystem.
- **Reference fallback** is a fixed two-tier lookup (live set, then cached
  set). It does not model *how* a cached snapshot was produced, how stale is
  "too stale" to serve, or automatic cache eviction — those are pipeline
  policy, not this gate's job.
- The demo's corpus (6 files, 3 reference edges) and its `RFCX`/`ERRX`-prefixed
  IDs are **synthetic and obviously fictional** — chosen so nothing here
  reads as a claim about any real IETF document.

## Files

| File | Purpose |
|------|---------|
| `syncgate.py` | core: `plan_sync`, `changed_files`, `reference_closure`, `resolve_reference(s)`, `run_gate`, + CLI |
| `eval.py` | first-milestone **red/green self-test** (exit 0 on pass) |

## Run the self-test

```
python syncgate/eval.py
```

It builds a 6-file corpus snapshot and edits one file, then:
- **RED/measured** — `plan_sync` flags exactly the edited file; the
  head-to-head against "reprocess everything" measures a **6.0× reduction**;
- **RED** — three reference edges are resolved: one renamed target (cached
  snapshot exists) and one never-existed target are flagged **identically**
  by a naive presence check, but SyncGate's fallback correctly reports
  `STALE_CACHED` vs `DANGLING`, naming the exact broken edge;
- **GREEN** — fixing the dangling target (making it live) clears the gate.

No hard-coded verdicts: the diff comes from a real hash comparison, and the
three-way status comes from real set-membership lookups.

### Measured self-test output (real run)

```
[one file edited] changed=['rfc/rfcx1002.txt']
  incumbent (reprocess everything): 6/6 files
  SyncGate (incremental diff)     : 1/6 files
  measured reduction              : 6.0x fewer files reprocessed

[naive presence check] missing-from-live: {'RFCX1001': False, 'RFCX2099': True, 'RFCX3333': True}
  -> naive check flags RFCX2099 and RFCX3333 IDENTICALLY (both 'missing')

[SyncGate three-way fallback]
  RFCX1004 -> RFCX2099 (Obsoletes): STALE_CACHED -- serving last-known-good cached snapshot
  ERRX-9002 -> RFCX3333 (Corrects): DANGLING -- no live target AND no cached snapshot exists

PASS (exit 0)
```

## CLI

```
# from the repo root
python syncgate/syncgate.py --current syncgate/examples/current_hashes.json --prior syncgate/examples/state.json \
    --edges syncgate/examples/edges.json --live syncgate/examples/live.json --cached syncgate/examples/cached.json
```

`examples/` is a tiny runnable fixture: one file changed since `state.json`,
and one reference edge whose target isn't live but has a cached snapshot
(`STALE_CACHED`, not `DANGLING`, so the gate passes).

`--current`/`--prior` are JSON `{path: sha256}` maps; `--edges` is a JSON list
of `{from, to, reason}`; `--live`/`--cached` are JSON lists of entity IDs.
Exits **2** when a `DANGLING` reference is found — drop it straight into CI.

## Determinism

No RNG, no wall-clock, no network. Hashing is `hashlib.sha256` over supplied
content; resolution is plain set membership. Identical inputs → identical
result (asserted in the self-test).
