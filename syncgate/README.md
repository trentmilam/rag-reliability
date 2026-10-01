# SyncGate

Label-free incremental-sync and cross-document reference-integrity gate for RAG pipelines.

- Which files changed since the last sync.
- Whether each cross-document reference still resolves: `LIVE` / `STALE_CACHED` / `DANGLING`.

Input: the `{relative_file_path: content_sha256}` watermark shape of the sibling `agentic-rag`
project's `ingest/state.py`, plus typed reference edges
(`{"from": ..., "to": ..., "reason": ...}`, e.g. `Obsoletes`, "corrects").

## Mechanisms

1. Three reference states.
   - `LIVE`: target present.
   - `STALE_CACHED`: target renamed or superseded, cached snapshot served.
   - `DANGLING`: target never existed. Names the broken edge (`from`/`to`/`reason`).
   - A naive presence check flags a renamed and a never-existed entity identically.
2. Incremental diff. Hash comparison against the prior watermark.
   Editing 1 of 6 files reprocesses 1/6, a **6.0× reduction**.
3. `reference_closure`: the strict, cache-blind closure, equivalent to the naive check.

Standard sha256 hashing and dict/set lookups.

## Related

- Hash-based incremental build tools (Make, `dvc`, LangChain Indexing API dedup): same watermark shape, no reference-integrity half.
- Link-checkers (`linkchecker`, broken-link CI actions): binary, web-URL-shaped.
- ChunkLedger (`chunkledger/`): conservation within one document's chunking. SyncGate: corpus-wide, cross-run and cross-document.

## Scope

- Deterministic, offline, stdlib-only. No numpy.
- Uses hashes you supply. Computes none, touches no filesystem.
- Fixed two-tier lookup (live set, then cached set). No cache production, staleness limit or eviction.
- Demo corpus: 6 files, 3 reference edges, synthetic `RFCX`/`ERRX` IDs.

## Files

| File | Purpose |
|------|---------|
| `syncgate.py` | core: `plan_sync`, `changed_files`, `reference_closure`, `resolve_reference(s)`, `run_gate`, + CLI |
| `eval.py` | first-milestone **red/green self-test** (exit 0 on pass) |

## Run the self-test

```
python syncgate/eval.py
```

6-file snapshot, one file edited.

- `RED`/measured: `plan_sync` flags exactly the edited file. **6.0× reduction** vs reprocess-everything.
- `RED`: three reference edges. Naive check flags the renamed and the never-existed target identically. SyncGate reports `STALE_CACHED` vs `DANGLING`.
- `GREEN`: making the dangling target live clears the gate.

Diff from a real hash comparison, status from set membership. No hard-coded verdicts.

### Measured output

```
[one file edited] changed=['rfc/rfcx1002.txt']
  incumbent (reprocess everything): 6/6 files
  SyncGate (incremental diff)     : 1/6 files
  measured reduction              : 6.0x fewer files reprocessed

[naive presence check] missing-from-live: {'RFCX1001': False, 'RFCX2099': True, 'RFCX3333': True}
  -> naive check flags RFCX2099 and RFCX3333 IDENTICALLY (both 'missing')

[SyncGate three-way fallback]
  RFCX1003 -> RFCX1001 (Obsoletes): LIVE -- target present in the current live set
  RFCX1004 -> RFCX2099 (Obsoletes): STALE_CACHED -- target absent from the live set; serving last-known-good cached snapshot
  ERRX-9002 -> RFCX3333 (Corrects): DANGLING -- target absent from the live set AND no cached snapshot exists

PASS (exit 0)
```

## CLI

```
# from the repo root
python syncgate/syncgate.py --current syncgate/examples/current_hashes.json --prior syncgate/examples/state.json \
    --edges syncgate/examples/edges.json --live syncgate/examples/live.json --cached syncgate/examples/cached.json
```

`examples/`: one file changed since `state.json`, one edge with a non-live target and a cached snapshot (`STALE_CACHED`, gate passes).

- `--current`/`--prior`: JSON `{path: sha256}` maps.
- `--edges`: JSON list of `{from, to, reason}`.
- `--live`/`--cached`: JSON lists of entity IDs.
- Exits **2** on a `DANGLING` reference.

## Determinism

No RNG, no wall-clock, no network. `hashlib.sha256` over supplied content, plain set membership. Identical inputs, identical results (asserted in the self-test).
