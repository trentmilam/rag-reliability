# CovGate

Corpus-wide gap-hunting audit against an external authoritative registry. Catches entities that look
well covered by a keyword count but are only mentioned in their own record.

- Input: registry of known entity IDs (e.g. every RFC number an RFC index lists) and mentions tagged by source type (RFC body, errata, citing RFCs, IANA-style registry, ...)
- Scores each entity's `type_coverage`: fraction of other source types that corroborate it, own defining record excluded
- Buckets: `orphan` / `thin` / `adequate`

## Not a ChunkLedger reskin

- **ChunkLedger**: self-referential, within one document. Proves nothing in the document was dropped in chunking. Unit: byte span.
- **CovGate**: external-referential, across the corpus. Flags registry entities never surfaced outside their own record. Unit: cross-source corroboration.

## What it adds

1. Coverage excludes the entity's own source by construction. In `eval.py`, an entity mentioned **50 times**, all inside its own record, reads as well covered (count ≥ 5) to a keyword-frequency audit; CovGate gives `type_coverage=0.0`. Two anchors (one well-corroborated, one under-discussed) where both metrics agree.
2. `coverage_drift`: severity-regression list, parallel to ChunkLedger's `drift_gate` (same prior/current dict-in contract). Output is a punch list of entities that got worse.

Set/dict bookkeeping and threshold buckets are standard.

## Comparable approaches

- Data-quality completeness audits (does every ID in a reference list appear at all). CovGate scores how independently an entity is corroborated, by distinct source type.
- Citation-network / h-index-style metrics: count citations, do not separate independent sources from the entity's own record.
- Siblings: **ChunkLedger** (per-document conservation) and **SyncGate** (`syncgate/`, bookkeeping across runs and cross-document reference integrity).

## Limits

Deterministic, offline, stdlib-only proof of concept with a red/green self-test and a CLI.

- No numpy: dict/set bookkeeping over mention records
- Thresholds are knobs, not tuned on real data: `THIN_COVERAGE_MAX = 1/3`, `THIN_MENTION_MIN = 3` (frozen for this fixture's `OTHER_SOURCE_TYPES`)
- Does not say why an entity is orphaned (never cross-referenced, recently added, obscure)
- Demo registry (4 entities) and `RFCX`-prefixed IDs are synthetic and fictional

## Files

| File | Purpose |
|------|---------|
| `covgate.py` | core: `score_entity`, `audit_corpus`, `coverage_drift`, + CLI |
| `eval.py` | red/green self-test (exit 0 on pass) |

## Self-test

```
python covgate/eval.py
```

Scores 4 fictional RFC-style entities against 3 other source types:
- **RED/GREEN:** full cross-source corroboration scores `adequate`, one type `thin`, zero `orphan`
- **RED:** `coverage_drift` names an entity regressing from `adequate` to `thin`; an identical re-run reports zero regressions
- **head-to-head:** entity mentioned 50 times, all in its own record: keyword audit says "well covered", CovGate scores `orphan` (`type_coverage=0.0`)

### Self-test output (real run)

```
CovGate: 4 entities scored (2 orphan, 1 thin, 1 adequate)
  ORPHAN  RFCX4003: zero corroboration from any other source type
  ORPHAN  RFCX4004: zero corroboration from any other source type
  THIN    RFCX4002: type_coverage=0.333333 mentions=1 types=['errata']

[coverage_drift] prior -> current regressions:
  [{'entity': 'RFCX4002', 'prior_severity': 'adequate', 'current_severity': 'thin', ...}]

  RFCX4001: naive count=8  well_covered=True    CovGate=adequate
  RFCX4003: naive count=1  well_covered=False   CovGate=orphan
  RFCX4004: naive count=50 well_covered=True    CovGate type_coverage=0.0 severity=orphan

PASS (exit 0)
```

## CLI

```
# from the repo root
python covgate/covgate.py --registry covgate/examples/registry.json --mentions covgate/examples/mentions.json --out covgate/current.json
python covgate/covgate.py --registry covgate/examples/registry.json --mentions covgate/examples/mentions.json --prior covgate/current.json
```

- `examples/registry.json`, `examples/mentions.json`: runnable fixture (2 entities: one `adequate`, one `orphan`)
- Second command re-runs against the first's output (`covgate/current.json`, gitignored): zero regressions
- `--registry`: JSON list of entity IDs
- `--mentions`: JSON list of `{entity, source_type, detail}`
- With `--prior`, exits non-zero (2) when `coverage_drift` finds a regression

## Determinism

No RNG, wall-clock or network. Identical inputs give an identical result (asserted in the self-test).
