# CovGate

CovGate guards the coverage stage: it checks whether a corpus actually talks about the things it
should, not just whether a keyword shows up somewhere in it. It catches entities that look "well
covered" by a naive word count but are really only ever mentioned in their own record, never
independently confirmed by anything else in the corpus.

More precisely, it is a corpus-wide gap-hunting audit against an external authoritative
registry for RAG pipelines. It answers a question that most pipelines never check because
there is no error to raise: entities the authoritative source says exist but the corpus
never actually talked about anywhere else, never cross-referenced, never corrected, never
independently corroborated.

Given a registry of known entity IDs (e.g. every RFC number a real RFC index
says exists) and a set of mentions tagged by source type (an RFC's own body,
errata, other RFCs citing it, an IANA-style registry, and so on), CovGate scores each
entity's `type_coverage`: the fraction of other source types that
corroborate it, explicitly excluding the entity's own defining/authoritative
record, and buckets it `orphan` / `thin` / `adequate`.

## Explicitly NOT a ChunkLedger reskin

This repo's `ChunkLedger` (`chunkledger/`) and CovGate look superficially
similar (both are "conservation"-flavored gates over structured content) but
check opposite failure modes:

- **ChunkLedger** is **self-referential, within one document**: the parsed
  source is its own reference, and it proves nothing already *in* that
  document was *dropped* while turning it into chunks. Its unit of analysis
  is a byte span inside one document's own chunking.
- **CovGate** is **external-referential, across the whole corpus**: it
  compares the corpus against an authoritative registry that lists what
  *should* exist, and flags entities that were never surfaced anywhere else
  in the corpus at all. Nothing was "dropped in transformation": the
  content simply never existed anywhere but the entity's own record. There is
  no byte span to point at; the unit of analysis is an entity's
  cross-source corroboration.

## Two things this adds

1. Coverage excludes the entity's own authoritative source by
   construction, not as an afterthought filter. The head-to-head in
   `eval.py` backs this with numbers: an entity mentioned **50
   times**, every mention inside its own defining record, reads as
   "well covered" (count ≥ 5) to a naive keyword-frequency audit, while
   CovGate's `type_coverage=0.0` correctly reveals zero independent
   corroboration. The fixture also includes two fairness anchors, a
   genuinely well-corroborated entity and a genuinely under-discussed one,
   where both metrics agree, so the trap case is a real disagreement and
   not a rigged strawman.
2. `coverage_drift` as a severity-regression list, structurally parallel
   to ChunkLedger's own `drift_gate` (same prior/current dict-in contract),
   but shaped as "which entities got worse" rather than a single trip switch,
   because a coverage audit's natural output is a punch list, not a
   boolean.

Everything else here (set/dict bookkeeping, threshold buckets) is standard
and **not** claimed as novel.

## Comparable approaches

- Standard **data-quality "completeness" audits** (does every ID in a
  reference list appear in the dataset at all) are common in data
  engineering. CovGate's distinct contribution is scoring *how independently*
  an entity is corroborated: by distinct source *type*, with the entity's
  own authoritative record excluded, not just whether it appears once,
  anywhere.
- Citation-network / h-index-style corroboration metrics count
  citations, but typically don't separate "cited by an independent source"
  from "appears in its own record," which is exactly the blind spot this
  tool's head-to-head demonstrates against a naive raw count.
- Sibling tools in this repo: **ChunkLedger** (self-referential, per-document
  conservation, see above) and **SyncGate** (`syncgate/`, corpus
  bookkeeping across *runs* and cross-document *reference* integrity) are
  related but check different things than CovGate's cross-source
  corroboration audit.

No existing tool found under a name close to this one covers the same ground.
The closest prior art is the completeness-audit and citation-network work
cited above, and it differs on the axes described there.

## Where CovGate stops

CovGate is a deterministic, offline, stdlib-only proof-of-concept of the
coverage-scoring mechanism, with a real red/green self-test and a working
CLI.

- No numpy needed. Like SyncGate, CovGate's core is dict/set bookkeeping
  over mention records, not numeric scoring: pure standard library.
- Thresholds are documented knobs, not tuned against real data:
  `THIN_COVERAGE_MAX = 1/3` (at most one of three other source types) and
  `THIN_MENTION_MIN = 3` (fewer than three corroborating mentions) are the
  frozen defaults for this fixture's `OTHER_SOURCE_TYPES`; a real deployment
  with a different registry/source-type set should re-derive these.
- CovGate does not decide *why* an entity is orphaned. Never
  cross-referenced, recently added, or genuinely obscure all look the same
  to it. It surfaces the gap; triage is a human/downstream job.
- The demo's registry (4 entities) and its `RFCX`-prefixed IDs are
  **synthetic and obviously fictional**, chosen so nothing here reads as a
  claim about any real IETF document's actual coverage.

## Files

| File | Purpose |
|------|---------|
| `covgate.py` | core: `score_entity`, `audit_corpus`, `coverage_drift`, + CLI |
| `eval.py` | first-milestone **red/green self-test** (exit 0 on pass) |

## Run the self-test

```
python covgate/eval.py
```

It scores 4 fictional RFC-style entities against 3 other source types, then:
- **RED/GREEN:** buckets are verified against the real per-type mechanism:
  full cross-source corroboration scores `adequate`, one corroborating type
  scores `thin`, and zero scores `orphan`;
- **RED:** `coverage_drift` catches an entity regressing from `adequate` to
  `thin` between two runs, naming it explicitly; an identical re-run reports
  zero regressions;
- **head-to-head:** an entity mentioned 50 times, entirely within its own
  record, fools a naive keyword-frequency audit into "well covered" while
  CovGate correctly scores it `orphan` (`type_coverage=0.0`).

### Measured self-test output (real run)

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

`examples/registry.json` + `examples/mentions.json` are a tiny runnable
fixture (2 entities: one `adequate`, one `orphan`); the second command is a
no-op re-run against the first's output (written to `covgate/current.json`,
gitignored; see `.gitignore`), so it reports zero regressions.

`--registry` is a JSON list of entity IDs; `--mentions` is a JSON list of
`{entity, source_type, detail}`. With `--prior`, the process **exits non-zero
(2) when `coverage_drift` finds a regression**, so it drops straight into CI.

## Determinism

No RNG, no wall-clock, no network. Scoring is plain dict/set bookkeeping over
supplied mention records. Identical inputs produce an identical result
(asserted in the self-test).
