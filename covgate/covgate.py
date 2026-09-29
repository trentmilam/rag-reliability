"""CovGate: corpus-wide gap-hunting against an external authoritative registry.

Explicitly not a ChunkLedger reskin. ChunkLedger (`chunkledger/chunkledger.py`)
is SELF-referential: it proves nothing already-ingested was *dropped* during
one document's own chunking, checked against that same document at
chunk-boundary granularity. CovGate checks a different failure mode entirely:
it compares the WHOLE corpus against an EXTERNAL authoritative registry (e.g.
the real, complete RFC index: every RFC number the index says exists) to
find entities that were never actually written about, cross-referenced, or
corrected anywhere else in the corpus. That content isn't lost in
transformation; it was simply never surfaced anywhere but its own record.

The core idea: for each entity in the registry, score

    type_coverage = (# of DISTINCT *other* source types that mention it)
                    / (total other source types available)

explicitly EXCLUDING the entity's own defining/authoritative source (an RFC's
own body always "mentions" itself; that is not corroboration). An entity is:

  * orphan: zero corroboration from any other source type;
  * thin: corroborated, but coverage is very low or the raw mention
                count is below a documented floor;
  * adequate: otherwise.

`coverage_drift` is CovGate's run-over-run companion, structurally parallel to
ChunkLedger's own `drift_gate` (`chunkledger.py:drift_gate(prior, current,
eps) -> dict`): same dict-in/dict-in "did anything get WORSE" contract, applied
to entity coverage severity instead of per-type conservation ratios. Its shape
differs: it returns a LIST of the entities that regressed (empty == no
regression) rather than a `{"tripped": ...}` boolean envelope, because a
coverage audit's natural output is "which entities need a look", not a single
CI trip switch.

Deterministic, offline, stdlib only (no numpy needed: this is dict/set
bookkeeping, not numeric scoring). No network, no wall-clock, no RNG.

Public API:
  score_entity(entity, mentions, own_source_type, other_source_types) -> CoverageRecord
  audit_corpus(registry, mentions, own_source_type, other_source_types) -> GateResult
  coverage_drift(previous_run, current_run) -> list[dict]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

# ---- tuning knobs (documented, not magic) ----------------------------------
OWN_SOURCE_TYPE = "rfc_body"
OTHER_SOURCE_TYPES: tuple[str, ...] = ("errata", "citing_rfcs", "iana_registry")

THIN_COVERAGE_MAX = 1 / len(OTHER_SOURCE_TYPES)  # only 1 of 3 other types corroborates
THIN_MENTION_MIN = 3                              # fewer than this many corroborating mentions

_SEVERITY_RANK = {"adequate": 0, "thin": 1, "orphan": 2}  # ascending badness, for drift


@dataclass(frozen=True)
class Mention:
    entity: str
    source_type: str
    detail: str = ""


@dataclass
class CoverageRecord:
    entity: str
    type_coverage: float
    total_other_mentions: int
    mentioning_types: list[str]
    severity: str  # "orphan" | "thin" | "adequate"


def classify(type_coverage: float, total_other_mentions: int) -> str:
    """Bucket a scored entity. Zero-corroboration always wins as 'orphan' even
    if a mention count could otherwise clear the thin floor (it can't: zero
    types means zero mentions), so the checks are ordered orphan -> thin -> adequate.
    """
    if type_coverage == 0.0:
        return "orphan"
    if type_coverage <= THIN_COVERAGE_MAX or total_other_mentions < THIN_MENTION_MIN:
        return "thin"
    return "adequate"


def score_entity(
    entity: str,
    mentions: list[Mention],
    own_source_type: str = OWN_SOURCE_TYPE,
    other_source_types: tuple[str, ...] = OTHER_SOURCE_TYPES,
) -> CoverageRecord:
    """Score one entity's corroboration, EXCLUDING its own defining source.

    A mention from `own_source_type` (the entity's own authoritative record --
    an RFC's own body, self-references, its own table of contents) never
    counts toward coverage: it is not independent corroboration that anyone
    else ever engaged with this entity.
    """
    others = [m for m in mentions if m.entity == entity and m.source_type != own_source_type]
    mentioning_types = sorted({m.source_type for m in others})
    type_coverage = (
        len(mentioning_types) / len(other_source_types) if other_source_types else 1.0
    )
    severity = classify(type_coverage, len(others))
    return CoverageRecord(
        entity=entity,
        type_coverage=round(type_coverage, 6),
        total_other_mentions=len(others),
        mentioning_types=mentioning_types,
        severity=severity,
    )


@dataclass
class GateResult:
    records: dict[str, CoverageRecord] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def orphans(self) -> list[str]:
        return sorted(e for e, r in self.records.items() if r.severity == "orphan")

    @property
    def thin(self) -> list[str]:
        return sorted(e for e, r in self.records.items() if r.severity == "thin")

    @property
    def adequate(self) -> list[str]:
        return sorted(e for e, r in self.records.items() if r.severity == "adequate")

    def to_dict(self) -> dict:
        return {
            "records": {e: asdict(r) for e, r in self.records.items()},
            "params": self.params,
        }

    def report(self) -> str:
        lines = [
            f"CovGate: {len(self.records)} entities scored "
            f"({len(self.orphans)} orphan, {len(self.thin)} thin, {len(self.adequate)} adequate)"
        ]
        for e in self.orphans:
            lines.append(f"  ORPHAN  {e}: zero corroboration from any other source type")
        for e in self.thin:
            r = self.records[e]
            lines.append(
                f"  THIN    {e}: type_coverage={r.type_coverage} "
                f"mentions={r.total_other_mentions} types={r.mentioning_types}"
            )
        return "\n".join(lines)


def audit_corpus(
    registry: list[str],
    mentions: list[Mention],
    own_source_type: str = OWN_SOURCE_TYPE,
    other_source_types: tuple[str, ...] = OTHER_SOURCE_TYPES,
) -> GateResult:
    """Score every entity the authoritative registry says exists."""
    records = {
        e: score_entity(e, mentions, own_source_type, other_source_types) for e in registry
    }
    params = {
        "own_source_type": own_source_type,
        "other_source_types": list(other_source_types),
        "thin_coverage_max": round(THIN_COVERAGE_MAX, 6),
        "thin_mention_min": THIN_MENTION_MIN,
    }
    return GateResult(records=records, params=params)


def coverage_drift(previous_run: dict, current_run: dict) -> list[dict]:
    """Run-over-run regression list: entities whose severity WORSENED.

    Structurally parallel to ChunkLedger's `drift_gate(prior, current, eps) ->
    dict`: same dict-in/dict-in contract over a prior/current JSON-shaped
    run (as produced by `GateResult.to_dict()`), applied to entity coverage
    severity instead of per-type conservation ratios. Shape differs on
    purpose: this returns a LIST of the regressed entities (empty == clean)
    rather than drift_gate's `{"tripped": bool, "regressions": [...]}`
    envelope; wrap in `bool(coverage_drift(...))` for gate-style usage.
    """
    prior = previous_run.get("records", {})
    cur = current_run.get("records", {})
    regressions: list[dict] = []
    for entity in sorted(set(prior) | set(cur)):
        p = prior.get(entity, {"severity": "adequate", "type_coverage": 1.0})
        c = cur.get(entity, {"severity": "adequate", "type_coverage": 1.0})
        if _SEVERITY_RANK[c["severity"]] > _SEVERITY_RANK[p["severity"]]:
            regressions.append(
                {
                    "entity": entity,
                    "prior_severity": p["severity"],
                    "current_severity": c["severity"],
                    "prior_type_coverage": p.get("type_coverage"),
                    "current_type_coverage": c.get("type_coverage"),
                }
            )
    return regressions


# ---- tiny CLI ---------------------------------------------------------------
def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"file not found: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CovGate: corpus-wide coverage gap audit")
    ap.add_argument("--registry", required=True, help="JSON file: list[str] of authoritative entity IDs")
    ap.add_argument("--mentions", required=True, help="JSON file: list of {entity,source_type,detail}")
    ap.add_argument("--prior", help="prior GateResult.to_dict() JSON to run coverage_drift against")
    ap.add_argument("--out", help="write the current GateResult JSON here")
    args = ap.parse_args(argv)

    registry = _load(args.registry)
    mentions = [Mention(**m) for m in _load(args.mentions)]
    result = audit_corpus(registry, mentions)
    print(result.report())

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    rc = 0
    if args.prior:
        regressions = coverage_drift(_load(args.prior), result.to_dict())
        if regressions:
            print("COVERAGE DRIFT: regressions found")
            for r in regressions:
                print(f"  {r['entity']}: {r['prior_severity']} -> {r['current_severity']}")
            rc = 2
        else:
            print("COVERAGE DRIFT: clean")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
