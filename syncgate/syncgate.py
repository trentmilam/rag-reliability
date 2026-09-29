"""SyncGate: incremental-sync correctness + cross-document reference integrity.

The problem: a RAG ingestion pipeline maintains a `state.json`-shaped watermark
(``{relative_file_path: content_sha256}``) across runs so a repeat ingest only
re-embeds files that actually changed (see e.g. the sibling `agentic-rag`
project's `ingest/state.py`: `load_state`/`save_state`/`changed_files`). Two
things silently rot around that watermark and are otherwise invisible until
retrieval quality craters:

  1. the incremental diff itself can be wrong (reprocessing too much wastes
     work; reprocessing too little serves stale content as current);
  2. a document can reference another entity by ID (an RFC's `Obsoletes: RFC
     NNNN`, an erratum's "corrects RFC NNNN") whose target has since vanished
     from the corpus (renamed, withdrawn, or never existed), and nothing
     flags the broken pointer until a reader hits it.

SyncGate covers three DISTINCT, precisely-named concepts, each its own
function so they cannot be silently conflated:

  * INCREMENTAL SYNC (`changed_files` / `plan_sync`): label-free, self-
    referential. The hash comparison against the prior watermark IS the
    ground truth of what changed. No external "what should have changed"
    oracle is needed.
  * REFERENCE CLOSURE (`reference_closure`): the strict, cache-blind
    definition. Every reference a live document makes must resolve to a
    target itself present in the current live entity set, full stop. This is
    also exactly what a naive `dict.get(ref) is None` check computes.
  * REFERENCE FALLBACK (`resolve_reference` / `resolve_references`): a
    three-way outcome that closure alone cannot express. LIVE (target is
    live), STALE_CACHED (target isn't live but a last-known-good snapshot
    exists, so it is served with an explicit staleness flag), or DANGLING (no
    live target and no cached snapshot, a hard fail that never fabricates a
    target).

This is NOT ChunkLedger's job (see `chunkledger/chunkledger.py`): ChunkLedger
proves nothing was dropped *within* one document's own chunking, self-
referentially, at chunk-boundary granularity. SyncGate audits the corpus-wide
bookkeeping *across* documents and *across* runs: which files need
reprocessing, and whether cross-document pointers still resolve.

Deterministic, offline, stdlib only (no numpy needed: this is dict/set
bookkeeping, not numeric scoring). No network, no wall-clock, no RNG.

Public API:
  changed_files(current_hashes, old_state) -> list[str]
  plan_sync(current_hashes, old_state) -> SyncPlan
  reference_closure(edges, live_entities) -> list[RefEdge]
  resolve_reference(edge, live_entities, cached_entities) -> RefResolution
  resolve_references(edges, live_entities, cached_entities) -> list[RefResolution]
  run_gate(current_hashes, old_state, edges, live_entities, cached_entities) -> SyncGateResult
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum


# ---- incremental sync -----------------------------------------------------
@dataclass
class SyncPlan:
    changed: list[str]
    unchanged: list[str]

    @property
    def total(self) -> int:
        return len(self.changed) + len(self.unchanged)

    @property
    def reduction_factor(self) -> float:
        """How many times fewer files this plan reprocesses vs. reprocessing
        everything every run (the naive incumbent). `inf` when nothing changed,
        the everyday case for an incremental sync, not a hypothetical."""
        if not self.changed:
            return float("inf")
        return self.total / len(self.changed)


def changed_files(current_hashes: dict[str, str], old_state: dict[str, str]) -> list[str]:
    """New-or-hash-differs files: the incremental-sync ground truth.

    Same semantics as `agentic-rag/ingest/state.py`'s `changed_files(
    all_files_with_hashes, old_state) -> list[Path]`, operating on plain
    relative-path strings (this module's fixtures don't touch the real
    filesystem). Self-referential: the hash comparison itself is the ground
    truth of what changed, so no external oracle is required.
    """
    return sorted(
        rel for rel, chash in current_hashes.items() if old_state.get(rel) != chash
    )


def plan_sync(current_hashes: dict[str, str], old_state: dict[str, str]) -> SyncPlan:
    """The full incremental-sync plan: exactly what changed vs. what didn't."""
    changed = changed_files(current_hashes, old_state)
    changed_set = set(changed)
    unchanged = sorted(rel for rel in current_hashes if rel not in changed_set)
    return SyncPlan(changed=changed, unchanged=unchanged)


# ---- reference integrity ---------------------------------------------------
class RefStatus(str, Enum):
    LIVE = "LIVE"
    STALE_CACHED = "STALE_CACHED"
    DANGLING = "DANGLING"


@dataclass(frozen=True)
class RefEdge:
    from_id: str
    to_id: str
    reason: str


@dataclass
class RefResolution:
    edge: RefEdge
    status: RefStatus
    detail: str


def reference_closure(edges: list[RefEdge], live_entities: set[str]) -> list[RefEdge]:
    """Strict, cache-blind closure check: edges whose target is NOT in the
    current live entity set. This is the classic naive-adjacent definition:
    it is exactly what a `dict.get(ref) is None` presence check computes, and
    on its own cannot distinguish a target that was renamed (recoverable from
    a cached snapshot) from one that never existed (a hard failure). Use
    `resolve_references` for that distinction.
    """
    return [e for e in edges if e.to_id not in live_entities]


def resolve_reference(
    edge: RefEdge,
    live_entities: set[str],
    cached_entities: set[str] = frozenset(),
) -> RefResolution:
    """Three-way fallback classification for one reference edge.

    LIVE:           target is present in the current live entity set.
    STALE_CACHED:   target isn't live, but a last-known-good snapshot exists
                     (e.g. the target was renamed/superseded); serve it, but
                     the caller MUST surface the staleness, never pass it off
                     as current.
    DANGLING:       no live target and no cached snapshot. Never fabricate a
                     target that doesn't exist: hard-fail instead.
    """
    if edge.to_id in live_entities:
        return RefResolution(edge, RefStatus.LIVE, "target present in the current live set")
    if edge.to_id in cached_entities:
        return RefResolution(
            edge, RefStatus.STALE_CACHED,
            "target absent from the live set; serving last-known-good cached snapshot",
        )
    return RefResolution(
        edge, RefStatus.DANGLING,
        "target absent from the live set AND no cached snapshot exists",
    )


def resolve_references(
    edges: list[RefEdge],
    live_entities: set[str],
    cached_entities: set[str] = frozenset(),
) -> list[RefResolution]:
    return [resolve_reference(e, live_entities, cached_entities) for e in edges]


# ---- the combined gate ------------------------------------------------------
@dataclass
class SyncGateResult:
    plan: SyncPlan
    resolutions: list[RefResolution] = field(default_factory=list)

    @property
    def live(self) -> list[RefResolution]:
        return [r for r in self.resolutions if r.status is RefStatus.LIVE]

    @property
    def stale_cached(self) -> list[RefResolution]:
        return [r for r in self.resolutions if r.status is RefStatus.STALE_CACHED]

    @property
    def dangling(self) -> list[RefResolution]:
        return [r for r in self.resolutions if r.status is RefStatus.DANGLING]

    @property
    def passed(self) -> bool:
        """Gate policy: DANGLING references fail the gate. STALE_CACHED does
        not: it is a legitimate, explicitly-flagged degraded-serve outcome,
        not a defect."""
        return not self.dangling

    def report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"SyncGate verdict: {verdict}",
            f"  sync : {len(self.plan.changed)} changed / {self.plan.total} total files"
            + (
                f"  ({self.plan.reduction_factor:.1f}x fewer than reprocess-everything)"
                if self.plan.changed
                else "  (nothing to reprocess)"
            ),
            f"  refs : {len(self.resolutions)} checked -> "
            f"{len(self.live)} live, {len(self.stale_cached)} stale-cached, "
            f"{len(self.dangling)} dangling",
        ]
        for r in self.dangling:
            lines.append(
                f"    DANGLING {r.edge.from_id} -> {r.edge.to_id} "
                f"({r.edge.reason}): {r.detail}"
            )
        return "\n".join(lines)


def run_gate(
    current_hashes: dict[str, str],
    old_state: dict[str, str],
    edges: list[RefEdge],
    live_entities: set[str],
    cached_entities: set[str] = frozenset(),
) -> SyncGateResult:
    plan = plan_sync(current_hashes, old_state)
    resolutions = resolve_references(edges, live_entities, cached_entities)
    return SyncGateResult(plan=plan, resolutions=resolutions)


# ---- tiny CLI ---------------------------------------------------------------
def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"file not found: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SyncGate: incremental sync + reference integrity")
    ap.add_argument("--current", required=True, help="JSON file: {path: sha256} current hashes")
    ap.add_argument("--prior", required=True, help="JSON file: prior state.json-shaped watermark")
    ap.add_argument("--edges", help="JSON file: list of {from,to,reason} reference edges")
    ap.add_argument("--live", help="JSON file: list of currently-live entity IDs")
    ap.add_argument("--cached", help="JSON file: list of entity IDs with a cached snapshot")
    args = ap.parse_args(argv)

    current_hashes = _load(args.current)
    old_state = _load(args.prior)
    edges = ([RefEdge(from_id=e["from"], to_id=e["to"], reason=e["reason"])
              for e in _load(args.edges)] if args.edges else [])
    live_entities = set(_load(args.live)) if args.live else set()
    cached_entities = set(_load(args.cached)) if args.cached else set()

    result = run_gate(current_hashes, old_state, edges, live_entities, cached_entities)
    print(result.report())
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
