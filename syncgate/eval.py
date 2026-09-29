"""SyncGate FIRST-MILESTONE red/green self-test.

Milestone: prove the incremental-sync diff is exactly right (not "roughly
fine") and that reference-integrity resolution genuinely distinguishes a
DANGLING reference from a STALE_CACHED one (something a naive presence
check structurally cannot do), with no rigging: the sync diff is a real hash
comparison, and the dangling verdict comes from actually looking up the
target in two real sets (live, cached), not from a hard-coded outcome.

Scenario (obviously-fictional IDs; the "RFCX"/"ERRX" prefixes do not collide
with any real IETF numbering):
  * A 6-file corpus snapshot, then one file's content edited, so exactly 1 of
    6 should be flagged for reprocessing.
  * Three reference edges: one resolves cleanly; one points at an entity that
    was renamed/superseded (absent from the live set, but a cached snapshot
    survives); one points at an entity that never existed (absent from both).

Exit 0 iff every RED fault is genuinely caught, every GREEN case stays clean,
and the head-to-head measurements hold.
"""

from __future__ import annotations

import hashlib

from syncgate import (
    RefEdge,
    RefStatus,
    changed_files,
    plan_sync,
    reference_closure,
    resolve_references,
    run_gate,
)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _state_of(files: dict[str, str]) -> dict[str, str]:
    return {path: _sha256(content) for path, content in files.items()}


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


# --- fixture: a 6-file corpus snapshot (obviously-fictional RFC-style IDs)
FILES_V1 = {
    "rfc/rfcx1001.txt": "RFCX1001 defines the Widget Transfer Protocol version 1.",
    "rfc/rfcx1002.txt": "RFCX1002 defines the Gadget Naming Scheme.",
    "rfc/rfcx1003.txt": "RFCX1003 obsoletes RFCX1001 with a revised handshake.",
    "errata/errx-9001.txt": "Erratum 9001 corrects RFCX1002 section 3.",
    "iana/registry-a.txt": "IANA-style registry entries for Widget Transfer Protocol.",
    "index/rfcx-index.txt": "Index of all known RFCX documents.",
}


def main() -> int:
    print("deterministic: sha256 content hashing + set/dict lookups, no RNG needed")

    # --- GREEN precondition: an unchanged snapshot has nothing to reprocess -
    v1_hashes = _state_of(FILES_V1)
    plan_noop = plan_sync(v1_hashes, v1_hashes)
    print(f"\n[no-op resync] changed={plan_noop.changed}")
    if plan_noop.changed:
        _fail(f"identical current/prior state must yield zero changed files, got {plan_noop.changed}")
    if len(plan_noop.unchanged) != len(FILES_V1):
        _fail("no-op resync should report every file unchanged")
    print("  -> GREEN precondition OK: identical snapshot reprocesses nothing")

    # --- RED: edit exactly ONE of 6 files; only it must be flagged ----------
    FILES_V2 = dict(FILES_V1)
    FILES_V2["rfc/rfcx1002.txt"] = (
        "RFCX1002 defines the Gadget Naming Scheme, revision 2 (adds a namespace field)."
    )
    v2_hashes = _state_of(FILES_V2)
    plan = plan_sync(v2_hashes, v1_hashes)
    print(f"\n[one file edited] changed={plan.changed}")
    if plan.changed != ["rfc/rfcx1002.txt"]:
        _fail(f"expected exactly ['rfc/rfcx1002.txt'] changed, got {plan.changed}")
    if len(plan.unchanged) != len(FILES_V1) - 1:
        _fail(f"expected {len(FILES_V1) - 1} unchanged files, got {len(plan.unchanged)}")
    print("  -> RED caught: exactly the edited file is flagged, nothing else")

    # --- HEAD-TO-HEAD: incumbent "reprocess everything" vs SyncGate's diff
    naive_reprocessed = len(v2_hashes)          # the incumbent: N/N, every run
    real_reprocessed = len(plan.changed)        # SyncGate: only what actually changed
    reduction = plan.reduction_factor
    print(f"\n=== HEAD-TO-HEAD: reprocess-everything vs SyncGate incremental diff ===")
    print(
        f"  incumbent (reprocess everything): {naive_reprocessed}/{naive_reprocessed} files\n"
        f"  SyncGate (incremental diff)     : {real_reprocessed}/{naive_reprocessed} files\n"
        f"  measured reduction              : {reduction:.1f}x fewer files reprocessed"
    )
    if real_reprocessed != 1:
        _fail(f"expected 1 file reprocessed by SyncGate, got {real_reprocessed}")
    if reduction < float(len(FILES_V1)) - 1e-9:
        _fail(f"expected an {len(FILES_V1)}x reduction (1 of {len(FILES_V1)}), measured {reduction:.2f}x")

    # sanity: changed_files() alone (the primitive) agrees with plan_sync()
    if changed_files(v2_hashes, v1_hashes) != plan.changed:
        _fail("changed_files() and plan_sync().changed disagree")

    # --- reference integrity: three edges, three distinct fates ------------
    edges = [
        RefEdge("RFCX1003", "RFCX1001", "Obsoletes"),   # both live -> resolves
        RefEdge("RFCX1004", "RFCX2099", "Obsoletes"),   # renamed; cached snapshot survives
        RefEdge("ERRX-9002", "RFCX3333", "Corrects"),   # never existed; no cache either
    ]
    live_entities = {"RFCX1001", "RFCX1002", "RFCX1003", "RFCX1004"}
    cached_entities = {"RFCX2099"}  # last-known-good snapshot of the renamed/superseded doc

    # --- the naive incumbent: a plain presence check (what dict.get gives)
    naive_missing = {e.to_id: (e.to_id not in live_entities) for e in edges}
    closure_violations = {e.to_id for e in reference_closure(edges, live_entities)}
    print(f"\n[naive presence check] missing-from-live: {naive_missing}")
    if naive_missing != {"RFCX1001": False, "RFCX2099": True, "RFCX3333": True}:
        _fail(f"unexpected naive presence-check result: {naive_missing}")
    if closure_violations != {"RFCX2099", "RFCX3333"}:
        _fail(f"reference_closure should flag exactly the two absent targets, got {closure_violations}")
    if naive_missing["RFCX2099"] != naive_missing["RFCX3333"]:
        _fail("the naive check should flag BOTH absent targets identically -- that IS the blind spot")
    print(
        "  -> naive check flags RFCX2099 and RFCX3333 IDENTICALLY (both 'missing'): "
        "it cannot tell a renamed target from one that never existed"
    )

    # --- SyncGate's three-way fallback DOES distinguish them ----------------
    resolutions = resolve_references(edges, live_entities, cached_entities)
    by_target = {r.edge.to_id: r for r in resolutions}
    print("\n[SyncGate three-way fallback]")
    for r in resolutions:
        print(f"  {r.edge.from_id} -> {r.edge.to_id} ({r.edge.reason}): {r.status.value} -- {r.detail}")

    if by_target["RFCX1001"].status is not RefStatus.LIVE:
        _fail("RFCX1001 is in the live set and should resolve LIVE")
    if by_target["RFCX2099"].status is not RefStatus.STALE_CACHED:
        _fail("RFCX2099 has a cached snapshot and should resolve STALE_CACHED, not DANGLING")
    if by_target["RFCX3333"].status is not RefStatus.DANGLING:
        _fail("RFCX3333 has no live target and no cache; must resolve DANGLING")
    print(
        "  -> RED caught: DANGLING correctly isolated to RFCX3333 only, naming the exact\n"
        "     broken edge (from=ERRX-9002, to=RFCX3333, reason=Corrects); RFCX2099 is\n"
        "     correctly served STALE_CACHED instead of being conflated with a hard failure"
    )

    # --- the combined gate: passes iff no DANGLING (STALE_CACHED is not a fail)
    gate_red = run_gate(v2_hashes, v1_hashes, edges, live_entities, cached_entities)
    print(f"\n{gate_red.report()}")
    if gate_red.passed:
        _fail("gate should NOT pass while a DANGLING reference exists")
    if len(gate_red.dangling) != 1 or gate_red.dangling[0].edge.to_id != "RFCX3333":
        _fail(f"expected exactly one DANGLING resolution (RFCX3333), got {gate_red.dangling}")
    if len(gate_red.stale_cached) != 1 or gate_red.stale_cached[0].edge.to_id != "RFCX2099":
        _fail(f"expected exactly one STALE_CACHED resolution (RFCX2099), got {gate_red.stale_cached}")

    # --- GREEN: fix the dangling edge (target now live) -> gate passes ------
    live_fixed = live_entities | {"RFCX3333"}
    gate_green = run_gate(v2_hashes, v1_hashes, edges, live_fixed, cached_entities)
    print(f"\n[fixed: RFCX3333 now live]\n{gate_green.report()}")
    if not gate_green.passed:
        _fail("gate should pass once every reference target is live or cached")
    if gate_green.dangling:
        _fail(f"no DANGLING resolutions should remain, got {gate_green.dangling}")

    # --- determinism check: identical inputs -> identical result -----------
    repeat = run_gate(v2_hashes, v1_hashes, edges, live_entities, cached_entities)
    if [r.status for r in repeat.resolutions] != [r.status for r in gate_red.resolutions]:
        _fail("non-deterministic resolution across identical runs")
    if repeat.plan.changed != gate_red.plan.changed:
        _fail("non-deterministic sync plan across identical runs")

    print(
        "\nPASS: sync diff exact (1/6 flagged, "
        f"{reduction:.1f}x reduction vs reprocess-everything); reference fallback "
        "correctly separates STALE_CACHED from DANGLING where a naive presence "
        "check sees only one undifferentiated 'missing' bucket. Deterministic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
