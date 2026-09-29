"""A minimal agentic-RAG loop, deterministic and offline.

Each "hop" = one retrieval + reasoning step that grows the KV cache and heats
the card. A real framework decides to escalate to another hop when the current
answer is judged insufficient. Here the sufficiency signal is SIMULATED (the toy
query is never satisfied, so an *ungoverned* loop keeps escalating until it hits
the hardware wall). The point of the demo is the hardware gate, not the answerer.

Run modes:
  * ungoverned: escalates for all scheduled hops (what frameworks do today).
  * governed:   consults a Headroom governor BEFORE each escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embedder import embed, embed_batch
from headroom import Decision, GpuProfile, Headroom, Trajectory


@dataclass
class RunResult:
    hops_executed: int
    breached: bool
    breach_reason: str
    stopped_by: str            # "abort" | "governor" | "schedule-end"
    peak_temp_c: float
    peak_vram_mb: float
    telemetry: list = field(default_factory=list)


# tiny fixed corpus so a hop does real (toy) retrieval work
_CORPUS = [
    "card a wedges near fifteen thousand megabytes of vram",
    "thermal abort on card a is ninety celsius",
    "adaptive rag escalates hops based on token cost and sufficiency",
    "hardware aware gating reads vram watts and thermal margin",
]
_CORPUS_VEC = embed_batch(_CORPUS)


def _retrieve(query: str) -> int:
    """Toy cosine retrieval: real mechanism, just to make a hop do work."""
    q = embed(query)
    sims = _CORPUS_VEC @ q
    return int(np.argmax(sims))


def run_loop(traj: Trajectory, profile: GpuProfile, *, governor: Headroom | None) -> RunResult:
    telem = []
    n = len(traj)

    # hop 0: baseline load (the retrieval that always happens)
    _retrieve("initial question about the card")
    s0 = traj.sample(0)
    if governor is not None:
        governor.observe(s0)
    telem.append(s0)
    # seed peak telemetry from the baseline sample so a governor DENY/DEFER on
    # the very first gate() call (before any hop executes) still reports the
    # REAL observed baseline, not an impossible 0.0.
    peak_t = s0.temp_c
    peak_v = s0.vram_used_mb

    for hop in range(1, n + 1):
        # BEFORE escalating: consult the governor (if any)
        if governor is not None:
            dec = governor.gate()
            if dec in (Decision.DENY, Decision.DEFER):
                return RunResult(hop - 1, False, "", "governor", peak_t, peak_v, telem)

        # escalate: do the hop, then read telemetry
        _retrieve(_CORPUS[(hop - 1) % len(_CORPUS)])
        s = traj.sample(hop)
        telem.append(s)
        peak_t = max(peak_t, s.temp_c)
        peak_v = max(peak_v, s.vram_used_mb)

        # hardware truth: did this hop breach a physical limit?
        if s.temp_c >= profile.abort_temp_c:
            return RunResult(hop, True, f"thermal abort {s.temp_c:.1f}C >= "
                             f"{profile.abort_temp_c:.0f}C", "abort", peak_t, peak_v, telem)
        if s.vram_used_mb >= profile.vram_hard_mb:
            return RunResult(hop, True, f"vram wedge {s.vram_used_mb:.0f}MB >= "
                             f"{profile.vram_hard_mb:.0f}MB", "abort", peak_t, peak_v, telem)

        if governor is not None:
            governor.observe(s)

    return RunResult(n, False, "", "schedule-end", peak_t, peak_v, telem)
