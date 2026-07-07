"""Headroom -- a hardware-aware budget governor for agentic RAG.

A framework consults Headroom BEFORE escalating to another retrieval/reasoning
hop. Headroom reads GPU telemetry -- VRAM headroom, thermal margin to a
throttle/abort line, and the measured per-hop latency budget -- and returns
ALLOW / DEFER / DENY. Unlike token-cost / sufficiency governors (Adaptive-RAG,
A2RAG, CA-RAG), the gate here is *physical*: measured VRAM, degrees C to the
abort line, and milliseconds to the deadline.

Everything hardware is SIMULATED deterministically (clearly labelled). The
`Trajectory` produces a scripted GPU telemetry stream; the governor never sees
"it's simulated" -- it only sees numbers, exactly as it would from nvidia-smi.

The governing mechanism is honest: the decision is derived from the observed
telemetry plus the *measured* per-hop slope (extrapolated from history). No
hop index is special-cased; no verdict is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

SEED = 0xEAD00


class Decision(str, Enum):
    ALLOW = "allow"
    DEFER = "defer"  # back off / cool down / evict KV, then retry
    DENY = "deny"    # do not escalate -- answer with what you have


@dataclass(frozen=True)
class GpuProfile:
    """Physical limits of the card the loop is running on."""

    name: str
    vram_total_mb: float
    vram_hard_mb: float      # >= this = wedge / OOM (the abort ceiling)
    abort_temp_c: float      # >= this = thermal abort
    deadline_ms: float       # per-request latency budget

    # governor safety margins (how much daylight to keep before the hard line)
    vram_margin_mb: float = 1024.0
    thermal_margin_c: float = 2.0
    # how close a *prediction* may get before we DEFER (softer than DENY)
    vram_defer_mb: float = 1536.0
    thermal_defer_c: float = 3.5


# --- Reference profiles (illustrative, configurable via GpuProfile) -----------
# "card-a" example: a smaller-VRAM, lower-thermal-margin card configuration.
CARD_A = GpuProfile(
    name="card-a",
    vram_total_mb=16384.0,
    vram_hard_mb=15000.0,   # illustrative VRAM ceiling for this profile
    abort_temp_c=90.0,
    deadline_ms=6000.0,
)

# "card-b" example: a card configuration with more headroom, cooler, faster.
CARD_B = GpuProfile(
    name="card-b",
    vram_total_mb=24000.0,
    vram_hard_mb=21000.0,
    abort_temp_c=95.0,
    deadline_ms=8000.0,
)


@dataclass(frozen=True)
class GpuState:
    """One telemetry sample (what nvidia-smi would report)."""

    hop: int
    vram_used_mb: float
    temp_c: float
    last_hop_ms: float
    elapsed_ms: float


@dataclass
class GateLog:
    """The audited record of a single gate() call."""

    decision: Decision
    reason: str
    observed_temp_c: float
    observed_vram_mb: float
    pred_temp_c: float
    pred_vram_mb: float
    pred_hop_ms: float
    elapsed_ms: float


class Trajectory:
    """Deterministic SIMULATED telemetry stream for one run.

    `schedule` is the scripted post-hop telemetry (vram_mb, temp_c, hop_ms) for
    hops 1..N. Tiny seeded jitter is added so the numbers look like real
    hardware without being random across runs.
    """

    def __init__(self, schedule, *, seed: int = SEED, jitter_vram=8.0, jitter_temp=0.15):
        self._schedule = list(schedule)
        self._rng = np.random.default_rng(seed)
        self._jv = jitter_vram
        self._jt = jitter_temp
        self._elapsed = 0.0

    def sample(self, hop: int) -> GpuState:
        """Telemetry AFTER completing `hop` (1-indexed). hop=0 = idle baseline."""
        if hop <= 0:
            base = self._schedule[0]
            return GpuState(0, base[0] * 0.92, base[1] - 4.0, 0.0, 0.0)
        vram, temp, hop_ms = self._schedule[hop - 1]
        vram += float(self._rng.normal(0.0, self._jv))
        temp += float(self._rng.normal(0.0, self._jt))
        self._elapsed += hop_ms
        return GpuState(hop, vram, temp, hop_ms, self._elapsed)

    def __len__(self) -> int:
        return len(self._schedule)


class Headroom:
    """The governor. Stateful: observe() each completed hop, gate() before the next."""

    def __init__(self, profile: GpuProfile):
        self.profile = profile
        self._history: list[GpuState] = []
        self.log: list[GateLog] = []

    def observe(self, state: GpuState) -> None:
        self._history.append(state)

    def _slope(self, key):
        """Measured per-hop rate of change from the last two observations."""
        if len(self._history) < 2:
            return 0.0
        a, b = self._history[-2], self._history[-1]
        return getattr(b, key) - getattr(a, key)

    def gate(self) -> Decision:
        """Ask permission to escalate one more hop. Requires >=1 observation."""
        p = self.profile
        if not self._history:
            # cold: no telemetry yet -- allow the first hop unconditionally.
            self._record(Decision.ALLOW, "cold-start (no telemetry)",
                         GpuState(0, 0, 0, 0, 0), 0, 0, 0)
            return Decision.ALLOW

        s = self._history[-1]

        # measured slopes -> predict the state AFTER the next hop
        d_temp = self._slope("temp_c")
        d_vram = self._slope("vram_used_mb")
        d_ms = self._slope("last_hop_ms")
        pred_temp = s.temp_c + d_temp
        pred_vram = s.vram_used_mb + d_vram
        pred_hop_ms = max(s.last_hop_ms + d_ms, s.last_hop_ms)

        # --- static guards: already too close to the line -----------------
        if s.temp_c >= p.abort_temp_c - p.thermal_margin_c:
            return self._deny(s, pred_temp, pred_vram, pred_hop_ms,
                              f"thermal: observed {s.temp_c:.1f}C within "
                              f"{p.thermal_margin_c:.1f}C of {p.abort_temp_c:.0f}C abort")
        if s.vram_used_mb >= p.vram_hard_mb - p.vram_margin_mb:
            return self._deny(s, pred_temp, pred_vram, pred_hop_ms,
                              f"vram: observed {s.vram_used_mb:.0f}MB within "
                              f"{p.vram_margin_mb:.0f}MB of {p.vram_hard_mb:.0f}MB ceiling")

        # --- predictive guards: next hop would cross the line -------------
        if pred_temp >= p.abort_temp_c:
            return self._deny(s, pred_temp, pred_vram, pred_hop_ms,
                              f"thermal: next hop predicted {pred_temp:.1f}C "
                              f">= {p.abort_temp_c:.0f}C abort (slope {d_temp:+.1f}C/hop)")
        if pred_vram >= p.vram_hard_mb:
            return self._deny(s, pred_temp, pred_vram, pred_hop_ms,
                              f"vram: next hop predicted {pred_vram:.0f}MB "
                              f">= {p.vram_hard_mb:.0f}MB ceiling (slope {d_vram:+.0f}MB/hop)")

        # --- softer DEFER band: close but not over -----------------------
        if pred_temp >= p.abort_temp_c - p.thermal_defer_c:
            return self._defer(s, pred_temp, pred_vram, pred_hop_ms,
                               f"thermal: next hop predicted {pred_temp:.1f}C in "
                               f"{p.thermal_defer_c:.1f}C defer band -- cool down / evict")
        if pred_vram >= p.vram_hard_mb - p.vram_defer_mb:
            return self._defer(s, pred_temp, pred_vram, pred_hop_ms,
                               f"vram: next hop predicted {pred_vram:.0f}MB in defer band "
                               f"-- evict KV before escalating")

        # --- latency budget ----------------------------------------------
        if s.elapsed_ms + pred_hop_ms > p.deadline_ms:
            return self._defer(s, pred_temp, pred_vram, pred_hop_ms,
                               f"latency: {s.elapsed_ms:.0f}ms + predicted {pred_hop_ms:.0f}ms "
                               f"> {p.deadline_ms:.0f}ms deadline")

        return self._allow(s, pred_temp, pred_vram, pred_hop_ms)

    # -- log helpers -----------------------------------------------------------
    def _record(self, dec, reason, s, pt, pv, pm):
        self.log.append(GateLog(dec, reason, s.temp_c, s.vram_used_mb, pt, pv, pm, s.elapsed_ms))
        return dec

    def _allow(self, s, pt, pv, pm):
        return self._record(Decision.ALLOW, "headroom ok", s, pt, pv, pm)

    def _defer(self, s, pt, pv, pm, reason):
        return self._record(Decision.DEFER, reason, s, pt, pv, pm)

    def _deny(self, s, pt, pv, pm, reason):
        return self._record(Decision.DENY, reason, s, pt, pv, pm)
