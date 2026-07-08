# Headroom — hardware-aware agentic-RAG budget governor

An agentic RAG system can keep retrieving and reasoning in a loop, chasing a better answer one more
hop at a time. Headroom is the check that stops it from doing that when the GPU running it is
about to run out of memory or overheat, even if the software logic thinks one more hop is a good
idea.

More precisely: it is a tiny library a RAG framework consults before escalating another retrieval /
reasoning hop. It reads GPU telemetry — VRAM headroom, thermal margin to the
throttle/abort line, and the measured per-hop latency budget — and returns
`ALLOW` / `DEFER` / `DENY`. The escalation decision becomes physical, not just
economic.

## What's new here (and what isn't)

Existing adaptive-RAG governors decide *whether to do another hop* from
**token cost and answer sufficiency**:

- **Adaptive-RAG** (Jeong et al., 2024) — a classifier routes queries to
  no-retrieval / single-hop / multi-hop by predicted complexity.
- **A2RAG / Adaptive-Agentic RAG** — agentic loops that escalate hops under a
  reasoning/sufficiency signal.
- **CA-RAG (cost-aware RAG)** — budgets retrieval against **token / dollar** cost
  and a sufficiency estimate.

All of these gate on *software economics* (tokens, latency-as-cost, sufficiency).
**None gate on measured hardware physics.** Headroom's wedge — and the only thing
claimed as novel here — is a RAG hop-governor whose gate is **measured VRAM /
watts / thermal margin**: it will `DEFER`/`DENY` a hop that *fits the token
budget and would improve the answer* purely because the next hop is predicted to
cross the card's VRAM ceiling or thermal abort line. It is complementary to the
above, not a replacement — you would run a sufficiency governor *and* Headroom.

This models an illustrative, configurable failure mode (KV-cache growth pushing
a card toward its VRAM ceiling and thermal abort line): the software layer
thinks another hop is a great idea; the hardware disagrees. The `card-a`
/ `card-b` profiles below are example `GpuProfile` configurations, not a
report of any specific machine's measured tuning history — swap in your own
card's numbers via the same `GpuProfile` interface.

## What is real vs. simulated

| Piece | Status |
|---|---|
| Governor decision logic (`headroom.py`) | **Real.** Decisions derive from observed telemetry + a **measured per-hop slope** extrapolated from history. No hop index is special-cased; no verdict hard-coded. |
| Per-hop retrieval (`loop.py`, `embedder.py`) | **Real** (toy). A vendored pure-python BLAKE2b hashing embedder (dim 256, L2-norm, cosine) does actual retrieval so a "hop" does work. |
| GPU telemetry (VRAM/temp/latency) | **SIMULATED**, deterministically. `Trajectory` replays a scripted schedule (with fixed-seed jitter) that stands in for `nvidia-smi`. The governor never knows it's simulated — it only sees numbers. |
| The 90 °C abort / 15000 MB wedge / sufficiency-never-satisfied | **SIMULATED** fixtures, clearly labelled, illustrating one example `card-a` `GpuProfile` configuration. |

To use against real hardware you replace `Trajectory.sample()` with an
`nvidia-smi` / NVML reader returning the same `GpuState`. The governor is
unchanged.

## Honest scope

- This is an **MVP first milestone**, not a product. One profile family
  (`card-a`, `card-b`), a linear-slope predictor, three gate signals.
- The predictor is a two-sample linear extrapolation. It is deliberately simple;
  a real deployment wants a smoothed / worst-case estimator and hysteresis on
  `DEFER`→`ALLOW` recovery.
- No watts/power-draw signal yet (the wedge claims it but the MVP gates on VRAM +
  thermal + latency). Power is a straightforward fourth signal on the same shape.
- Numbers in the demo are **simulated fixtures**, not measured hardware.

## Usage

```python
from headroom import CARD_A, Headroom, GpuState

gov = Headroom(CARD_A)
gov.observe(GpuState(hop=1, vram_used_mb=11500.0, temp_c=82.0, last_hop_ms=800.0, elapsed_ms=800.0))

decision = gov.gate()          # ALLOW / DEFER / DENY, consulted before the next hop
print(decision, gov.log[-1].reason)
```

Call `observe()` with each completed hop's telemetry, then `gate()` before
deciding whether to escalate to the next one. Swap in your own `GpuProfile`
(see `CARD_A`/`CARD_B` above) for a real card's numbers.

## Run the self-test

```
# from the repo root
python headroom/eval.py
```

`eval.py` has four parts:

- **RED** — the ungoverned loop on the near-ceiling trajectory *actually
  breaches* (thermal abort 91.6 °C, VRAM 15270 MB). The **same** trajectory under
  Headroom is stopped **before** the breach by the real mechanism (predicted next
  hop 89.7 °C / 14656 MB in the defer band), and the gate's reason + telemetry are
  logged.
- **GREEN** — on an ample-headroom trajectory Headroom allows every hop with
  **zero** false DENY/DEFER and the loop completes.
- **FAIR-BASELINE** — proves the comparator is a *real* governor, not a rigged
  no-op: the token/sufficiency incumbent (`baselines.CostGovernor`) is shown to
  stop correctly on **sufficiency** (answer good enough after 2 hops) and on its
  **token budget** (~7828/8192 tok at hop 19 of a long safe run), while Headroom
  allows that entire hardware-safe run — economics and physics are complementary.
- **HEAD-TO-HEAD** — the money demo. Over a **scenario set** of four card-a
  trajectories (thermal-led, vram-led, mixed, slow-climb), the identical telemetry
  is run under the incumbent and under Headroom.

### Measured result (the wedge, not asserted)

The incumbent is a *fair* token-cost + sufficiency governor (Adaptive-RAG /
CA-RAG family): one fixed config, `TOKEN_BUDGET = 8192` (a common context
window), `TOKENS_PER_HOP` **derived from the real corpus** (~412 tok/hop). On the
dangerous set its budget is nowhere near binding (~1236–1648 of 8192 tok spent)
and the answer is still insufficient, so its economics say "keep going" — and it
walks straight into the wall:

| governor | breach rate on the dangerous set | measured |
|---|---|---|
| token+sufficiency incumbent (CA-RAG-style) | **4/4 = 100 %** | breaches at hop 3–4 with tokens/sufficiency both un-bound |
| Headroom (hardware-physics gate) | **0/4 = 0 %** | stops 1–2 hops before each breach, `stopped_by=governor` |

**Measured gap: 100 % of the physical breaches prevented** — the direct,
reproduced proof of the wedge (a cost governor cannot see VRAM MB / °C, so it
cannot avoid a wedge a hardware gate stops). Neither governor over-blocks the two
SAFE scenarios (both run to completion). Numbers are **simulated fixtures** (see
the table above), not measured hardware.

Exit `0` iff all four parts pass. Deterministic: `numpy.random.default_rng(SEED)`,
no wall-clock, no `random`.

## Files

- `headroom.py` — `GpuProfile`, `GpuState`, `Trajectory` (sim telemetry),
  `Headroom` governor (`observe` / `gate`), `Decision`, `GateLog`.
- `loop.py` — minimal agentic-RAG loop; runs governed or ungoverned.
- `embedder.py` — vendored deterministic hashing embedder (offline, numpy-only).
- `baselines.py` — the **incumbent** `CostGovernor` (token-cost + sufficiency,
  Adaptive-RAG / CA-RAG family) Headroom is measured against.
- `eval.py` — RED/GREEN self-test **plus** the fair-baseline check and the
  measured head-to-head vs the incumbent.

## Prior art (cited)

- Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs
  through Question Complexity*, 2024.
- Adaptive / agentic RAG hop-escalation lines of work (A2RAG and related).
- Cost-aware RAG (CA-RAG) — token/dollar-budgeted retrieval with a sufficiency
  estimate.

The only part Headroom claims as new is the hardware-physics gate (measured VRAM /
thermal / latency margin as the hop-escalation criterion), not on adaptive
retrieval, agentic loops, or cost-aware budgeting in general.
