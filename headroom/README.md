# Headroom

Hardware-aware budget governor for agentic RAG.

Consulted before each retrieval/reasoning hop. Reads VRAM headroom, thermal margin to the
throttle/abort line, and per-hop latency. Returns `ALLOW` / `DEFER` / `DENY`.

## Prior art

- Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question
  Complexity*, 2024
- A2RAG and related adaptive/agentic hop-escalation work
- CA-RAG: token/dollar-budgeted retrieval with a sufficiency estimate

All gate on tokens, cost or sufficiency. Headroom gates on VRAM, thermal and latency margin. It
complements them.

## Real vs simulated

| Piece | Status |
|---|---|
| Governor decision logic (`headroom.py`) | Real. Observed telemetry plus a per-hop slope extrapolated from history. |
| Per-hop retrieval (`loop.py`, `embedder.py`) | Real, toy. BLAKE2b hashing embedder, dim 256. |
| GPU telemetry (VRAM/temp/latency) | Simulated, deterministic. `Trajectory` replays a scripted schedule. |
| 90 °C abort / 15000 MB wedge / sufficiency never satisfied | Simulated fixtures (`card-a` profile). |

For real hardware, replace `Trajectory.sample()` with an `nvidia-smi` / NVML reader returning
`GpuState`.

## Limitations

- MVP. One profile family (`card-a`, `card-b`), linear-slope predictor, three signals.
- Two-sample linear extrapolation. No smoothing, worst-case estimate or hysteresis on
  `DEFER` to `ALLOW`.
- No watts/power signal yet.
- Demo numbers are simulated fixtures.

## Usage

```python
from headroom import CARD_A, Headroom, GpuState

gov = Headroom(CARD_A)
gov.observe(GpuState(hop=1, vram_used_mb=11500.0, temp_c=82.0, last_hop_ms=800.0, elapsed_ms=800.0))

decision = gov.gate()          # ALLOW / DEFER / DENY, consulted before the next hop
print(decision, gov.log[-1].reason)
```

`observe()` per completed hop, `gate()` before the next. Custom card: your own `GpuProfile`.

## Self-test

```
# from the repo root
python headroom/eval.py
```

- RED: ungoverned loop on the near-ceiling trajectory breaches (91.6 °C, 15270 MB). Under
  Headroom the same trajectory stops before the breach (predicted 89.7 °C / 14656 MB).
- GREEN: ample-headroom trajectory, every hop allowed, zero false DENY/DEFER.
- FAIR-BASELINE: the token/sufficiency incumbent (`baselines.CostGovernor`) stops on sufficiency
  (2 hops) and on token budget (~7828/8192 tok at hop 19). Headroom allows that whole run.
- HEAD-TO-HEAD: four `card-a` trajectories (thermal-led, vram-led, mixed, slow-climb), same
  telemetry under both governors.

Exit `0` iff all four pass. Deterministic: `numpy.random.default_rng(SEED)`, no `random`,
no wall-clock.

## Results

Incumbent: `TOKEN_BUDGET = 8192`, `TOKENS_PER_HOP` derived from the corpus (~412 tok/hop).

| governor | breach rate, dangerous set | notes |
|---|---|---|
| token+sufficiency incumbent (CA-RAG-style) | 4/4 = 100 % | breaches at hop 3-4, ~1236-1648 of 8192 tok spent |
| Headroom | 0/4 = 0 % | stops 1-2 hops before each breach, `stopped_by=governor` |

Neither governor over-blocks the two safe scenarios. Both run to completion. Numbers are simulated fixtures.

## Files

- `headroom.py`: `GpuProfile`, `GpuState`, `Trajectory`, `Headroom` (`observe` / `gate`),
  `Decision`, `GateLog`
- `loop.py`: agentic-RAG loop, governed or ungoverned
- `embedder.py`: vendored hashing embedder (numpy only)
- `baselines.py`: incumbent `CostGovernor`
- `eval.py`: self-test, fair-baseline check, head-to-head
