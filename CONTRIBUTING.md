# Contributing

Portfolio-tier toolkit: small, offline, deterministic tools, each independently runnable.

## Self-tests

```
pip install -r requirements.txt
python run_all_evals.py        # runs every tool's red/green self-test
python <tool>/eval.py          # run one tool in isolation
```

Both must exit `0`.

## Changes

- Each tool stays self-contained: own directory, own `eval.py`, no cross-tool dependencies.
- `eval.py` (and `bench.py`, where present) must still exit `0`. Extend the red/green fixture for a new failure mode.
- Deterministic (fixed seeds, no wall-clock, no network) and offline (numpy + stdlib only).
- Open a PR against `main`. CI (`.github/workflows/test.yml`) runs `run_all_evals.py` on Python 3.10/3.11/3.12.
