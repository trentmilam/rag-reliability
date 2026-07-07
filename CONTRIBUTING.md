# Contributing

This is a portfolio-tier toolkit (see the top-level README's "Status" section) —
small, offline, deterministic tools, each independently runnable.

## Running the self-tests

```
pip install -r requirements.txt
python run_all_evals.py        # runs every tool's red/green self-test
python <tool>/eval.py          # run one tool in isolation
```

Both must exit `0`.

## Making a change

- Keep each tool self-contained (its own directory, its own `eval.py`); avoid
  adding cross-tool dependencies.
- Every change to a tool's core logic needs its `eval.py` (and `bench.py`,
  where present) to still exit `0` — extend the red/green fixture if the
  change adds a new failure mode worth catching.
- Keep everything deterministic (fixed seeds, no wall-clock, no network) and
  offline (numpy + stdlib only).
- Open a PR against `main`; CI (`.github/workflows/test.yml`) runs
  `run_all_evals.py` across Python 3.10/3.11/3.12.
