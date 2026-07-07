"""Run every tool's red/green self-test and report one PASS/FAIL for the repo.

Discovers every `<tool>/eval.py` under the repo root, runs each with the
current Python interpreter (matching the documented `python <tool>/eval.py`
invocation), and aggregates the exit codes.

Exit 0 iff every tool's eval.py exits 0.

Usage:
    python run_all_evals.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    evals = sorted(ROOT.glob("*/eval.py"))
    if not evals:
        print("FAIL: no <tool>/eval.py files found under the repo root")
        return 1

    results: list[tuple[str, bool, int]] = []
    for path in evals:
        tool = path.parent.name
        rel = path.relative_to(ROOT).as_posix()
        print("=" * 70)
        print(f"[{tool}] python {rel}")
        print("=" * 70)
        proc = subprocess.run([sys.executable, rel], cwd=ROOT)
        ok = proc.returncode == 0
        results.append((tool, ok, proc.returncode))
        print(f"[{tool}] {'PASS' if ok else 'FAIL'} (exit {proc.returncode})\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for tool, ok, rc in results:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {tool:14s} (exit {rc})")

    n_ok = sum(1 for _, ok, _ in results if ok)
    overall_ok = n_ok == len(results)
    print(f"\n{n_ok}/{len(results)} tools passed.")
    print("OVERALL: PASS" if overall_ok else "OVERALL: FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
