"""One worked GraphRx run: lint a flawed synthetic GraphRAG graph and validate the
top repair with the poisoning-delta harness.

    python graphrx/run_demo.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from graphrx import fixtures                    # noqa: E402
from graphrx.lint import report                 # noqa: E402
from graphrx.probe import poisoning_delta       # noqa: E402

SEED = 20260704


def main():
    g = fixtures.build_flawed_graph(SEED)
    defects = report(g)

    print(f"graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges, "
          f"{len(g['facts'])} facts\n")
    print("RANKED DEFECTS (by GraphRAG answer-poisoning risk):")
    for i, d in enumerate(defects, 1):
        print(f"  {i}. [{d['score']:.3f}] {d['kind']} -> {d['target']}")
        print(f"       evidence: {d['evidence']}")
        print(f"       proposal: {d['proposal']['op']}")

    top = defects[0]
    print(f"\nVALIDATING top repair ({top['kind']} on {top['target']}) with the "
          "poisoning-delta harness:")
    d = poisoning_delta(g, top["proposal"], fixtures.PLANTED["collision_entities"][0],
                        mode="local", depth=2)
    print(f"  local-search poisoning  before={d['before']:.3f}  after={d['after']:.3f}"
          f"  delta={d['delta']:+.3f}")
    print("  -> repair " + ("REDUCES" if d["delta"] > 0 else "does not reduce")
          + " retrieval poisoning.")


if __name__ == "__main__":
    main()
