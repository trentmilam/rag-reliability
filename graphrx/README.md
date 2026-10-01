# GraphRx: a GraphRAG structural linter with retrieval-poisoning scoring

GraphRAG is a variant of RAG that retrieves from a graph of entities and relationships instead of
plain text chunks. GraphRx checks that graph for structural defects that would poison answers:
two unrelated entities merged into one node, a community of topics merged that shouldn't be, and
ranks them by how much damage they'd actually do, not by generic graph-health scores.

More precisely: point GraphRx at a built GraphRAG graph (entities, relationships, and communities,
as a plain Python adjacency dict) and it returns a ranked report of structural defects: ER
(entity-resolution) collisions, over-/under-merged communities, hub pollution, orphans, where each
defect is scored by how much it poisons GraphRAG answers, not by generic graph-health
metrics. Every proposed merge/split repair is then validated by a before/after retrieval-
poisoning-delta harness: a repair only earns credit if it measurably lowers poisoning on a
traversal probe.

## Quickstart

```
python graphrx/eval.py      # RED/GREEN self-test (exit 0)
python graphrx/run_demo.py  # one worked graph
```

## Measured (eval.py, exit 0)

On a clean synthetic graph the linter reports **zero high-risk defects** (max node fact-
separation 0.4286, max community member-separation 0.2965; both under the 0.5 threshold). On a
graph with a **planted ER collision** (a bank + a biotech merged into one node) and a **planted
over-merged community** (aviation + sports fused), GraphRx:

- flags both via its unsupervised signals (collision node separation **1.0000**, over-merged
  community separation **1.0000**) and **ranks them at the top** (orphan scored 0.05, correctly
  low, an orphan is a *recall* defect, not a poisoning one);
- its **collision split** cuts local-search poisoning from **0.5714 to 0.0000** (delta +0.5714)
  with a 100%-pure fact separation, and its **community split** cuts global-search poisoning
  from **0.5000 to 0.0000** (delta +0.5000);
- negative control: splitting a clean, coherent node yields **delta −0.0357** (no gain); the
  harness rewards only real repairs, so the green result is not rigged.

### Measured A/B vs naive generic graph-health

The claim, "scored by how much it poisons *answers*, not by generic graph-health metrics," is
now a printed A/B, not prose. On the **same flawed graph**, three fair, standard graph-health
baselines (no poisoning lens, no embeddings) are scored by how many of the **two planted
poisoners** (the collision **node** `apex` + the over-merged **community** `aviation_sports`) each
recovers in its **top-2**, and how many top-2 slots are false positives:

| method | top-2 | poisoners recovered | false positives |
|---|---|---|---|
| **GraphRx (poisoning lens)** | `apex`, `aviation_sports` | **2 / 2** | **0** |
| degree centrality | `apex`, `aerodyne` | 1 / 2 | 1: *node metric, blind to the over-merged community* |
| degree anomaly | `apex`, `lonelycorp` | 1 / 2 | 1: *flags the harmless orphan* |
| community edge-density | `aviation_sports`, `finance` | 1 / 2 | 1: *community metric, blind to the single-node collision* |

**No single generic structural metric recovers both poisoners; GraphRx recovers both with zero
false positives.** Degree centrality does rank `apex` #1, but only by a **1-edge margin** (degree
4 vs 3) over a *legitimate* hub it cannot distinguish, whereas GraphRx's poisoning-risk separates
them totally (**1.0000 vs 0.0000**: the legit hub is never flagged). Baselines live in
`graphrx/baseline.py`; the scored comparison prints in `eval.py`.

Detectors read **only structure + a vendored hashing embedder**, never the ground-truth labels
(those are read solely by the probe, exactly like a labeled test set), so the red/green result
reflects the mechanism genuinely discovering the fault, not a hard-coded verdict.

## How it works

Note on layout: unlike the other tools in this repo, GraphRx's modules live in a
nested `graphrx/graphrx/` package directory (so `import graphrx` works from the
tool's top directory) rather than flat alongside `eval.py`. File references
below (`graphrx/embed.py` etc.) are relative to this tool's top directory,
i.e. the full repo-root-relative path is `graphrx/graphrx/embed.py`.

- Embedding backend (SIMULATED, labeled as such): a vendored pure-Python BLAKE2b hashing
  embedder (`graphrx/embed.py`, dim 256, L2-normalized, cosine), no model download, no network. It
  captures the lexical overlap that carries topic/entity identity, which is the signal the checks
  need. A real deployment swaps in a sentence embedder behind the same interface.
- Detectors (`graphrx/lint.py`, unsupervised):
  - *ER collision*: a node whose facts split into two well-separated embedding clusters (2-means
    separation ≥ 0.5) **and** whose degree bridges communities, which produces *conflated evidence*
    plus *traversal fan-out*. risk = separation × reach(degree).
  - *Over-merged community*: a community whose member embeddings split into two clusters (already-
    flagged collision nodes excluded, so they aren't double-counted). risk = separation × size.
  - *Hub pollution*: a coherent but abnormally high-degree node (≥ 3× median). risk =
    fan-out-blowup × community-diversity.
  - *Orphan*: a disconnected node; scored **LOW** on poisoning by construction (it can't conflate
    what it never returns).
- Repairs: for a collision, split the node by fact-cluster and re-assign each edge to the nearer
  side (all unsupervised); for an over-merge, split the community.
- Validation (`graphrx/probe.py`): two probes mirror GraphRAG's two retrieval modes: *local search*
  (seed at the entity, BFS k hops) and *global search* (community summary). Poisoning is the
  fraction of retrieved facts from a **different topic/community** than the query entity (legit
  same-topic neighbours are not poison). `poisoning_delta` measures before − after applying a
  repair.

## Prior art & how this differs

- Fanghua (Joshua) Yu, "Knowledge Graph Health Assessment" (a Neo4j-focused article series,
  not a peer-reviewed paper) assesses
  KG health with graph-native metrics, **inside Neo4j**. What GraphRx adds is threefold and
  nothing is claimed beyond that: (1) it is **not Neo4j-bound**; it operates on a plain
  adjacency dict, so it fits any GraphRAG store; (2) defects are scored through a **retrieval-
  poisoning lens** (impact on the *answer*: conflated evidence + traversal fan-out), not generic
  graph health; (3) each repair is **validated by a measured poisoning delta**, not asserted.
- General GraphRAG (Microsoft GraphRAG, entity-resolution / Leiden community detection) is the
  context this lints; GraphRx is a **linter over** such a graph, not a builder of one.

## Scope: v1 limits

- The graph, the embedder, and the LLM answerer are **SIMULATED with deterministic synthetic
  fixtures and a held-out labeled answer key** (the same discipline used across every tool in
  this repo): there is **no live GraphRAG index, no real embedding model, and no LLM** here.
  The poisoning metric is a proxy (cross-community fact fraction under a BFS/community probe),
  not measured on real model outputs.
- Detection thresholds (separation ≥ 0.5, degree quantiles) are **calibrated on the fixture** and
  printed by `eval.py` for transparency; real corpora need recalibration.
- Two defect classes (collision, over-merge) are exercised end-to-end by the red/green;
  hub-pollution and orphan detection are implemented and reported but only lightly tested.
- Ground-truth labels used by the probe exist only because the corpus is synthetic; on real data
  the delta harness needs a held-out labeled query set (a real limitation, stated plainly).

Not a GraphRAG builder, not an ER system, and not a substitute for human graph review. It's a
**pre-flight linter** that ranks where a graph will most poison answers and proves its fixes help.

## Where it fits

First tool in the `rag-reliability/` track; shares the deterministic-fixture + independent-
answer-key discipline of the sibling tools in this repo.
