# GraphRx

GraphRAG structural linter with retrieval-poisoning scoring.

- Input: a built GraphRAG graph (entities, relationships, communities) as a plain Python adjacency dict
- Output: ranked report of structural defects: ER (entity-resolution) collisions, over-/under-merged communities, hub pollution, orphans
- Each defect scored by how much it poisons answers
- Each proposed merge/split repair validated by a before/after poisoning-delta harness

## Quickstart

```
python graphrx/eval.py      # RED/GREEN self-test (exit 0)
python graphrx/run_demo.py  # one worked graph
```

## Results (eval.py, exit 0)

Clean synthetic graph: **zero high-risk defects** (max node fact-separation 0.4286, max community member-separation 0.2965; threshold 0.5).

Graph with a **planted ER collision** (a bank + a biotech merged into one node) and a **planted over-merged community** (aviation + sports):

- Both flagged by unsupervised signals (collision node separation **1.0000**, over-merged community separation **1.0000**) and ranked at the top. Orphan scored 0.05.
- Collision split: local-search poisoning **0.5714 to 0.0000** (delta +0.5714), 100%-pure fact separation
- Community split: global-search poisoning **0.5000 to 0.0000** (delta +0.5000)
- Negative control: splitting a clean, coherent node gives delta **−0.0357** (no gain)

### A/B vs generic graph-health metrics

Same flawed graph. Three standard graph-health baselines (no poisoning lens, no embeddings), scored by how many of the two planted poisoners (collision node `apex`, over-merged community `aviation_sports`) land in their top-2:

| method | top-2 | poisoners recovered | false positives |
|---|---|---|---|
| **GraphRx (poisoning lens)** | `apex`, `aviation_sports` | **2 / 2** | **0** |
| degree centrality | `apex`, `aerodyne` | 1 / 2 | 1: *node metric, blind to the over-merged community* |
| degree anomaly | `apex`, `lonelycorp` | 1 / 2 | 1: *flags the harmless orphan* |
| community edge-density | `aviation_sports`, `finance` | 1 / 2 | 1: *community metric, blind to the single-node collision* |

- GraphRx: 2/2 poisoners recovered, 0 false positives; no generic metric recovers both.
- Degree centrality ranks `apex` #1 by a 1-edge margin (degree 4 vs 3) over a legitimate hub.
- GraphRx risk, apex vs legit hub: **1.0000 vs 0.0000** (hub never flagged).
- Baselines: `graphrx/baseline.py`. Scored comparison prints in `eval.py`.
- Detectors read only structure + a vendored hashing embedder. Ground-truth labels are read only by the probe.

## How it works

Layout: modules live in a nested `graphrx/graphrx/` package. File references below are relative to that package, e.g. `graphrx/graphrx/embed.py` from the repo root.

- Embedding backend (SIMULATED): vendored pure-Python BLAKE2b hashing embedder (`graphrx/embed.py`, dim 256, L2-normalized, cosine). No model download, no network. A real deployment swaps in a sentence embedder behind the same interface.
- Detectors (`graphrx/lint.py`, unsupervised):
  - *ER collision*: node whose facts split into two well-separated embedding clusters (2-means separation ≥ 0.5) and whose degree bridges communities. risk = separation × reach(degree).
  - *Over-merged community*: member embeddings split into two clusters (already-flagged collision nodes excluded). risk = separation × size.
  - *Hub pollution*: coherent node with abnormally high degree (≥ 3× median). risk = fan-out-blowup × community-diversity.
  - *Orphan*: disconnected node; scored LOW on poisoning by construction.
- Repairs: collision, split the node by fact-cluster and re-assign each edge to the nearer side; over-merge, split the community. All unsupervised.
- Validation (`graphrx/probe.py`): *local search* (seed at the entity, BFS k hops) and *global search* (community summary). Poisoning = fraction of retrieved facts from a different topic/community than the query entity. `poisoning_delta` = before − after a repair.

## Prior art

- Fanghua (Joshua) Yu, "Knowledge Graph Health Assessment" (Neo4j-focused article series, not peer-reviewed): graph-native KG health metrics inside Neo4j. GraphRx differs in three ways:
  1. Not Neo4j-bound: plain adjacency dict
  2. Defects scored through a retrieval-poisoning lens (conflated evidence + traversal fan-out)
  3. Repairs validated by a measured poisoning delta
- Microsoft GraphRAG, entity resolution, Leiden community detection: the context GraphRx lints. It is a linter over such a graph, not a builder.

## Limits (v1)

- Graph, embedder and LLM answerer are SIMULATED: deterministic synthetic fixtures with a held-out labeled answer key. No live GraphRAG index, real embedding model or LLM.
- Poisoning metric is a proxy (cross-community fact fraction under a BFS/community probe), not measured on real model outputs.
- Thresholds (separation ≥ 0.5, degree quantiles) calibrated on the fixture and printed by `eval.py`; real corpora need recalibration.
- Collision and over-merge exercised end-to-end by the red/green. Hub-pollution and orphan detection implemented and reported, lightly tested.
- Real data needs a held-out labeled query set for the delta harness.
- Not a GraphRAG builder, an ER system, or a substitute for human graph review.

## Where it fits

First tool in the `rag-reliability/` track. Shares the deterministic-fixture + independent-answer-key discipline of its siblings.
