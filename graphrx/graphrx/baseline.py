"""Naive generic graph-health baselines -- the INCUMBENT a competent engineer
would reach for, with NO poisoning lens and NO embeddings. Used by eval.py to
MEASURE (not assert) that GraphRx's poisoning-aware ranking recovers BOTH planted
poisoners in the top-2, while no single generic structural metric does.

Every metric here reads ONLY graph structure (degree, connectivity, community
membership). None reads fact text/embeddings or the ground-truth labels. That is
the whole point: generic graph health cannot see topic/entity conflation, so it
either misses a planted poisoner or flags a harmless one. These are fair,
straightforward implementations of the standard metrics -- not crippled strawmen:

  * degree centrality        -- the canonical node importance / fan-out metric;
  * degree anomaly           -- generic connectivity health (flag hubs AND
                                disconnected nodes as unhealthy);
  * community edge density    -- a modularity-flavored community-health metric.
"""
import numpy as np

from .graph import community_members, degrees


def degree_centrality_rank(graph):
    """Plain degree-centrality ranker: concern = node degree (importance/fan-out).
    The canonical generic graph-health metric. NODE-level only, so it is
    structurally blind to community-level defects. Returns [(node, score), ...]
    highest concern first; ties broken by node id for determinism."""
    deg = degrees(graph)
    ranked = sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(n, float(d)) for n, d in ranked]


def degree_anomaly_rank(graph):
    """Generic connectivity-health ranker: concern = |degree - median degree|, so
    it flags BOTH abnormally high-degree hubs AND disconnected/orphan nodes as
    'unhealthy' -- exactly what a generic health check does. NODE-level only.
    Returns [(node, score), ...] highest concern first; ties broken by node id."""
    deg = degrees(graph)
    med = float(np.median(np.array(list(deg.values()), dtype=float))) if deg else 0.0
    ranked = sorted(deg.items(), key=lambda kv: (-abs(kv[1] - med), kv[0]))
    return [(n, abs(float(d) - med)) for n, d in ranked]


def community_density_rank(graph):
    """Modularity-flavored community-health ranker: concern = 1 - intra-community
    edge density, so the least internally-connected community ranks worst.
    COMMUNITY-level only, so it is structurally blind to single-node collisions
    (a collision is one well-connected node -- there is no sparse internal cut to
    find). Returns [(community, score), ...] worst first; ties broken by name."""
    members = community_members(graph)
    node_comm = {n: graph["nodes"][n]["community"] for n in graph["nodes"]}
    intra = {c: 0 for c in members}
    for u, v in graph["edges"]:
        if node_comm[u] == node_comm[v]:
            intra[node_comm[u]] += 1
    out = []
    for c, mem in members.items():
        k = len(mem)
        possible = k * (k - 1) / 2.0
        density = intra[c] / possible if possible > 0 else 1.0
        out.append((c, 1.0 - density))
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out
