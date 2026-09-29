"""Graph representation + traversal helpers (pure dict/adjacency, no external graph lib).

A GraphRAG-style entity graph, deterministic in-memory:

    graph = {
        "nodes": {node_id: {"name": str, "facts": [fact_id, ...], "community": comm_id}},
        "edges": [(u, v), ...],                       # undirected relationships
        "facts": {fact_id: {"text": str,
                            "real_entity": str,        # GROUND TRUTH (probe/eval only)
                            "real_community": str}},   # GROUND TRUTH (probe/eval only)
    }

`real_entity` / `real_community` are the hidden answer key: the linter's detectors
NEVER read them (they are unsupervised/structural). Only the poisoning probe,
the evaluation instrument, reads them, exactly like a labeled test set.
"""
import copy
from collections import defaultdict, deque

import numpy as np

from .embed import embed_many, normalize


def deepcopy_graph(graph):
    return copy.deepcopy(graph)


def adjacency(graph):
    adj = defaultdict(set)
    for n in graph["nodes"]:
        adj[n]  # ensure present
    for u, v in graph["edges"]:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def bfs(graph, seeds, depth):
    """Set of nodes within `depth` hops of any seed (seeds included)."""
    adj = adjacency(graph)
    seen = set(seeds)
    frontier = deque(seeds)
    dist = {s: 0 for s in seeds}
    while frontier:
        u = frontier.popleft()
        if dist[u] >= depth:
            continue
        for w in sorted(adj[u]):
            if w not in seen:
                seen.add(w)
                dist[w] = dist[u] + 1
                frontier.append(w)
    return seen


def degrees(graph):
    adj = adjacency(graph)
    return {n: len(adj[n]) for n in sorted(graph["nodes"])}


def components(graph):
    adj = adjacency(graph)
    seen = set()
    comps = []
    for start in sorted(graph["nodes"]):
        if start in seen:
            continue
        stack = [start]
        comp = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(sorted(adj[u] - seen))
        comps.append(comp)
    return comps


def node_facts(graph, node_id):
    return [graph["facts"][f] for f in graph["nodes"][node_id]["facts"]]


def node_embedding(graph, node_id):
    """Mean of a node's fact embeddings, L2-normalized. Empty node -> zeros."""
    texts = [f["text"] for f in node_facts(graph, node_id)]
    mat = embed_many(texts)
    if len(mat) == 0:
        from .embed import DIM
        return np.zeros(DIM)
    return normalize(mat.mean(axis=0))


def community_members(graph):
    members = defaultdict(list)
    for n in sorted(graph["nodes"]):
        members[graph["nodes"][n]["community"]].append(n)
    return dict(members)
