"""The retrieval-poisoning delta harness -- the evaluation instrument that
VALIDATES a proposal by measuring answer poisoning before vs after applying it.

This is the ONLY place ground-truth labels are read. Poisoning is defined at the
community (topic) level: for a query about entity E, a retrieved fact is POISON if
its `real_community` differs from E's real community. Legitimately-related same-
topic neighbours are NOT poison -- only cross-topic conflation is.

Two probes mirror GraphRAG's two retrieval modes:
  * probe_local     -> local search: seed at the entity's node, BFS k hops,
                       collect facts (fan-out contamination).
  * probe_community -> global search: summarize the entity's whole community
                       (community-summary conflation).
"""
from .graph import bfs, community_members, deepcopy_graph


def _seed_node(graph, real_entity):
    """The node that a name lookup for `real_entity` resolves to (holds its facts)."""
    for n in sorted(graph["nodes"]):
        for f in graph["nodes"][n]["facts"]:
            if graph["facts"][f]["real_entity"] == real_entity:
                return n
    return None


def _query_community(graph, real_entity):
    for f in graph["facts"].values():
        if f["real_entity"] == real_entity:
            return f["real_community"]
    return None


def _poison_fraction(graph, fact_ids, query_comm):
    facts = [graph["facts"][f] for f in fact_ids]
    if not facts:
        return 0.0
    bad = sum(1 for f in facts if f["real_community"] != query_comm)
    return bad / len(facts)


def probe_local(graph, real_entity, depth=2):
    """Local-search poisoning: fraction of BFS-retrieved facts that are off-topic."""
    seed = _seed_node(graph, real_entity)
    if seed is None:
        return 1.0
    qc = _query_community(graph, real_entity)
    visited = bfs(graph, [seed], depth)
    fact_ids = [f for n in visited for f in graph["nodes"][n]["facts"]]
    return _poison_fraction(graph, fact_ids, qc)


def probe_community(graph, real_entity):
    """Global-search poisoning: fraction of the entity's community summary that is off-topic."""
    seed = _seed_node(graph, real_entity)
    if seed is None:
        return 1.0
    qc = _query_community(graph, real_entity)
    comm = graph["nodes"][seed]["community"]
    members = community_members(graph).get(comm, [])
    fact_ids = [f for n in members for f in graph["nodes"][n]["facts"]]
    return _poison_fraction(graph, fact_ids, qc)


def apply_proposal(graph, proposal):
    """Return a NEW graph with the proposal applied. Pure; input untouched."""
    g = deepcopy_graph(graph)
    op = proposal["op"]

    if op == "split_node":
        node = proposal["node"]
        parts = proposal["clusters"]
        comm = g["nodes"][node]["community"]
        a, b = node + "__a", node + "__b"
        g["nodes"][a] = {"name": a, "facts": list(parts[0]), "community": comm}
        g["nodes"][b] = {"name": b, "facts": list(parts[1]), "community": comm}
        # rewire edges to the assigned side; drop the old node
        assign = proposal["edge_assign"]
        new_edges = []
        for (u, v) in g["edges"]:
            if node in (u, v):
                other = v if u == node else u
                if other == node:
                    continue
                side = a if assign.get(other, 0) == 0 else b
                new_edges.append(tuple(sorted((side, other))))
            else:
                new_edges.append(tuple(sorted((u, v))))
        del g["nodes"][node]
        g["edges"] = sorted(set(new_edges))
        return g

    if op == "split_community":
        comm = proposal["community"]
        parts = proposal["parts"]
        for i, part in enumerate(parts):
            label = comm if i == 0 else f"{comm}__split{i}"
            for n in part:
                g["nodes"][n]["community"] = label
        return g

    # review_hub / attach_or_prune are advisory: no structural change.
    return g


def poisoning_delta(graph, proposal, real_entity, mode="local", depth=2):
    """before - after poisoning for `real_entity` under the given probe.

    Positive delta == the proposal REDUCES poisoning (a good repair).
    """
    probe = probe_local if mode == "local" else probe_community
    before = probe(graph, real_entity, depth) if mode == "local" else probe(graph, real_entity)
    fixed = apply_proposal(graph, proposal)
    after = probe(fixed, real_entity, depth) if mode == "local" else probe(fixed, real_entity)
    return {"before": before, "after": after, "delta": before - after}
