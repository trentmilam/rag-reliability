"""Deterministic synthetic GraphRAG fixtures (SIMULATED entity/community graph).

A clean graph has 4 topical communities, each with 3 single-topic entities. The
flawed graph is the clean graph plus two PLANTED, labeled defects:

  1. an ER (entity-resolution) COLLISION: two real-world entities from different
     communities (a bank + a biotech that share the surface alias "apex") were
     merged into one node -> its facts carry two topics and it BRIDGES two
     communities (traversal fan-out into unrelated evidence);
  2. an OVER-MERGED community: aviation + sports entities relabeled into one
     community (a "global" community summary would conflate two topics);
  plus one ORPHAN node (a recall problem, deliberately LOW poisoning risk).

Facts are built from the entity name + its topical VOCABULARY only -- the literal
ground-truth community label (the topic key) is deliberately NOT written into any
fact text, so the hashing embedder recovers topic/entity structure from context
alone, never from a leaked label. Ground-truth labels live on each fact for the
probe only.
"""
import numpy as np

# topic -> distinguishing vocabulary
TOPICS = {
    "finance": ["revenue", "fiscal", "quarterly", "earnings", "dividend", "ledger"],
    "biology": ["protein", "enzyme", "cellular", "genome", "assay", "mitochondria"],
    "aviation": ["fuselage", "altitude", "turbine", "runway", "avionics", "aileron"],
    "sports": ["tournament", "athlete", "stadium", "scoring", "league", "playoff"],
}
# entity name -> topic (3 entities per topic)
ENTITIES = {
    "alphabank": "finance", "betacap": "finance", "gammafund": "finance",
    "helixgen": "biology", "cytolab": "biology", "genoworks": "biology",
    "aerodyne": "aviation", "skyjet": "aviation", "altiplane": "aviation",
    "proleague": "sports", "victorsc": "sports", "apexsport": "sports",
}
FACTS_PER_ENTITY = 4


def _entity_facts(rng, name, topic, fid_start):
    """Build coherent single-topic facts for one entity."""
    vocab = TOPICS[topic]
    facts = {}
    fids = []
    for i in range(FACTS_PER_ENTITY):
        # Three overlapping topic-vocab words (NO literal topic/community label):
        # separation must come from the entity name + its topical context, not a
        # leaked ground-truth token. The rolling window keeps an entity's facts
        # mutually coherent while cross-topic facts stay well separated.
        w = [vocab[(i + j) % len(vocab)] for j in range(3)]
        text = f"{name} {w[0]} {w[1]} {w[2]}"
        fid = f"f{fid_start + i}"
        facts[fid] = {"text": text, "real_entity": name, "real_community": topic}
        fids.append(fid)
    # seeded, deterministic fact-order shuffle: the detectors must be order-
    # invariant, and this makes the fixed seed genuinely drive the output.
    order = rng.permutation(len(fids))
    fids = [fids[i] for i in order]
    return facts, fids


def _triangle_edges(nodes):
    """Ring/triangle edges so intra-community degree is a small constant."""
    edges = []
    k = len(nodes)
    for i in range(k):
        edges.append((nodes[i], nodes[(i + 1) % k]))
    return edges


def build_clean_graph(seed):
    rng = np.random.default_rng(seed)
    nodes, facts, edges = {}, {}, []
    fid = 0
    by_topic = {}
    for name, topic in ENTITIES.items():
        efacts, fids = _entity_facts(rng, name, topic, fid)
        fid += FACTS_PER_ENTITY
        facts.update(efacts)
        nodes[name] = {"name": name, "facts": fids, "community": topic}
        by_topic.setdefault(topic, []).append(name)
    # intra-community triangles
    for topic, members in by_topic.items():
        edges.extend(_triangle_edges(members))
    # one thin bridge to keep the whole graph connected (finance<->biology,
    # biology<->aviation, aviation<->sports) -- these are legitimate, not hubs.
    edges.append(("gammafund", "helixgen"))
    edges.append(("genoworks", "aerodyne"))
    edges.append(("altiplane", "proleague"))
    return {"nodes": nodes, "edges": edges, "facts": facts}


def build_flawed_graph(seed):
    """Clean graph + planted collision + over-merged community + orphan."""
    g = build_clean_graph(seed)
    nodes, facts, edges = g["nodes"], g["facts"], g["edges"]

    # --- (1) ER COLLISION: merge alphabank (finance) + helixgen (biology) ---
    # into a single node "apex" that inherits BOTH entities' facts and edges.
    merged_facts = nodes["alphabank"]["facts"] + nodes["helixgen"]["facts"]
    adj_before = set()
    for (u, v) in edges:
        if u in ("alphabank", "helixgen"):
            adj_before.add(v)
        if v in ("alphabank", "helixgen"):
            adj_before.add(u)
    adj_before.discard("alphabank")
    adj_before.discard("helixgen")
    # drop the two originals, add the collision node
    del nodes["alphabank"]
    del nodes["helixgen"]
    nodes["apex"] = {"name": "apex", "facts": merged_facts, "community": "finance"}
    # rebuild edge list, redirecting old endpoints to "apex"
    new_edges = []
    for (u, v) in edges:
        u2 = "apex" if u in ("alphabank", "helixgen") else u
        v2 = "apex" if v in ("alphabank", "helixgen") else v
        if u2 == v2:
            continue
        e = tuple(sorted((u2, v2)))
        new_edges.append(e)
    edges = sorted(set(new_edges))

    # --- (2) OVER-MERGED COMMUNITY: relabel aviation + sports as one ---
    for n in nodes:
        if nodes[n]["community"] in ("aviation", "sports"):
            nodes[n]["community"] = "aviation_sports"

    # --- (3) ORPHAN: a disconnected node (recall risk, low poisoning) ---
    ofacts, ofids = _entity_facts(np.random.default_rng(seed + 1),
                                  "lonelycorp", "finance", 9000)
    facts.update(ofacts)
    nodes["lonelycorp"] = {"name": "lonelycorp", "facts": ofids, "community": "finance"}
    # (no edges added for lonelycorp)

    return {"nodes": nodes, "edges": edges, "facts": facts}


# Ground-truth answer key for the eval (NOT used by detectors).
PLANTED = {
    "collision_node": "apex",
    "collision_entities": ("alphabank", "helixgen"),
    "overmerged_community": "aviation_sports",
    "orphan_node": "lonelycorp",
}
