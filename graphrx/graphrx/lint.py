"""The linter: unsupervised structural + embedding checks that score each defect
by GraphRAG answer-poisoning risk, and emit ranked merge/split proposals.

NONE of the detectors read the ground-truth `real_entity`/`real_community` labels.
They use only graph structure (degree, connectivity, community membership) and the
vendored hashing embedder. That is what makes the red/green honest: the mechanism
genuinely discovers the fault.

Poisoning-risk model (why each defect corrupts a GraphRAG answer):
  * COLLISION      -> conflated evidence: one node holds two entities' facts, so a
                      query about one retrieves the other's; the node also BRIDGES
                      communities, so k-hop traversal fans out into unrelated
                      evidence.  risk = separation * reach(degree).
  * OVER-MERGE     -> a "global"/community summary blends two topics.
                      risk = separation * size.
  * HUB POLLUTION  -> a coherent but very high-degree node balloons traversal
                      fan-out.  risk = fanout_blowup * community_diversity.
  * ORPHAN         -> unreachable in traversal: a RECALL defect, not poisoning.
                      risk is fixed LOW (it cannot conflate what it never returns).
"""
import numpy as np

from .embed import cosine
from .graph import (adjacency, community_members, components, degrees,
                    node_embedding)

# Calibrated on the clean fixture (see eval.py, which prints the measured
# separations). Coherent nodes/communities land well below 0.5; the planted
# two-topic defects land well above it.
SEP_THRESH = 0.5
MIN_FACTS_FOR_SPLIT = 4
MIN_COMMUNITY_FOR_SPLIT = 4
ORPHAN_RISK = 0.05


def two_means(vecs, iters=25):
    """Deterministic 2-means on L2-normalized rows. Returns (labels, centroids).

    Init = the single farthest-apart pair (min cosine). No randomness.
    """
    n = len(vecs)
    if n < 2:
        return np.zeros(n, dtype=int), np.repeat(vecs, 2, axis=0) if n else None
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, np.inf)          # ignore self-pairs when finding the min
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    cen = np.stack([vecs[i], vecs[j]]).astype(np.float64)
    labels = np.full(n, -1, dtype=int)
    for _ in range(iters):
        new = np.argmax(vecs @ cen.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            m = vecs[labels == k]
            if len(m):
                c = m.mean(axis=0)
                nrm = np.linalg.norm(c)
                if nrm > 0:
                    cen[k] = c / nrm
    return labels, cen


def _separation(vecs):
    """1 - cosine(centroid_A, centroid_B) of the best 2-way split. In [0, 2]."""
    labels, cen = two_means(vecs)
    if cen is None or len(np.unique(labels)) < 2:
        return 0.0, labels, cen
    return 1.0 - cosine(cen[0], cen[1]), labels, cen


def _node_incoherence(graph, node_id):
    from .embed import embed_many
    texts = [graph["facts"][f]["text"] for f in graph["nodes"][node_id]["facts"]]
    vecs = embed_many(texts)
    sep, labels, _ = _separation(vecs)
    # require both sides material, else the "split" is spurious
    if len(vecs) >= 2 and min((labels == 0).sum(), (labels == 1).sum()) < 1:
        sep = 0.0
    return sep, labels


def _community_incoherence(graph, comm_id, members):
    vecs = np.stack([node_embedding(graph, n) for n in members])
    return _separation(vecs)


def _collision_proposal(graph, node_id, labels):
    """Split a node's facts by cluster; assign each edge to the sub-node whose
    fact-centroid its neighbour is nearest -- all unsupervised."""
    fids = graph["nodes"][node_id]["facts"]
    from .embed import embed_many
    vecs = embed_many([graph["facts"][f]["text"] for f in fids])
    cen = []
    parts = [[], []]
    for k in (0, 1):
        idx = labels == k
        parts[k] = [fids[i] for i in range(len(fids)) if idx[i]]
        c = vecs[idx].mean(axis=0)
        n = np.linalg.norm(c)
        cen.append(c / n if n > 0 else c)
    adj = adjacency(graph)
    edge_assign = {}
    for nb in sorted(adj[node_id]):
        nbv = node_embedding(graph, nb)
        edge_assign[nb] = int(np.argmax([cosine(nbv, cen[0]), cosine(nbv, cen[1])]))
    return {"op": "split_node", "node": node_id,
            "clusters": parts, "edge_assign": edge_assign}


def _overmerge_proposal(graph, comm_id, members, labels):
    parts = [[members[i] for i in range(len(members)) if labels[i] == k] for k in (0, 1)]
    return {"op": "split_community", "community": comm_id, "parts": parts}


def report(graph):
    """Return a ranked list of scored defects with repair proposals.

    Each defect: {kind, target, score, evidence{...}, proposal}.
    Ranked by descending poisoning-risk score.
    """
    deg = degrees(graph)
    maxdeg = max(deg.values()) if deg else 1
    degvals = np.array(sorted(deg.values()))
    median_deg = float(np.median(degvals)) if len(degvals) else 0.0
    members = community_members(graph)
    defects = []
    flagged_nodes = set()

    # --- COLLISIONS (incoherent + bridging high-degree node) ---
    deg_q75 = float(np.quantile(degvals, 0.75)) if len(degvals) else 0.0
    for n in sorted(graph["nodes"]):
        nf = graph["nodes"][n]["facts"]
        if len(nf) < MIN_FACTS_FOR_SPLIT:
            continue
        sep, labels = _node_incoherence(graph, n)
        if sep >= SEP_THRESH and deg[n] >= max(deg_q75, 1):
            reach = deg[n] / maxdeg
            score = float(sep * reach)
            defects.append({
                "kind": "er_collision", "target": n, "score": score,
                "evidence": {"fact_separation": round(sep, 4), "degree": deg[n],
                             "reach": round(reach, 4)},
                "proposal": _collision_proposal(graph, n, labels),
            })
            flagged_nodes.add(n)

    # --- OVER-MERGED COMMUNITIES ---
    for c in sorted(members):
        # exclude already-flagged collision nodes: their intrinsic incoherence is
        # a node-level defect, not evidence that the community is over-merged.
        mem = [n for n in members[c] if n not in flagged_nodes]
        if len(mem) < MIN_COMMUNITY_FOR_SPLIT:
            continue
        sep, labels, _ = _community_incoherence(graph, c, mem)
        if sep >= SEP_THRESH:
            size_norm = len(mem) / max(len(graph["nodes"]), 1)
            score = float(sep * (0.5 + size_norm))  # size amplifies conflation
            defects.append({
                "kind": "over_merged_community", "target": c, "score": score,
                "evidence": {"member_separation": round(sep, 4), "size": len(mem)},
                "proposal": _overmerge_proposal(graph, c, mem, labels),
            })

    # --- HUB POLLUTION (coherent but abnormally high degree) ---
    adj = adjacency(graph)
    for n in sorted(graph["nodes"]):
        if n in flagged_nodes:
            continue  # a collision already subsumes/outranks a bare hub finding
        d = deg[n]
        if median_deg > 0 and d >= 3 * median_deg and d >= 4:
            nb_comms = {graph["nodes"][w]["community"] for w in adj[n]}
            blowup = d / max(median_deg, 1.0)
            div = len(nb_comms) / max(len(members), 1)
            score = float(min(1.0, blowup / 6.0) * div)
            defects.append({
                "kind": "hub_pollution", "target": n, "score": score,
                "evidence": {"degree": d, "blowup_x": round(blowup, 2),
                             "communities_bridged": len(nb_comms)},
                "proposal": {"op": "review_hub", "node": n},
            })

    # --- ORPHANS (recall defect; low poisoning by construction) ---
    singletons = {list(c)[0] for c in components(graph) if len(c) == 1}
    for n in sorted(singletons):
        defects.append({
            "kind": "orphan", "target": n, "score": ORPHAN_RISK,
            "evidence": {"degree": deg[n], "note": "unreachable: recall risk, not poisoning"},
            "proposal": {"op": "attach_or_prune", "node": n},
        })

    defects.sort(key=lambda d: (-d["score"], d["kind"], str(d["target"])))
    return defects


def high_risk(defects, thresh=0.25):
    return [d for d in defects if d["score"] >= thresh]
