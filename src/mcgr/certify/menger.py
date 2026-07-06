"""Deletion certificate via edge-disjoint path counting (Menger's theorem).

Definitions from AGENTS.md §7, implemented exactly:

    k(q) = (#edge-disjoint anchor->predicted-answer paths) - 1

With unit-capacity edges, max-flow = #edge-disjoint paths = min-cut
(Menger), so no adversary deleting <= k triples can destroy all support.
"""

import networkx as nx
from networkx.algorithms.connectivity import local_edge_connectivity


def edge_disjoint_path_count(graph: nx.DiGraph, anchor: str, answer: str) -> int:
    """Number of pairwise edge-disjoint paths from ``anchor`` to ``answer``.

    By Menger's theorem this equals the local edge connectivity (the value of
    a unit-capacity max-flow), which we compute directly rather than
    enumerating the paths — same integer, ~2-3x faster on dense subgraphs.
    Returns 0 when either node is absent or no path exists.
    """
    if anchor not in graph or answer not in graph or anchor == answer:
        return 0
    return local_edge_connectivity(graph, anchor, answer)


def deletion_certificate(graph: nx.DiGraph, anchor: str, answer: str) -> int:
    """Certified deletion budget ``k`` for ``answer`` given ``anchor``.

    k = 0 means a single fragile chain (one deleted triple can sever it);
    k = -1 means the answer has no path support at all.
    """
    return edge_disjoint_path_count(graph, anchor, answer) - 1


_SUPER_SOURCE = "__mcgr_super_source__"


def anchor_set_path_count(graph: nx.Graph, anchors: list[str], answer: str) -> int:
    """#edge-disjoint paths from *any* anchor to ``answer``.

    Uses a super-source joined to every anchor by uncuttable (infinite-
    capacity) edges, so the min cut falls only on real fact edges. This is
    exactly the min number of facts an adversary must delete to disconnect
    ``answer`` from the whole anchor set. Reduces to the single-pair count
    when there is one anchor.
    """
    present = [a for a in dict.fromkeys(anchors) if a in graph and a != answer]
    if not present or answer not in graph:
        return 0
    if len(present) == 1:
        return edge_disjoint_path_count(graph, present[0], answer)

    flow_graph = nx.Graph()
    for u, v in graph.edges():
        flow_graph.add_edge(u, v, capacity=1)
    for a in present:
        flow_graph.add_edge(_SUPER_SOURCE, a, capacity=float("inf"))
    flow_value, _ = nx.maximum_flow(flow_graph, _SUPER_SOURCE, answer)
    return int(flow_value)


def query_certificate(graph: nx.Graph, anchors: list[str], answer: str) -> int:
    """Deletion certificate ``k`` for a query with a set of anchors.

    k = (#edge-disjoint anchor-set -> answer paths) - 1; k = -1 when the
    answer is unsupported (unreachable from every anchor).
    """
    return anchor_set_path_count(graph, anchors, answer) - 1
