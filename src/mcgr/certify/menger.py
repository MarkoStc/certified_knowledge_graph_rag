"""Deletion certificate via edge-disjoint path counting (Menger's theorem).

Definitions from AGENTS.md §7, implemented exactly:

    k(q) = (#edge-disjoint anchor->predicted-answer paths) - 1

With unit-capacity edges, max-flow = #edge-disjoint paths = min-cut
(Menger), so no adversary deleting <= k triples can destroy all support.
"""

import networkx as nx


def edge_disjoint_path_count(graph: nx.DiGraph, anchor: str, answer: str) -> int:
    """Number of pairwise edge-disjoint paths from ``anchor`` to ``answer``.

    Returns 0 when either node is absent or no path exists.
    """
    if anchor not in graph or answer not in graph or anchor == answer:
        return 0
    try:
        return len(list(nx.edge_disjoint_paths(graph, anchor, answer)))
    except nx.NetworkXNoPath:
        return 0


def deletion_certificate(graph: nx.DiGraph, anchor: str, answer: str) -> int:
    """Certified deletion budget ``k`` for ``answer`` given ``anchor``.

    k = 0 means a single fragile chain (one deleted triple can sever it);
    k = -1 means the answer has no path support at all.
    """
    return edge_disjoint_path_count(graph, anchor, answer) - 1
