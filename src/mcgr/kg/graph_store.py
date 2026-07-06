"""Per-query subgraph extraction over a knowledge graph (AGENTS.md P2).

The graph is **undirected**: for the deletion threat model an adversary
removes *facts*, and a fact is one undirected edge. Edge-disjoint paths
between anchor and answer then count independent fact-chains (Menger's
classic undirected form), which is what the P4 certificate needs.

Parallel facts between the same entity pair are collapsed to one edge —
NetworkX's flow routines operate on simple graphs, and collapsing can only
*lower* the path count, so the resulting certificate is conservative
(under-certifies rather than over-certifies), matching the brief's stance.
"""

import networkx as nx

# edge attribute holding the relation label(s) for a collapsed edge
REL_ATTR = "relations"


def build_graph(triples: list[tuple[str, str, str]]) -> nx.Graph:
    """Undirected fact graph from (subject, relation, object) triples."""
    g = nx.Graph()
    for s, r, o in triples:
        if g.has_edge(s, o):
            g[s][o][REL_ATTR].add(r)
        else:
            g.add_edge(s, o, **{REL_ATTR: {r}})
    return g


def khop_subgraph(graph: nx.Graph, sources: list[str], radius: int) -> nx.Graph:
    """Node-induced subgraph within ``radius`` hops of any source.

    Contains every anchor->answer path of length <= radius, which bounds
    the certificate to the query's reasoning horizon (P2).
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")
    reached: set[str] = set()
    frontier = {s for s in sources if s in graph}
    reached |= frontier
    for _ in range(radius):
        nxt: set[str] = set()
        for node in frontier:
            nxt.update(graph.neighbors(node))
        nxt -= reached
        reached |= nxt
        frontier = nxt
        if not frontier:
            break
    return graph.subgraph(reached).copy()
