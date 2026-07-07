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


def prune_hubs(graph: nx.Graph, max_degree: int) -> tuple[nx.Graph, dict[str, set[str]]]:
    """Remove nodes with degree > ``max_degree``, returning (pruned graph,
    ``{hub: neighbors}`` adjacency for the removed nodes).

    In a Wikidata-derived KG the highest-degree nodes are type/attribute
    values (``human``, ``male``, ``film``, a country) that hundreds or
    thousands of entities point to. A path routed *through* such a hub —
    ``entityA -> human -> entityB`` — is shared-type coincidence, not
    independent supporting evidence, so counting it would inflate the
    certificate. Dropping hubs enforces *evidential* (not merely structural)
    independence (AGENTS.md P5 / §12). The cut is self-calibrating: type
    hubs accumulate degree with dataset size while specific entities stay
    small.

    A hub is legitimate as a query *endpoint* (many questions answer with a
    country or occupation), just not as a transit node. The returned
    adjacency lets ``reconnect_endpoints`` add a hub back when it is this
    query's anchor or answer — see that function.
    """
    removed = {n for n, d in graph.degree() if d > max_degree}
    hub_adj = {h: set(graph.neighbors(h)) for h in removed}
    if not removed:
        return graph, hub_adj
    kept = graph.copy()
    kept.remove_nodes_from(removed)
    return kept, hub_adj


def reconnect_endpoints(
    subgraph: nx.Graph, hub_adj: dict[str, set[str]], endpoints: list[str]
) -> None:
    """Re-add hub ``endpoints`` (this query's anchor/answer) to ``subgraph``,
    with edges only to nodes already present. In place.

    This restores a pruned hub as a legitimate terminal without reopening
    transit through it: the hub connects to the local non-hub structure the
    subgraph already reached, but no path can route through *other* hubs.
    """
    nodes = set(subgraph.nodes())
    for e in endpoints:
        if e in hub_adj and e not in subgraph:
            for v in hub_adj[e] & nodes:
                subgraph.add_edge(e, v, **{REL_ATTR: {"__endpoint__"}})


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
