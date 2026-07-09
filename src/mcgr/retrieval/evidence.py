"""Evidence retrieval for the P7 reasoning experiment (AGENTS.md P8 seed).

The certificate graph is undirected (deletion threat model), but the
reasoner needs readable *directed* triples. For MetaQA we recover direction
from the KB: index the original ``(subject, relation, object)`` triples by
the unordered pair, then, for each query, return the directed triples lying
on the edge-disjoint anchor->answer paths (the genuine supporting evidence,
whose count scales with the certificate ``k``), capped for context length.
"""

from functools import cache
from itertools import pairwise

import networkx as nx

Triple = tuple[str, str, str]


@cache
def metaqa_pair_index() -> dict[frozenset, list[Triple]]:
    """Map each unordered entity pair to its directed KB triple(s)."""
    from mcgr.data.metaqa import load_kb

    index: dict[frozenset, list[Triple]] = {}
    for s, r, o in load_kb():
        index.setdefault(frozenset((s, o)), []).append((s, r, o))
    return index


@cache
def metaqa_relation_objects() -> dict[str, tuple[str, ...]]:
    """Map each relation to the tuple of its object entities (for picking a
    type-consistent 'sibling' wrong answer under the same relation)."""
    from mcgr.data.metaqa import load_kb

    objs: dict[str, list[str]] = {}
    for _s, r, o in load_kb():
        objs.setdefault(r, []).append(o)
    return {r: tuple(dict.fromkeys(v)) for r, v in objs.items()}


def _edge_to_triples(u: str, v: str, pair_index: dict[frozenset, list[Triple]]) -> list[Triple]:
    return pair_index.get(frozenset((u, v)), [(u, "related_to", v)])


def path_evidence(
    subgraph: nx.Graph,
    anchor: str,
    answer: str,
    pair_index: dict[frozenset, list[Triple]],
    max_paths: int = 8,
) -> list[Triple]:
    """Directed triples along up to ``max_paths`` edge-disjoint anchor->answer
    paths in ``subgraph``. Empty if unreachable."""
    if anchor not in subgraph or answer not in subgraph or anchor == answer:
        return []
    try:
        paths = list(nx.edge_disjoint_paths(subgraph, anchor, answer))
    except nx.NetworkXNoPath:
        return []
    triples: list[Triple] = []
    seen: set[Triple] = set()
    for path in paths[:max_paths]:
        for u, v in pairwise(path):
            for t in _edge_to_triples(u, v, pair_index):
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
    return triples
