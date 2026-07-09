"""Retrieval policies for the P8 ablation (AGENTS.md P8).

Two policies over the same KG subgraph, returning directed evidence triples:

- ``single_best_chain`` — the standard KG-RAG baseline: one shortest
  supporting chain. Its retrieved certificate is always 0 (one path).
- ``certificate_maximizing`` — an agentic policy that iteratively seeks
  additional *edge-disjoint* anchor->answer paths (each new path avoids all
  edges already used), up to a budget. Its retrieved certificate is
  (#paths found) - 1, so it actively raises certifiability.

Both convert paths to directed triples via a KB pair index (see
retrieval.evidence).
"""

from itertools import pairwise

import networkx as nx

from mcgr.retrieval.evidence import Triple, _edge_to_triples


def _path_triples(path, pair_index) -> list[Triple]:
    out: list[Triple] = []
    for u, v in pairwise(path):
        out.extend(_edge_to_triples(u, v, pair_index))
    return out


def single_best_chain(
    subgraph: nx.Graph, anchor: str, answer: str, pair_index
) -> tuple[list[Triple], int]:
    """Retrieve one shortest anchor->answer chain. Returns (triples, n_paths)."""
    if anchor not in subgraph or answer not in subgraph or anchor == answer:
        return [], 0
    try:
        path = nx.shortest_path(subgraph, anchor, answer)
    except nx.NetworkXNoPath:
        return [], 0
    return _dedup(_path_triples(path, pair_index)), 1


def certificate_maximizing(
    subgraph: nx.Graph, anchor: str, answer: str, pair_index, max_paths: int = 8
) -> tuple[list[Triple], int]:
    """Retrieve up to ``max_paths`` edge-disjoint anchor->answer paths.

    Each additional path is independent of those already found (edge-disjoint),
    so the retrieved evidence supports a certificate k = n_paths - 1.
    """
    if anchor not in subgraph or answer not in subgraph or anchor == answer:
        return [], 0
    try:
        paths = list(nx.edge_disjoint_paths(subgraph, anchor, answer))
    except nx.NetworkXNoPath:
        return [], 0
    paths = paths[:max_paths]
    triples: list[Triple] = []
    for p in paths:
        triples.extend(_path_triples(p, pair_index))
    return _dedup(triples), len(paths)


def _dedup(triples: list[Triple]) -> list[Triple]:
    seen: set[Triple] = set()
    out: list[Triple] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
