"""MetaQA knowledge graph (AGENTS.md P2, self-contained — no Freebase).

The movie KB (kb.txt) is a complete graph, so each query's reasoning space
is the k-hop neighborhood of its anchor entity. Built once, cached, and
reused across queries.
"""

from functools import cache

import networkx as nx

from mcgr.data.metaqa import load_kb
from mcgr.kg.graph_store import build_graph


@cache
def metaqa_graph() -> nx.Graph:
    """The full MetaQA movie KB as one undirected fact graph (cached)."""
    return build_graph(load_kb())
