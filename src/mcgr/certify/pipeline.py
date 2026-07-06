"""Compute the deletion certificate over a dataset's queries (P4).

Per query: extract the anchor's k-hop subgraph from the dataset's KG, then
k(q) = max over gold answers of query_certificate(subgraph, anchors, answer).
Taking the max reflects "the answer the system would defend"; per-answer
detail is kept in the emitted record.

Only KG-native datasets whose graph needs no Freebase are wired here so far
(MetaQA). Others register their KG provider as P2 builds them.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import networkx as nx

from mcgr.certify.menger import query_certificate
from mcgr.data import load_dataset
from mcgr.kg.graph_store import khop_subgraph


@dataclass(frozen=True)
class CertifiedQuery:
    qid: str
    k: int
    n_anchors: int
    n_answers_supported: int
    subgraph_edges: int


def _metaqa_provider(hops: int):
    from mcgr.kg.metaqa_kg import metaqa_graph

    graph = metaqa_graph()
    return graph, hops


# dataset-family -> callable(name) -> (full_graph, radius)
def _resolve_kg(name: str) -> tuple[nx.Graph, int]:
    if name.startswith("metaqa-"):
        hops = int(name.removeprefix("metaqa-").removesuffix("hop"))
        return _metaqa_provider(hops)
    raise NotImplementedError(f"no Freebase-free KG provider for {name!r} yet (P2 builds the rest)")


def certify_dataset(name: str, split: str, limit: int | None = None) -> Iterator[CertifiedQuery]:
    graph, radius = _resolve_kg(name)
    for i, r in enumerate(load_dataset(name, split)):
        if limit is not None and i >= limit:
            return
        anchors = list(r.anchor_entities)
        sub = khop_subgraph(graph, anchors, radius)
        ks = [query_certificate(sub, anchors, a) for a in r.answers if a in sub]
        supported = [k for k in ks if k >= 0]
        yield CertifiedQuery(
            qid=r.qid,
            k=max(ks) if ks else -1,
            n_anchors=len(anchors),
            n_answers_supported=len(supported),
            subgraph_edges=sub.number_of_edges(),
        )
