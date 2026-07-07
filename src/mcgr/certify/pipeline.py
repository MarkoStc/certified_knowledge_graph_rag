"""Compute the deletion certificate over a dataset's queries (P4).

Per query: extract the anchor's k-hop subgraph from the dataset's KG, then
k(q) = max over gold answers of query_certificate(subgraph, anchors, answer).
Taking the max reflects "the answer the system would defend"; per-answer
detail is kept in the emitted record.

Datasets whose graph needs no Freebase are wired here: MetaQA (names as
nodes) and 2WikiMultiHopQA (Wikidata Q-ids as nodes). Each provides a KG,
a radius, and a stream of (qid, anchor-nodes, answer-nodes) in node space.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool

import networkx as nx

from mcgr.certify.menger import query_certificate
from mcgr.data import load_dataset
from mcgr.kg.graph_store import khop_subgraph

# 2Wiki gold chains are length 2; radius 2 captures same-length alternative
# paths (the conservative, gold-comparable horizon).
_TWOWIKI_RADIUS = 2


@dataclass(frozen=True)
class CertifiedQuery:
    qid: str
    k: int
    n_anchors: int
    n_answers_supported: int
    subgraph_edges: int


def _resolve_kg(name: str) -> tuple[nx.Graph, int]:
    if name.startswith("metaqa-"):
        from mcgr.kg.metaqa_kg import metaqa_graph

        hops = int(name.removeprefix("metaqa-").removesuffix("hop"))
        return metaqa_graph(), hops
    if name == "2wikimultihopqa":
        from mcgr.kg.twowiki_kg import twowiki_graph

        return twowiki_graph(), _TWOWIKI_RADIUS
    raise NotImplementedError(f"no Freebase-free KG provider for {name!r} yet (P2 builds the rest)")


def _certify_one(graph: nx.Graph, radius: int, qid, anchors, answers) -> CertifiedQuery:
    sub = khop_subgraph(graph, anchors, radius)
    ks = [query_certificate(sub, anchors, a) for a in answers if a in sub]
    supported = [k for k in ks if k >= 0]
    return CertifiedQuery(
        qid=qid,
        k=max(ks) if ks else -1,
        n_anchors=len(anchors),
        n_answers_supported=len(supported),
        subgraph_edges=sub.number_of_edges(),
    )


# worker globals: the full KB graph is built once per process, never pickled
_W_GRAPH: nx.Graph | None = None
_W_RADIUS: int = 0


def _worker_init(name: str) -> None:
    global _W_GRAPH, _W_RADIUS
    _W_GRAPH, _W_RADIUS = _resolve_kg(name)


def _worker_task(item: tuple[str, list[str], list[str]]) -> CertifiedQuery:
    assert _W_GRAPH is not None
    qid, anchors, answers = item
    return _certify_one(_W_GRAPH, _W_RADIUS, qid, anchors, answers)


def _iter_query_items(name: str, split: str, limit: int | None):
    """Yield (qid, anchor-nodes, answer-nodes) in the KG's node space."""
    if name == "2wikimultihopqa":
        # nodes are Wikidata Q-ids; take anchors/answers from the id-annotated
        # compositional/inference gold chains (the certifiable subset).
        from mcgr.kg.twowiki_kg import seed_qids

        _, queries = seed_qids(splits=(split,))
        for i, (qid, _split, anchor, answer) in enumerate(queries):
            if limit is not None and i >= limit:
                return
            yield (qid, [anchor], [answer])
        return
    for i, r in enumerate(load_dataset(name, split)):
        if limit is not None and i >= limit:
            return
        yield (r.qid, list(r.anchor_entities), list(r.answers))


def certify_dataset(
    name: str, split: str, limit: int | None = None, workers: int = 1
) -> Iterator[CertifiedQuery]:
    """Certificate per query. ``workers > 1`` fans out across processes,
    each building the KB graph once (the graph is never pickled)."""
    items = _iter_query_items(name, split, limit)
    if workers <= 1:
        graph, radius = _resolve_kg(name)
        for qid, anchors, answers in items:
            yield _certify_one(graph, radius, qid, anchors, answers)
        return
    with Pool(processes=workers, initializer=_worker_init, initargs=(name,)) as pool:
        yield from pool.imap_unordered(_worker_task, items, chunksize=16)
