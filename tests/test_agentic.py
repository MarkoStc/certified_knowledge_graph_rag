"""Retrieval policies for P8 (hermetic)."""

from mcgr.kg.graph_store import build_graph
from mcgr.retrieval.agentic import certificate_maximizing, single_best_chain

# two edge-disjoint paths a->b->c and a->d->c
G = build_graph([("a", "r1", "b"), ("b", "r2", "c"), ("a", "r3", "d"), ("d", "r4", "c")])
PAIR = {
    frozenset(("a", "b")): [("a", "r1", "b")],
    frozenset(("b", "c")): [("b", "r2", "c")],
    frozenset(("a", "d")): [("a", "r3", "d")],
    frozenset(("d", "c")): [("d", "r4", "c")],
}


def test_single_best_chain_returns_one_path() -> None:
    triples, n = single_best_chain(G, "a", "c", PAIR)
    assert n == 1  # retrieved certificate k = 0
    assert len(triples) == 2  # one 2-hop chain


def test_certificate_maximizing_finds_both_paths() -> None:
    triples, n = certificate_maximizing(G, "a", "c", PAIR)
    assert n == 2  # retrieved certificate k = 1
    assert len(triples) == 4  # both independent chains


def test_unreachable_returns_empty() -> None:
    g = build_graph([("a", "r", "b"), ("c", "r", "d")])
    assert single_best_chain(g, "a", "d", {}) == ([], 0)
    assert certificate_maximizing(g, "a", "d", {}) == ([], 0)
