"""Path-evidence retrieval (hermetic)."""

from mcgr.kg.graph_store import build_graph
from mcgr.retrieval.evidence import path_evidence


def test_path_evidence_recovers_directed_triples() -> None:
    # two edge-disjoint paths a->b->c and a->d->c; pair index gives direction
    g = build_graph([("a", "r1", "b"), ("b", "r2", "c"), ("a", "r3", "d"), ("d", "r4", "c")])
    pair_index = {
        frozenset(("a", "b")): [("a", "r1", "b")],
        frozenset(("b", "c")): [("b", "r2", "c")],
        frozenset(("a", "d")): [("a", "r3", "d")],
        frozenset(("d", "c")): [("d", "r4", "c")],
    }
    triples = set(path_evidence(g, "a", "c", pair_index))
    assert ("a", "r1", "b") in triples
    assert ("d", "r4", "c") in triples
    # both independent paths contribute (4 directed triples total)
    assert len(triples) == 4


def test_path_evidence_empty_when_unreachable() -> None:
    g = build_graph([("a", "r", "b"), ("c", "r", "d")])
    assert path_evidence(g, "a", "d", {}) == []
