"""Hand-checkable toy-graph cases for the deletion certificate.

AGENTS.md P4: build tiny graphs by hand, compute k by hand, assert the
engine agrees. These exact three cases are specified in the brief.
"""

import networkx as nx

from mcgr.certify import deletion_certificate, edge_disjoint_path_count


def single_chain() -> nx.DiGraph:
    # a -> b -> c : one path, min-cut 1, k = 0
    return nx.DiGraph([("a", "b"), ("b", "c")])


def two_disjoint_paths() -> nx.DiGraph:
    # a -> b -> c and a -> d -> c : two edge-disjoint paths, k = 1
    return nx.DiGraph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")])


def two_paths_sharing_one_edge() -> nx.DiGraph:
    # a -> b -> x -> c and a -> d -> x -> c share edge x -> c:
    # min-cut is 1 (delete x -> c), so k = 0 despite two branches.
    return nx.DiGraph([("a", "b"), ("b", "x"), ("a", "d"), ("d", "x"), ("x", "c")])


def test_single_chain_certifies_zero() -> None:
    g = single_chain()
    assert edge_disjoint_path_count(g, "a", "c") == 1
    assert deletion_certificate(g, "a", "c") == 0


def test_two_disjoint_paths_certify_one() -> None:
    g = two_disjoint_paths()
    assert edge_disjoint_path_count(g, "a", "c") == 2
    assert deletion_certificate(g, "a", "c") == 1


def test_shared_edge_bottleneck_certifies_zero() -> None:
    g = two_paths_sharing_one_edge()
    assert edge_disjoint_path_count(g, "a", "c") == 1
    assert deletion_certificate(g, "a", "c") == 0


def test_no_path_yields_negative_one() -> None:
    g = nx.DiGraph([("a", "b"), ("c", "d")])
    assert edge_disjoint_path_count(g, "a", "d") == 0
    assert deletion_certificate(g, "a", "d") == -1


def test_missing_node_yields_negative_one() -> None:
    g = single_chain()
    assert deletion_certificate(g, "a", "nonexistent") == -1
