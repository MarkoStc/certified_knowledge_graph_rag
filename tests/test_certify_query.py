"""Anchor-set certificate + k-hop subgraph, on hand-checked graphs."""

import networkx as nx

from mcgr.certify import anchor_set_path_count, query_certificate
from mcgr.kg.graph_store import build_graph, khop_subgraph, prune_hubs, reconnect_endpoints


def test_single_anchor_matches_pair_count() -> None:
    # a-b-c, a-d-c: two edge-disjoint paths -> count 2, k = 1
    g = nx.Graph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")])
    assert anchor_set_path_count(g, ["a"], "c") == 2
    assert query_certificate(g, ["a"], "c") == 1


def test_multi_anchor_super_source() -> None:
    # from {a, x} to c: a-b-c, a-d-c, x-c are 3 edge-disjoint paths.
    g = nx.Graph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c"), ("x", "c"), ("x", "b")])
    assert anchor_set_path_count(g, ["a", "x"], "c") == 3
    assert query_certificate(g, ["a", "x"], "c") == 2


def test_unsupported_answer_is_minus_one() -> None:
    g = nx.Graph([("a", "b"), ("c", "d")])
    assert anchor_set_path_count(g, ["a"], "d") == 0
    assert query_certificate(g, ["a"], "d") == -1


def test_absent_anchor_ignored() -> None:
    g = nx.Graph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")])
    # 'ghost' anchor not in graph is dropped; real anchor 'a' still gives 2
    assert anchor_set_path_count(g, ["ghost", "a"], "c") == 2


# ------------------------------------------------------------- graph_store --
def test_build_graph_collapses_parallel_facts() -> None:
    g = build_graph([("m", "directed_by", "p"), ("m", "written_by", "p"), ("m", "in_lang", "en")])
    assert g.number_of_edges() == 2  # (m,p) collapsed, (m,en) separate
    assert g["m"]["p"]["relations"] == {"directed_by", "written_by"}


def test_khop_subgraph_bounds_radius() -> None:
    # path graph 0-1-2-3-4; 2-hop ball around 0 is {0,1,2}
    g = build_graph([(str(i), "r", str(i + 1)) for i in range(4)])
    sub = khop_subgraph(g, ["0"], radius=2)
    assert set(sub.nodes()) == {"0", "1", "2"}


def test_khop_subgraph_preserves_alternate_paths() -> None:
    # diamond within radius is fully retained, so the certificate is exact
    g = build_graph([("a", "r", "b"), ("b", "r", "c"), ("a", "r", "d"), ("d", "r", "c")])
    sub = khop_subgraph(g, ["a"], radius=2)
    assert anchor_set_path_count(sub, ["a"], "c") == 2


def test_prune_hubs_removes_shared_type_paths() -> None:
    # a and c both point to a "human" hub H that also links b..e: without
    # pruning, a-H-c is a spurious second path; pruning H removes it.
    edges = [("a", "mother", "c")]  # the one true fact
    edges += [(n, "instance_of", "H") for n in ("a", "b", "d", "e", "c")]
    g = build_graph(edges)
    assert anchor_set_path_count(g, ["a"], "c") == 2  # true edge + a-H-c
    pruned, hub_adj = prune_hubs(g, max_degree=3)
    assert "H" in hub_adj
    assert anchor_set_path_count(pruned, ["a"], "c") == 1  # only the real fact


def test_reconnect_endpoints_restores_hub_answer() -> None:
    # Answer B is a high-degree hub linking many entities; a reaches B both
    # directly and via m. Pruning drops B (transit hub), but as THIS query's
    # answer it must be restored — without reopening transit through it.
    edges = [("a", "r", "B"), ("a", "r", "m"), ("m", "r", "B")]
    edges += [(n, "r", "B") for n in ("c", "d", "e", "f")]  # makes B a hub
    g = build_graph(edges)
    pruned, hub_adj = prune_hubs(g, max_degree=3)
    assert "B" in hub_adj and "B" not in pruned  # B pruned as a transit hub

    sub = khop_subgraph(pruned, ["a"], radius=2)
    assert "B" not in sub  # unsupported before reconnection
    reconnect_endpoints(sub, hub_adj, ["a", "B"])
    # a->B direct and a->m->B are two edge-disjoint paths -> k = 1
    assert anchor_set_path_count(sub, ["a"], "B") == 2
