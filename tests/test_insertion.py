"""Insertion certificate on hand-checked graphs (AGENTS.md P5)."""

from mcgr.certify import insertion_certificate
from mcgr.kg.graph_store import build_graph


def test_margin_over_competitor() -> None:
    # answer c has 2 edge-disjoint paths from a; competitor e has 1.
    g = build_graph(
        [
            ("a", "r", "b"),
            ("b", "r", "c"),
            ("a", "r", "d"),
            ("d", "r", "c"),  # 2 paths a->c
            ("a", "r", "e"),  # 1 path a->e
        ]
    )
    # b_ins = paths(c) - paths(e) - 1 = 2 - 1 - 1 = 0
    assert insertion_certificate(g, ["a"], "c", ["e"]) == 0


def test_no_competitor_support_reduces_to_deletion_k() -> None:
    # answer c has 2 paths; competitor z is unreachable (0 paths).
    g = build_graph([("a", "r", "b"), ("b", "r", "c"), ("a", "r", "d"), ("d", "r", "c")])
    # b_ins = 2 - 0 - 1 = 1, same as deletion k here
    assert insertion_certificate(g, ["a"], "c", ["z"]) == 1


def test_strong_competitor_gives_negative() -> None:
    # competitor e has 2 paths, answer c has 1 -> c does not dominate
    g = build_graph(
        [
            ("a", "r", "b"),
            ("b", "r", "c"),  # 1 path a->c
            ("a", "r", "x"),
            ("x", "r", "e"),
            ("a", "r", "y"),
            ("y", "r", "e"),  # 2 paths a->e
        ]
    )
    assert insertion_certificate(g, ["a"], "c", ["e"]) == -2  # 1 - 2 - 1


def test_ignores_answer_listed_as_competitor() -> None:
    g = build_graph([("a", "r", "b"), ("b", "r", "c"), ("a", "r", "d"), ("d", "r", "c")])
    # listing the answer among competitors must not self-cancel
    assert insertion_certificate(g, ["a"], "c", ["c", "z"]) == 1
