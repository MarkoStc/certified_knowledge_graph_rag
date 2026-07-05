"""End-to-end smoke test: compute the deletion certificate on toy graphs.

``uv run mcgr-demo`` proves the environment, package, and certificate
core all work. Not a research entry point — real experiments are driven
by configs (AGENTS.md §1).
"""

import sys

import networkx as nx

from mcgr import __version__
from mcgr.certify import deletion_certificate

TOY_CASES: list[tuple[str, list[tuple[str, str]], int]] = [
    ("single chain a->b->c", [("a", "b"), ("b", "c")], 0),
    ("two edge-disjoint paths", [("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")], 1),
    (
        "two paths sharing edge x->c",
        [("a", "b"), ("b", "x"), ("a", "d"), ("d", "x"), ("x", "c")],
        0,
    ),
]


def main() -> int:
    print(f"mcgr {__version__} — deletion certificate on hand-checked toy graphs")
    failures = 0
    for name, edges, expected_k in TOY_CASES:
        k = deletion_certificate(nx.DiGraph(edges), "a", "c")
        status = "ok" if k == expected_k else "MISMATCH"
        if k != expected_k:
            failures += 1
        print(f"  {name}: k = {k} (expected {expected_k}) [{status}]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
