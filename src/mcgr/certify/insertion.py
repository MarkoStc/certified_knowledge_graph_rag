"""Insertion-aware certificate (AGENTS.md P5 — CORE, beyond textbook Menger).

The deletion certificate asks how many triples an adversary must *remove* to
disconnect the answer. The insertion certificate asks the dual: how many
triples must an adversary *insert* before some WRONG answer out-supports the
true one?

Under the plurality-of-independent-paths abstraction (the reasoner favours
the answer with the most edge-disjoint anchor->answer support), and the
conservative assumption that each inserted triple adds at most one competing
edge-disjoint path to a wrong answer (achieved by a direct forged
anchor->wrong edge), the true answer ``a`` survives insertion of ``b``
triples iff

    paths(a) > max_{a' != a} paths(a') + b .

So the certified insertion budget is the *support margin* over the strongest
competitor:

    b_ins(a) = paths(a) - max_{a' in competitors} paths(a') - 1 .

b_ins >= 0 means no adversary inserting <= b_ins triples can make any listed
competitor reach or exceed ``a``'s support. This is strictly stronger to
establish than deletion `k` (it depends on competitors' real support), and
reduces to deletion `k` only when no competitor has any support. The formal
threat model and its assumptions are in ``threat_model.md``.
"""

import networkx as nx

from mcgr.certify.menger import anchor_set_path_count


def insertion_certificate(
    graph: nx.Graph, anchors: list[str], answer: str, competitors: list[str]
) -> int:
    """Certified insertion budget for ``answer`` against ``competitors``.

    Returns paths(answer) - max_c paths(c) - 1. A value >= 0 is the number of
    inserted triples provably insufficient to flip the plurality decision;
    -1 means a competitor already ties or beats the answer's support.
    """
    p_answer = anchor_set_path_count(graph, anchors, answer)
    p_best_competitor = max(
        (anchor_set_path_count(graph, anchors, c) for c in competitors if c != answer),
        default=0,
    )
    return p_answer - p_best_competitor - 1
