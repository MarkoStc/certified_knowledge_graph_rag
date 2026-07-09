"""Selective-prediction metrics (AGENTS.md P10, §7).

A selective predictor answers the queries it trusts most and abstains on the
rest. Given a trust ``signal`` per query and whether each answer is
``correct``, the risk-coverage curve plots error rate (risk) against the
fraction answered (coverage) as the abstention threshold sweeps. AURC (area
under it) summarizes the signal: lower = better (correct answers ranked
before wrong ones).
"""

from itertools import pairwise


def risk_coverage(signals: list[float], correct: list[bool]) -> list[tuple[float, float]]:
    """Risk-coverage points, most-trusted first.

    Ties in ``signal`` are grouped (all-or-nothing at a coverage step) so a
    coarse integer signal like ``k`` is not flattered by lucky tie ordering.
    Returns [(coverage, risk), ...] with increasing coverage.
    """
    if len(signals) != len(correct):
        raise ValueError("signals and correct must be the same length")
    n = len(signals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: signals[i], reverse=True)
    points = []
    answered = 0
    errors = 0
    i = 0
    while i < n:
        # consume a whole tie-group of equal signal
        j = i
        s = signals[order[i]]
        while j < n and signals[order[j]] == s:
            answered += 1
            errors += not correct[order[j]]
            j += 1
        points.append((answered / n, errors / answered))
        i = j
    return points


def aurc(signals: list[float], correct: list[bool]) -> float:
    """Area under the risk-coverage curve (trapezoidal). Lower is better."""
    pts = risk_coverage(signals, correct)
    if not pts:
        return 0.0
    pts = [(0.0, pts[0][1]), *pts]  # anchor at coverage 0
    area = 0.0
    for (c0, r0), (c1, r1) in pairwise(pts):
        area += (c1 - c0) * (r0 + r1) / 2
    return area
