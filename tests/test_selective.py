"""Risk-coverage / AURC metrics (hermetic)."""

from mcgr.eval.selective import aurc, risk_coverage


def test_perfect_signal_low_aurc() -> None:
    # signal perfectly ranks correct (high) above wrong (low)
    signals = [3.0, 2.0, 1.0, 0.0]
    correct = [True, True, False, False]
    pts = risk_coverage(signals, correct)
    # first two answered -> 0 risk; then risk climbs
    assert pts[0] == (0.25, 0.0)
    assert pts[1] == (0.5, 0.0)
    # a good signal has lower AURC than a bad (reversed) one
    bad = aurc([0.0, 1.0, 2.0, 3.0], correct)
    good = aurc(signals, correct)
    assert good < bad


def test_tie_group_all_or_nothing() -> None:
    # all equal signal -> single group at full coverage, risk = overall error
    pts = risk_coverage([1, 1, 1, 1], [True, False, True, False])
    assert pts == [(1.0, 0.5)]


def test_aurc_bounds() -> None:
    # all correct -> zero risk everywhere -> AURC 0
    assert aurc([1.0, 2.0, 3.0], [True, True, True]) == 0.0
    # all wrong -> risk 1 everywhere -> AURC 1
    assert abs(aurc([1.0, 2.0, 3.0], [False, False, False]) - 1.0) < 1e-9
