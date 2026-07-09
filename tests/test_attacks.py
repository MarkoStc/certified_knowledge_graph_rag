"""RAG-Safety insertion attack (hermetic)."""

from mcgr.attacks.rag_safety import InsertionAttack

GOLD = [("X", "director", "Bob"), ("Bob", "mother", "Alice")]


def test_forge_full_competing_chain() -> None:
    atk = InsertionAttack(budget=2, seed=0)
    forged = atk.forge_chain("X", GOLD, "Mallory")
    # two triples: X -director-> fake, fake -mother-> Mallory
    assert len(forged) == 2
    assert forged[0][0] == "X" and forged[0][1] == "director"
    assert forged[-1][1] == "mother" and forged[-1][2] == "Mallory"
    # the chain connects: object of hop 0 is subject of hop 1
    assert forged[0][2] == forged[1][0]


def test_budget_one_is_partial() -> None:
    atk = InsertionAttack(budget=1, seed=0)
    forged = atk.forge_chain("X", GOLD, "Mallory")
    assert len(forged) == 1
    assert forged[0] == ("X", "director", "Mallory")


def test_extra_budget_adds_corroboration() -> None:
    atk = InsertionAttack(budget=4, seed=0)
    forged = atk.forge_chain("X", GOLD, "Mallory")
    assert len(forged) == 4
    # every triple beyond the 2-hop chain corroborates (…, mother, Mallory)
    corroborating = [t for t in forged[2:] if t[1] == "mother" and t[2] == "Mallory"]
    assert len(corroborating) == 2


def test_deterministic_under_seed() -> None:
    a = InsertionAttack(budget=3, seed=7).forge_chain("X", GOLD, "M")
    b = InsertionAttack(budget=3, seed=7).forge_chain("X", GOLD, "M")
    c = InsertionAttack(budget=3, seed=8).forge_chain("X", GOLD, "M")
    assert a == b
    assert a != c  # different seed -> different fake intermediates


def test_apply_appends_to_context() -> None:
    ctx = [("P", "q", "R")]
    out = InsertionAttack(budget=2, seed=0).apply(ctx, "X", GOLD, "Mallory")
    assert out[0] == ("P", "q", "R")
    assert len(out) == 3


def test_zero_budget_noop() -> None:
    assert InsertionAttack(budget=0).forge_chain("X", GOLD, "M") == []
