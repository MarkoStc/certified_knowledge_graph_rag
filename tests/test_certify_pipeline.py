"""Certificate pipeline wiring (hermetic + a small real-data check)."""

import pytest

from mcgr.certify.pipeline import certify_dataset


def test_unsupported_kg_provider_raises() -> None:
    with pytest.raises(NotImplementedError, match="Freebase-free KG"):
        next(certify_dataset("webqsp", "test"))


@pytest.mark.data
def test_metaqa_1hop_is_all_k0() -> None:
    # A 1-hop answer is a single fact -> no path redundancy -> k = 0
    # (or -1 if the gold answer node is absent). Never k >= 1.
    results = list(certify_dataset("metaqa-1hop", "dev", limit=50))
    assert results
    assert all(cq.k <= 0 for cq in results)


@pytest.mark.data
def test_metaqa_3hop_shows_redundancy() -> None:
    # 3-hop MetaQA queries traverse a dense KB region; most admit k >= 1.
    results = list(certify_dataset("metaqa-3hop", "dev", limit=20))
    assert results
    assert sum(cq.k >= 1 for cq in results) >= len(results) // 2
