"""Wikidata triple extraction (hermetic — no network)."""

from mcgr.kg.wikidata import entity_triples


def test_entity_triples_keeps_only_entity_valued() -> None:
    entity = {
        "id": "Q1",
        "claims": {
            "P57": [  # director -> entity (kept)
                {
                    "mainsnak": {
                        "datatype": "wikibase-item",
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q2"}},
                    }
                }
            ],
            "P577": [  # publication date -> time literal (dropped)
                {"mainsnak": {"datatype": "time", "snaktype": "value", "datavalue": {"value": {}}}}
            ],
            "P345": [  # IMDb id -> external-id literal (dropped)
                {
                    "mainsnak": {
                        "datatype": "external-id",
                        "snaktype": "value",
                        "datavalue": {"value": "tt123"},
                    }
                }
            ],
            "P170": [  # somevalue snak, no target (dropped)
                {"mainsnak": {"datatype": "wikibase-item", "snaktype": "somevalue"}}
            ],
        },
    }
    assert entity_triples(entity) == [("Q1", "P57", "Q2")]


def test_entity_triples_handles_no_claims() -> None:
    assert entity_triples({"id": "Q9", "claims": {}}) == []
