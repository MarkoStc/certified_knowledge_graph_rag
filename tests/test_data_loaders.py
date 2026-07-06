"""Dataset loader tests (AGENTS.md P1: hand-verified examples per dataset).

Two layers:
 - hermetic tests of parsing logic on synthetic files (always run);
 - ``data``-marked tests asserting hand-checked real records, which read
   from $SCRATCH and are deselected by default (run with `pytest -m data`).
"""

import json
from itertools import islice
from pathlib import Path

import pytest

from mcgr.data import load_dataset
from mcgr.data.schema import QARecord


# ---------------------------------------------------------------- hermetic --
def test_twowiki_parses_anchors_and_gold_path(tmp_path: Path, monkeypatch) -> None:
    record = {
        "_id": "q1",
        "question": "Who is the mother of the director of film X?",
        "answer": "Alice",
        "type": "compositional",
        "evidences": [["X", "director", "Bob"], ["Bob", "mother", "Alice"]],
    }
    root = tmp_path / "2wikimultihopqa" / "extracted"
    root.mkdir(parents=True)
    (root / "dev.json").write_text(json.dumps([record]))
    monkeypatch.setenv("MCGR_DATA_ROOT", str(tmp_path))

    rec = next(iter(load_dataset("2wikimultihopqa", "dev")))
    assert rec.answers == ("Alice",)
    # X is a subject that is never an object -> the anchor; Bob is internal.
    assert rec.anchor_entities == ("X",)
    assert rec.gold_paths == ((("X", "director", "Bob"), ("Bob", "mother", "Alice")),)


def test_metaqa_extracts_bracketed_anchor(tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "metaqa" / "MetaQA" / "2-hop" / "vanilla"
    d.mkdir(parents=True)
    (d / "qa_dev.txt").write_text("what shares actors with [Movie A]\tX|Y|Z\n")
    monkeypatch.setenv("MCGR_DATA_ROOT", str(tmp_path))

    rec = next(iter(load_dataset("metaqa-2hop", "dev")))
    assert rec.anchor_entities == ("Movie A",)
    assert rec.answers == ("X", "Y", "Z")
    assert rec.source_dataset == "metaqa-2hop"


def test_musique_reconstructs_chain(tmp_path: Path, monkeypatch) -> None:
    record = {
        "id": "m1",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Carol",
        "answer_aliases": ["C."],
        "answerable": True,
        "question_decomposition": [
            {"question": "Green >> performer", "answer": "Dave"},
            {"question": "#1 >> spouse", "answer": "Carol"},
        ],
    }
    d = tmp_path / "musique" / "extracted" / "data"
    d.mkdir(parents=True)
    (d / "musique_ans_v1.0_dev.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setenv("MCGR_DATA_ROOT", str(tmp_path))

    rec = next(iter(load_dataset("musique", "dev")))
    assert rec.anchor_entities == ("Green",)
    assert rec.gold_paths == ((("Green", "performer", "Dave"), ("#1", "spouse", "Carol")),)
    assert rec.answers == ("Carol", "C.")


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        next(iter(load_dataset("nope", "dev")))


def test_metaqa_rejects_bad_hop_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MCGR_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="hops must be"):
        next(iter(load_dataset("metaqa-4hop", "dev")))


# --------------------------------------------------------- real data (-m) --
@pytest.mark.data
def test_twowiki_first_dev_record_is_hand_verified() -> None:
    rec = next(iter(load_dataset("2wikimultihopqa", "dev")))
    assert rec.question == "Who is the mother of the director of film Polish-Russian War (Film)?"
    assert rec.answers == ("Małgorzata Braunek",)
    assert rec.anchor_entities == ("Polish-Russian War",)
    assert rec.gold_paths == (
        (
            ("Polish-Russian War", "director", "Xawery Żuławski"),
            ("Xawery Żuławski", "mother", "Małgorzata Braunek"),
        ),
    )


@pytest.mark.data
@pytest.mark.parametrize(
    ("name", "split"),
    [
        ("2wikimultihopqa", "dev"),
        ("metaqa-1hop", "dev"),
        ("metaqa-2hop", "dev"),
        ("metaqa-3hop", "dev"),
        ("musique", "dev"),
        ("webqsp", "test"),
        ("cwq", "dev"),
        ("hotpotqa", "validation"),
    ],
)
def test_every_loader_yields_wellformed_records(name: str, split: str) -> None:
    recs = list(islice(load_dataset(name, split), 20))
    assert recs, f"{name} yielded nothing"
    for r in recs:
        assert isinstance(r, QARecord)
        assert r.qid and r.question
        assert r.source_dataset.startswith(name.split("-")[0])
        assert isinstance(r.answers, tuple)
