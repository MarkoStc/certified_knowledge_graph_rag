"""Text-constructed KG from HotpotQA context (hermetic)."""

from mcgr.certify import query_certificate
from mcgr.kg.text_kg import build_hotpot_kg


def make_record(question, answer, context):
    return {"question": question, "answer": answer, "context": context}


def test_single_chain_gives_k0() -> None:
    # Alpha mentions Beta mentions the answer -> one path -> k=0
    rec = make_record(
        "Where did Alpha work?",
        "Gamma Corp",
        {
            "title": ["Alpha", "Beta", "Gamma Corp"],
            "sentences": [
                ["Alpha worked with Beta."],
                ["Beta founded Gamma Corp."],
                ["Gamma Corp is a firm."],
            ],
        },
    )
    g, anchors, ans = build_hotpot_kg(rec)
    assert anchors == ["Alpha"]
    assert ans == "Gamma Corp"
    assert query_certificate(g, anchors, ans) == 0


def test_two_independent_mentions_give_k1() -> None:
    # Alpha reaches the Target via two distinct intermediates -> k=1
    rec = make_record(
        "What is linked to Alpha?",
        "Target",
        {
            "title": ["Alpha", "Beta", "Cappa", "Target"],
            "sentences": [
                ["Alpha relates to Beta and Cappa."],
                ["Beta is about Target."],
                ["Cappa also concerns Target."],
                ["Target is a thing."],
            ],
        },
    )
    g, anchors, ans = build_hotpot_kg(rec)
    assert query_certificate(g, anchors, ans) == 1


def test_yes_no_answer_is_uncertifiable() -> None:
    rec = make_record(
        "Are Alpha and Beta the same?",
        "yes",
        {
            "title": ["Alpha", "Beta"],
            "sentences": [["Alpha is a director."], ["Beta is a director."]],
        },
    )
    _g, _anchors, ans = build_hotpot_kg(rec)
    assert ans is None  # 'yes' is not an entity node
