"""2WikiMultiHopQA loader — the primary dataset (triple-level gold paths).

Source: data_ids_april7.zip → extracted/{train,dev,test}.json.
``evidences`` is a list of [subject, relation, object] triples; we treat it
as the single gold path. Anchor entities are the path roots: subjects that
never appear as an object (for comparison-type questions both compared
entities qualify).
"""

import json
from collections.abc import Iterator

from mcgr.data.schema import QARecord, data_root, register


@register("2wikimultihopqa")
def load(split: str) -> Iterator[QARecord]:
    path = data_root() / "2wikimultihopqa" / "extracted" / f"{split}.json"
    with path.open() as f:
        records = json.load(f)
    for r in records:
        evidences = tuple(tuple(t) for t in r.get("evidences", []))
        objects = {t[2] for t in evidences}
        anchors = tuple(dict.fromkeys(t[0] for t in evidences if t[0] not in objects))
        yield QARecord(
            qid=r["_id"],
            question=r["question"],
            answers=(r["answer"],) if r.get("answer") else (),
            anchor_entities=anchors,
            gold_paths=(evidences,) if evidences else (),
            source_dataset="2wikimultihopqa",
            split=split,
            meta={
                "type": r.get("type"),
                "evidences_id": r.get("evidences_id"),
                "answer_id": r.get("answer_id"),
                "supporting_facts": r.get("supporting_facts"),
            },
        )
