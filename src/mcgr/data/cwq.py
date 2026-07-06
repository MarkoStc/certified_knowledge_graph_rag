"""ComplexWebQuestions loader (compositional multi-hop over Freebase).

Source: ComplexWebQuestions_{train,dev,test}.json (a JSON list). Answers
carry an entity name + MID + aliases. Anchors are not annotated directly;
the SPARQL query encodes them (materialized against Freebase in P2), so we
record the SPARQL in meta and leave anchor_entities empty here.
"""

import json
from collections.abc import Iterator

from mcgr.data.schema import QARecord, data_root, register

_SPLIT_FILE = {
    "train": "ComplexWebQuestions_train.json",
    "dev": "ComplexWebQuestions_dev.json",
    "test": "ComplexWebQuestions_test.json",
}


@register("cwq")
def load(split: str) -> Iterator[QARecord]:
    path = data_root() / "cwq" / "extracted" / _SPLIT_FILE[split]
    with path.open() as f:
        records = json.load(f)
    for r in records:
        answers = []
        for a in r.get("answers", []):
            if a.get("answer"):
                answers.append(a["answer"])
            answers.extend(a.get("aliases", []))
        yield QARecord(
            qid=r["ID"],
            question=r["question"],
            answers=tuple(dict.fromkeys(answers)),
            anchor_entities=(),
            gold_paths=(),
            source_dataset="cwq",
            split=split,
            meta={
                "compositionality_type": r.get("compositionality_type"),
                "sparql": r.get("sparql"),
                "webqsp_id": r.get("webqsp_ID"),
            },
        )
