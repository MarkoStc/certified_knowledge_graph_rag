"""WebQSP loader (Freebase-grounded KGQA).

Source: WebQSP.{train,test}.json ({'Questions': [...]}). Each question has
one or more Parses; we take the topic entity (MID + name) as the anchor and
union the answer entities across parses. Gold paths need Freebase to
materialize (P2), so they are left empty here — the inference chain is the
parse's relation sequence, recorded in meta.
"""

import json
from collections.abc import Iterator

from mcgr.data.schema import QARecord, data_root, register

_SPLIT_FILE = {"train": "WebQSP.train.json", "test": "WebQSP.test.json"}


@register("webqsp")
def load(split: str) -> Iterator[QARecord]:
    path = data_root() / "webqsp" / "extracted" / "WebQSP" / "data" / _SPLIT_FILE[split]
    with path.open() as f:
        blob = json.load(f)
    for q in blob["Questions"]:
        answers: list[str] = []
        anchors: list[str] = []
        chains: list[list[str]] = []
        for parse in q.get("Parses", []):
            if name := parse.get("TopicEntityName"):
                anchors.append(name)
            for a in parse.get("Answers", []):
                if a.get("EntityName"):
                    answers.append(a["EntityName"])
                elif a.get("AnswerArgument"):
                    answers.append(a["AnswerArgument"])
            if chain := parse.get("InferentialChain"):
                chains.append(chain)
        yield QARecord(
            qid=q["QuestionId"],
            question=q["RawQuestion"],
            answers=tuple(dict.fromkeys(answers)),
            anchor_entities=tuple(dict.fromkeys(anchors)),
            gold_paths=(),
            source_dataset="webqsp",
            split=split,
            meta={
                "topic_mids": [
                    p.get("TopicEntityMid") for p in q.get("Parses", []) if p.get("TopicEntityMid")
                ],
                "inferential_chains": chains,
            },
        )
