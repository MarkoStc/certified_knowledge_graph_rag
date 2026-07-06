"""MuSiQue loader (harder multi-hop text QA; answerable subset).

Source: musique_ans_v1.0_{train,dev}.jsonl (test has no answers).
``question_decomposition`` gives an ordered chain of sub-questions with
per-hop answers. Only ~66% of hops use the structured ``"<entity> >>
<relation>"`` template; the rest are natural-language sub-questions that
don't decompose into a clean triple. We build ``gold_paths`` from the
structured hops only (honest, possibly truncated) and keep the *full* raw
decomposition in ``meta['decomposition']`` so P2 KG construction loses
nothing. The first structured hop's entity is the anchor.
"""

import json
from collections.abc import Iterator

from mcgr.data.schema import QARecord, data_root, register

_SPLIT_FILE = {
    "train": "musique_ans_v1.0_train.jsonl",
    "dev": "musique_ans_v1.0_dev.jsonl",
    "test": "musique_full_v1.0_test.jsonl",
}


def _parse_hop(sub_q: str, answer: str) -> tuple[str, str, str] | None:
    entity, sep, relation = sub_q.partition(">>")
    if not sep:
        return None
    return (entity.strip(), relation.strip(), answer)


@register("musique")
def load(split: str) -> Iterator[QARecord]:
    path = data_root() / "musique" / "extracted" / "data" / _SPLIT_FILE[split]
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            decomp = r.get("question_decomposition", [])
            triples = tuple(
                t for hop in decomp if (t := _parse_hop(hop["question"], hop.get("answer", "")))
            )
            answer = r.get("answer", "")
            answers = (answer, *r.get("answer_aliases", [])) if answer else ()
            yield QARecord(
                qid=r["id"],
                question=r["question"],
                answers=tuple(dict.fromkeys(answers)),
                anchor_entities=(triples[0][0],) if triples else (),
                gold_paths=(triples,) if triples else (),
                source_dataset="musique",
                split=split,
                meta={
                    "answerable": r.get("answerable"),
                    "n_hops": len(decomp),
                    "decomposition": decomp,
                },
            )
