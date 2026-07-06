"""HotpotQA loader (multi-hop text QA; KG constructed from text in P2).

Source: HF parquet conversion. The distractor setting is standard for
answer accuracy; ``supporting_facts`` names the gold (title, sentence)
pairs, which P2 turns into a text-constructed KG. No triple-level gold
paths natively, so ``gold_paths`` is empty here; anchors are the
supporting-fact article titles.
"""

from collections.abc import Iterator

import pyarrow.parquet as pq

from mcgr.data.schema import QARecord, data_root, register

_SPLIT_FILES = {
    "train": ["hotpot_distractor_train_0.parquet", "hotpot_distractor_train_1.parquet"],
    "validation": ["hotpot_distractor_validation_0.parquet"],
}


@register("hotpotqa")
def load(split: str) -> Iterator[QARecord]:
    if split == "dev":  # accept the common alias
        split = "validation"
    root = data_root() / "hotpotqa"
    for fname in _SPLIT_FILES[split]:
        table = pq.read_table(root / fname)
        for r in table.to_pylist():
            titles = r.get("supporting_facts", {}).get("title", [])
            yield QARecord(
                qid=r["id"],
                question=r["question"],
                answers=(r["answer"],) if r.get("answer") else (),
                anchor_entities=tuple(dict.fromkeys(titles)),
                gold_paths=(),
                source_dataset="hotpotqa",
                split=split,
                meta={
                    "type": r.get("type"),
                    "level": r.get("level"),
                    "supporting_facts": r.get("supporting_facts"),
                },
            )
