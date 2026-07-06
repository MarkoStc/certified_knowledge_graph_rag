"""MetaQA loader (1/2/3-hop movie QA over a self-contained KB).

QA lines: ``question text with [Anchor Entity]\\tans1|ans2|...``
KB lines (kb.txt): ``subject|relation|object``.
"""

from collections.abc import Iterator
from pathlib import Path

from mcgr.data.schema import QARecord, Triple, data_root, register

_ROOT = "metaqa/MetaQA"


@register("metaqa")
def load(split: str, hops: int = 2) -> Iterator[QARecord]:
    if hops not in (1, 2, 3):
        raise ValueError("MetaQA hops must be 1, 2, or 3")
    path = data_root() / _ROOT / f"{hops}-hop" / "vanilla" / f"qa_{split}.txt"
    with path.open() as f:
        for i, line in enumerate(f):
            question, _, answer_str = line.rstrip("\n").partition("\t")
            start = question.find("[")
            end = question.find("]", start)
            anchor = question[start + 1 : end] if 0 <= start < end else ""
            yield QARecord(
                qid=f"metaqa-{hops}hop-{split}-{i}",
                question=question,
                answers=tuple(a for a in answer_str.split("|") if a),
                anchor_entities=(anchor,) if anchor else (),
                gold_paths=(),
                source_dataset=f"metaqa-{hops}hop",
                split=split,
                meta={"hops": hops},
            )


def load_kb(path: Path | None = None) -> list[Triple]:
    """The full MetaQA movie KB as (subject, relation, object) triples."""
    path = path or data_root() / _ROOT / "kb" / "kb.txt"
    triples = []
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples
