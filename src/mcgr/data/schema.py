"""Uniform record schema every dataset loader emits (AGENTS.md P1).

``{question, answers, anchor_entities, gold_paths (if any), source_dataset}``
plus split/qid bookkeeping. ``gold_paths`` is a tuple of paths, each path a
tuple of (subject, predicate, object) triples — only 2WikiMultiHopQA provides
these natively; others leave it empty.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Triple = tuple[str, str, str]
Path_ = tuple[Triple, ...]


@dataclass(frozen=True)
class QARecord:
    qid: str
    question: str
    answers: tuple[str, ...]
    anchor_entities: tuple[str, ...]
    gold_paths: tuple[Path_, ...]
    source_dataset: str
    split: str
    meta: dict[str, Any] = field(default_factory=dict)


def data_root() -> Path:
    """Dataset root: $MCGR_DATA_ROOT if set, else $SCRATCH/data."""
    if root := os.environ.get("MCGR_DATA_ROOT"):
        return Path(root)
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise RuntimeError("neither MCGR_DATA_ROOT nor SCRATCH is set")
    return Path(scratch) / "data"


LoaderFn = Any  # Callable[[str], Iterator[QARecord]] — kept loose for per-loader kwargs

_REGISTRY: dict[str, LoaderFn] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def load_dataset(name: str, split: str, **kwargs) -> Iterator[QARecord]:
    """Load records for ``name`` (e.g. '2wikimultihopqa', 'metaqa-2hop')."""
    if name.startswith("metaqa-"):
        hops = int(name.removeprefix("metaqa-").removesuffix("hop"))
        return _REGISTRY["metaqa"](split, hops=hops, **kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](split, **kwargs)
