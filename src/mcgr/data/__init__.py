"""Dataset loaders emitting a uniform QARecord schema (AGENTS.md P1).

Importing this package registers every loader; use ``load_dataset``.
"""

from mcgr.data import (  # noqa: F401 (register side-effect)
    cwq,
    hotpotqa,
    metaqa,
    musique,
    twowiki,
    webqsp,
)
from mcgr.data.schema import QARecord, load_dataset

__all__ = ["QARecord", "load_dataset"]
