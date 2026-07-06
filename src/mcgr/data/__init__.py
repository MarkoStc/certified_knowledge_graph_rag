"""Dataset loaders emitting a uniform QARecord schema (AGENTS.md P1).

Importing this package registers every loader; use ``load_dataset``.
"""

from mcgr.data import cwq, hotpotqa, metaqa, musique, twowiki, webqsp  # noqa: F401 (register side-effect)
from mcgr.data.schema import QARecord, load_dataset

__all__ = ["QARecord", "load_dataset"]
