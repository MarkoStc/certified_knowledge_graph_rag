"""Structured logging + the local CSV run-log backup (AGENTS.md §2).

``get_logger`` gives a consistently formatted stdlib logger.
``RunLogger`` appends metric rows to a CSV under the run directory —
this is the always-on local backup regardless of wandb.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class RunLogger:
    """Append metric rows to ``<run_dir>/metrics.csv``.

    Columns are fixed by the first row logged; later rows must use the
    same keys (missing keys become empty, unknown keys are an error) so
    the CSV stays rectangular and analysis-friendly.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None
        if self.csv_path.exists():
            with self.csv_path.open() as f:
                header = f.readline().strip()
            if header:
                self._fieldnames = header.split(",")

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row)
            write_header = True
        else:
            unknown = set(row) - set(self._fieldnames)
            if unknown:
                raise ValueError(
                    f"row keys {sorted(unknown)} not in established columns {self._fieldnames}"
                )
            write_header = False
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
