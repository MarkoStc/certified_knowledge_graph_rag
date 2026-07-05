"""P0 plumbing: config loading, seeding determinism, CSV run logging."""

import random
from pathlib import Path

import pytest

from mcgr.config import load_config
from mcgr.logging_utils import RunLogger
from mcgr.seeding import seed_everything


def write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(text)
    return p


def test_load_config_stamps_path_and_commit(tmp_path: Path) -> None:
    p = write_yaml(tmp_path, "name: toy\nseed: 7\ndataset: 2wiki\nparams:\n  hops: 2\n")
    cfg = load_config(p)
    assert (cfg.name, cfg.seed, cfg.dataset) == ("toy", 7, "2wiki")
    assert cfg.params == {"hops": 2}
    assert cfg.config_path.endswith("exp.yaml")
    assert cfg.git_commit  # 'unknown' outside a repo, a hash inside one


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    p = write_yaml(tmp_path, "name: toy\nseed: 7\ndataset: 2wiki\ndatset_typo: x\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_load_config_rejects_missing_keys(tmp_path: Path) -> None:
    p = write_yaml(tmp_path, "name: toy\n")
    with pytest.raises(ValueError, match="missing required"):
        load_config(p)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_run_logger_appends_rectangular_csv(tmp_path: Path) -> None:
    rl = RunLogger(tmp_path / "run1")
    rl.log({"step": 1, "em": 0.5})
    rl.log({"step": 2, "em": 0.6})
    with pytest.raises(ValueError, match="not in established columns"):
        rl.log({"step": 3, "f1": 0.7})
    lines = (tmp_path / "run1" / "metrics.csv").read_text().strip().splitlines()
    assert lines[0] == "step,em"
    assert len(lines) == 3

    # a re-opened logger picks up the existing header
    rl2 = RunLogger(tmp_path / "run1")
    rl2.log({"step": 3, "em": 0.7})
    assert len((tmp_path / "run1" / "metrics.csv").read_text().strip().splitlines()) == 4
