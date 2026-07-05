"""Experiment config loading (AGENTS.md §1: every experiment is driven by a
versioned ``configs/*.yaml`` + a fixed seed + a logged git commit hash).

Configs are flat-ish YAML; unknown keys are rejected so a typo in a config
fails loudly instead of silently falling back to a default.
"""

import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """The common envelope every experiment config must carry.

    Phase-specific settings go in ``params`` (a free dict) until a phase
    stabilizes enough to promote its keys to typed fields.
    """

    name: str
    seed: int
    dataset: str
    params: dict[str, Any] = field(default_factory=dict)

    # populated at load time, not from YAML
    config_path: str = ""
    git_commit: str = ""


def current_git_commit(repo_root: Path | None = None) -> str:
    """Short commit hash of the repo, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config, stamping path + git commit."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a YAML mapping, got {type(raw).__name__}")

    allowed = {f.name for f in fields(ExperimentConfig)} - {"config_path", "git_commit"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{path}: unknown config keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    missing = {"name", "seed", "dataset"} - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required config keys {sorted(missing)}")

    return ExperimentConfig(
        **raw,
        config_path=str(path),
        git_commit=current_git_commit(path.resolve().parent),
    )
