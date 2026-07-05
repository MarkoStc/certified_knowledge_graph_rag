"""check_prereqs wired as pytest (AGENTS.md P0).

The full §3 verification is marked ``prereqs`` and deselected by default
(see pyproject addopts): it depends on $SCRATCH state and external
services, and stays red by design until P1 downloads and the
human-required items land. Run explicitly with:

    uv run pytest -m prereqs --no-header -rA

The unmarked tests below cover the checker's own logic and always run.
"""

from pathlib import Path

import pytest

from mcgr.prereqs import (
    check_datasets,
    check_lockfile_current,
    load_manifest,
    run_all_checks,
    sha256_of,
)


def test_manifest_parses_and_covers_section_3() -> None:
    manifest = load_manifest()
    dataset_names = {d["name"] for d in manifest["datasets"]}
    assert {"2wikimultihopqa", "metaqa", "hotpotqa", "musique", "webqsp", "cwq"} <= dataset_names
    model_names = {m["name"] for m in manifest["models"]}
    assert any("Qwen2.5-14B" in n for n in model_names)
    assert any(m["gated"] for m in manifest["models"])  # Llama is the gated one


def test_sha256_of_known_content(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"mcgr")
    # printf 'mcgr' | sha256sum
    assert sha256_of(p) == "7a7557d91f61137318f7dccb6035f2a869a6e9b1723e45fdcd1a3c4eba73f364"


def test_dataset_check_reports_empty_manifest_entry() -> None:
    results = check_datasets({"datasets": [{"name": "2wikimultihopqa", "files": []}]})
    assert len(results) == 1
    assert not results[0].ok
    assert "no files recorded" in results[0].detail


def test_lockfile_is_current() -> None:
    result = check_lockfile_current()
    assert result.ok, result.detail


@pytest.mark.prereqs
def test_all_section_3_prerequisites_present() -> None:
    failures = [f"{r.item}: {r.detail}" for r in run_all_checks() if not r.ok]
    assert not failures, "missing prerequisites:\n" + "\n".join(failures)
