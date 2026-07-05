"""Prerequisite verification (AGENTS.md §3 / P0 gate).

Every §3 item becomes a named check returning a ``CheckResult``. The
source of truth for dataset archives/checksums and model dirs is
``context/manifest.yaml``; ``context/SOURCES.md`` is the human-facing
record derived from it.
"""

import hashlib
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "context" / "manifest.yaml"


@dataclass(frozen=True)
class CheckResult:
    item: str
    ok: bool
    detail: str
    human_required: bool = False


def scratch_dir() -> Path | None:
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) if scratch else None


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def check_hf_token() -> CheckResult:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path.home() / ".config" / "hf" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
    ok = bool(token)
    return CheckResult(
        "HF token (§3.1)",
        ok,
        "found" if ok else "set HF_TOKEN, ~/.config/hf/token, or HF_TOKEN= in .env",
        human_required=True,
    )


def check_datasets(manifest: dict) -> list[CheckResult]:
    results = []
    scratch = scratch_dir()
    for ds in manifest.get("datasets", []):
        name = f"dataset: {ds['name']} (§3.2)"
        if scratch is None:
            results.append(CheckResult(name, False, "$SCRATCH not set"))
            continue
        if not ds.get("files"):
            results.append(CheckResult(name, False, "no files recorded in manifest yet"))
            continue
        missing, bad = [], []
        for f in ds["files"]:
            p = scratch / "data" / ds["name"] / f["path"]
            if not p.exists():
                missing.append(f["path"])
            elif f.get("sha256") and sha256_of(p) != f["sha256"]:
                bad.append(f["path"])
        ok = not missing and not bad
        detail = "all files present, checksums match" if ok else f"missing={missing} bad={bad}"
        results.append(CheckResult(name, ok, detail))
    return results


def check_models(manifest: dict) -> list[CheckResult]:
    results = []
    scratch = scratch_dir()
    for m in manifest.get("models", []):
        name = f"model: {m['name']} (§3.4)"
        human = bool(m.get("gated"))
        if scratch is None:
            results.append(CheckResult(name, False, "$SCRATCH not set", human))
            continue
        p = scratch / "models" / m["name"]
        # a downloaded HF model dir always carries at least a config.json
        ok = (p / "config.json").exists()
        detail = str(p) if ok else f"not downloaded to {p}"
        results.append(CheckResult(name, ok, detail, human))
    return results


def check_freebase_endpoint(manifest: dict) -> CheckResult:
    url = os.environ.get("FREEBASE_ENDPOINT") or manifest.get("freebase_endpoint")
    if not url:
        return CheckResult(
            "Freebase SPARQL endpoint (§3.3)",
            False,
            "no endpoint URL (set FREEBASE_ENDPOINT or manifest freebase_endpoint)",
            human_required=True,
        )
    query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
    try:
        req = urllib.request.Request(f"{url}?query={urllib.parse.quote(query)}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
        return CheckResult("Freebase SPARQL endpoint (§3.3)", ok, url, human_required=True)
    except (urllib.error.URLError, OSError) as e:
        return CheckResult(
            "Freebase SPARQL endpoint (§3.3)", False, f"{url}: {e}", human_required=True
        )


def check_lockfile_current() -> CheckResult:
    try:
        subprocess.run(
            ["uv", "lock", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
        return CheckResult("uv lockfile current (§3.5)", True, "uv lock --check passed")
    except subprocess.CalledProcessError as e:
        return CheckResult(
            "uv lockfile current (§3.5)", False, e.stderr.decode().strip() or "out of date"
        )
    except FileNotFoundError:
        return CheckResult("uv lockfile current (§3.5)", False, "uv not on PATH")


def check_baseline_repos(manifest: dict) -> list[CheckResult]:
    results = []
    for repo in manifest.get("baseline_repos", []):
        name = f"baseline repo: {repo['name']} (§3.6)"
        p = REPO_ROOT / "third_party" / repo["name"]
        ok = (p / ".git").exists()
        results.append(CheckResult(name, ok, str(p) if ok else f"not cloned to {p}"))
    return results


def run_all_checks() -> list[CheckResult]:
    manifest = load_manifest()
    results = [check_hf_token()]
    results.extend(check_datasets(manifest))
    results.extend(check_models(manifest))
    results.append(check_freebase_endpoint(manifest))
    results.append(check_lockfile_current())
    results.extend(check_baseline_repos(manifest))
    return results
