# Menger-Certified Graph-RAG (mcgr)

Provable per-answer robustness for multi-hop knowledge-graph reasoning via
path redundancy: each answer gets a certified budget `k` — the number of
edge-disjoint supporting paths minus one — so no adversary deleting ≤ `k`
triples can destroy all support (Menger's theorem / max-flow–min-cut).

**`AGENTS.md` is the authoritative project brief** (goals, phases P0–P12,
experiment matrix, definition of done). This README covers setup and the dev
workflow only.

Current state: environment baseline + the deletion-certificate core
(`src/mcgr/certify/`) with the hand-checked toy-graph tests from the brief.
Phases P0+ build on this.

## Setup

Requires [uv](https://docs.astral.sh/uv/). One command builds the virtualenv
(Python 3.12) and installs everything from the lockfile:

```sh
uv sync
```

On CSCS compute nodes, keep caches off the home quota (AGENTS.md §2):

```sh
export UV_CACHE_DIR=$SCRATCH/.uv-cache
uv sync --frozen
```

## Development workflow

```sh
uv run pytest              # test suite (incl. hand-checked certificate cases)
uv run ruff format .       # format
uv run ruff check .        # lint (add --fix to auto-fix)
uv run mcgr-demo           # smoke test: certificates on toy graphs
```

Lint/format/test settings live in `pyproject.toml`. `uv.lock` is committed;
after changing dependencies, run `uv sync` and commit the updated lockfile.
