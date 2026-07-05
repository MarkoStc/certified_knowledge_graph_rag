# CLAUDE.md

## Read AGENTS.md first

`AGENTS.md` at the repo root is the **authoritative operating brief**: north
star, rules of engagement, prerequisites, phases P0–P12, experiment matrix,
metric definitions, definition of done, escalation triggers. Nothing here
overrides it. This file only adds environment/tooling context the brief
doesn't cover.

## What exists so far

The `/python-setup-demo` baseline the brief's P0 assumes (§2: lockfile, git
init, dummy tests) — plus the P4 toy-certificate seed:

```
pyproject.toml               # metadata, deps, ruff + pytest config (single source of truth)
uv.lock                      # committed; regenerate with `uv sync` after dep changes
src/mcgr/
  certify/menger.py          # deletion certificate k = #edge-disjoint paths − 1 (NetworkX)
  cli.py                     # `mcgr-demo` smoke test (toy graphs, hand-checked k)
tests/test_certify_menger.py # the brief's P4 hand-checkable toy cases
```

The rest of the §4 target layout (`configs/`, `slurm/`, `src/mcgr/{data,kg,
retrieval,attacks,nli,eval,models}`, `context/`, `PROGRESS.md`,
`DECISIONS.md`) is **not created yet** — P0 builds it.

## Environment & tooling

- **uv** manages everything: interpreter, venv, deps, running. Never
  pip-install into the venv directly.
- Python **3.12** (pinned in `.python-version`). The login node's system
  `python3` is **3.6** — never use it; always go through `uv run`.
- This is a **CSCS Clariden login node** (aarch64, Slurm). Heavy work goes
  through `srun`/`sbatch` per AGENTS.md §2; don't run compute here.
- Runtime deps: only `networkx` so far. Add deps (torch, transformers,
  wandb, …) when the code needing them lands, and commit the updated
  `uv.lock` in the same change.

## Command cheat-sheet

```sh
uv sync                    # create/update .venv from lockfile
uv run pytest              # tests (certificate toy cases are the critical ones)
uv run ruff format .       # format
uv run ruff check .        # lint (--fix to auto-fix)
uv run mcgr-demo           # end-to-end smoke test
```

## Conventions & constraints

- Ruff: line length 100, rules `E W F I UP B SIM RUF`; `third_party/` is
  excluded (cloned baseline repos are not ours to lint). Keep the tree clean
  under `ruff format` + `ruff check`; use scoped `# noqa` with a reason, never
  global rule disables.
- Certificate semantics (from AGENTS.md §7, implemented in
  `mcgr/certify/menger.py`): `k = edge-disjoint paths − 1`; `k = 0` is a
  single fragile chain; `k = −1` (our convention) means no path support.
- No CI, Docker, or pre-commit; the brief drives verification through pytest
  + `check_prereqs.py` (P0) instead. Ask Marko before adding infra.

## Gotchas

- uv warns `Failed to hardlink files; falling back to full copy` — cache and
  repo are on different filesystems. Harmless; `export UV_LINK_MODE=copy`
  silences it. On compute nodes set `UV_CACHE_DIR=$SCRATCH/.uv-cache`
  (AGENTS.md §2) so installs don't hit the home quota.
- The GitHub remote (`MarkoStc/certified_knowledge_graph_rag`) started empty;
  history begins with the local bootstrap commits. Nothing has been pushed
  yet. Note the remote name differs from the brief's `menger-certified-
  graphrag` layout root — the layout's *contents* are what matters.
- Repo-local git identity is set to Marko's GMail; the machine's global
  default derives a bogus `@clariden-ln003.cscs.ch` address — don't commit
  with it.
