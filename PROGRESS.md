# PROGRESS.md — append-only run log

Format per entry: date, phase, what was attempted, result, next.

---

## 2026-07-05 — pre-P0 — environment baseline

- Ran `/python-setup-demo`: uv env (Python 3.12.13), lockfile, ruff, git
  history, README/CLAUDE.md. Seeded the P4 deletion-certificate core
  (`src/mcgr/certify/menger.py`) with the brief's hand-checked toy tests.
- Result: `ruff check` clean; 5/5 tests green; `mcgr-demo` reproduces
  hand-computed k on all three toy graphs. Committed (3 commits).
- Next: P0 scaffold.

## 2026-07-06 — P0 — scaffold & environment sanity (started)

- Verified cluster facts: `$SCRATCH=/iopsstor/scratch/cscs/marko_stojanovic`
  (Lustre, space available), Slurm partitions `debug`/`normal`/`low` with
  gpu:4 per node, `xfer` partition present. Matches AGENTS.md §2.
- Created §4 directory skeleton; started config/seed/logging plumbing and
  `check_prereqs.py`.
- Blocked items known in advance ([HUMAN-REQUIRED], §3): HF token, Llama-3.1
  license, Freebase endpoint. Proceeding on everything else; escalation to
  Marko queued.
