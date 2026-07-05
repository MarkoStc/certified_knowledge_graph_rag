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
- P0 code complete: config loader / seeding / RunLogger (tested),
  `check_prereqs.py` verifying every §3 item against `context/manifest.yaml`,
  wired as pytest (marker `prereqs`, deselected by default). 14 unit tests
  green, ruff clean.

## 2026-07-06 — P1 — data & model acquisition

- Downloaded to `$SCRATCH/data/` with SHA256s recorded in
  `context/manifest.yaml` (details in `context/SOURCES.md`):
  2WikiMultiHopQA (`data_ids_april7.zip`, evidence-annotated version),
  MetaQA (HF mirror; official GDrive is gdown-inaccessible — content
  verified), HotpotQA (HF parquet; curtis.ml.cmu.edu was down), MuSiQue
  v1.0 (official GDrive), WebQSP (Microsoft, current URL — old one 404s),
  CWQ v1.1 (official Dropbox).
- Downloaded to `$SCRATCH/models/`: Qwen2.5-14B-Instruct (28G),
  Qwen2.5-7B-Instruct (15G), DeBERTa-v3-large-MNLI (3.9G). Llama-3.1-8B
  blocked on license+token.
- Cloned baselines to `third_party/` (commits in SOURCES.md): RoG,
  SubgraphRAG, Microsoft GraphRAG.
- `check_prereqs`: **13/16 satisfied; the 3 misses are exactly the
  [HUMAN-REQUIRED] items** (HF token, Llama license, Freebase endpoint).
- Next: unpack archives, dataset loaders with uniform schema + hand-checked
  tests, `results/data_stats.md`.
