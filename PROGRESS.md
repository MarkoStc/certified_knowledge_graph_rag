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
- Unpacked all archives; wrote 8 loaders (2Wiki, MetaQA 1/2/3-hop, MuSiQue,
  WebQSP, CWQ, HotpotQA) emitting a uniform QARecord. Hermetic parsing tests
  (19) + hand-verified real-data tests (9, `pytest -m data`) all green.
  `results/data_stats.md` regenerated; counts match published sizes.
- Honesty notes: only 2Wiki has native triple gold paths; MuSiQue structured
  hops ~66% (raw decomposition preserved in meta); WebQSP/CWQ gold+anchors
  await Freebase (P2). Fixed a .gitignore bug (`data/` was shadowing
  `src/mcgr/data`); root-anchored the large-artifact rules.
- **P0+P1 complete. check_prereqs 13/16 — the 3 misses are exactly the
  [HUMAN-REQUIRED] items.** Escalation prepared for Marko.
- Next: P2 KG construction, starting with MetaQA's self-contained KB (no
  Freebase, no gated model) to reach a real P4 certificate distribution —
  the brief's "lead with 2Wiki + MetaQA" fallback path.

## 2026-07-06 — P2/P4 — MetaQA KG + first real certificate (started)

- Rationale: MetaQA's kb.txt is a complete movie KB, so we can build the
  reasoning graph and compute edge-disjoint-path certificates on real data
  with zero dependency on the blocked items. This is the cleanest route to
  P7 go/no-go evidence and the hop-count-vs-path-count ablation (§12).
- Built: `kg/graph_store.py` (undirected fact graph + k-hop subgraph),
  `kg/metaqa_kg.py` (cached full KB graph: 43,234 nodes / 124,680 edges from
  134,741 triples), `certify/menger.py` extended with anchor-set certificate
  (super-source max-flow), `certify/pipeline.py`, `scripts/compute_certificates.py`,
  `slurm/certify_metaqa.sbatch`. 27 hermetic tests green (+ data-marked).
- **First real certificate result** (validated end-to-end via `srun` on a
  compute node, partition debug): MetaQA 2-hop dev, 300 queries →
  frac(k≥1)=28.7%, all supported, mean k=0.64. Login-node probes:
  1-hop 0%, 2-hop 37%, 3-hop 98% certifiable (60 q each). The monotonic
  hop-count → path-redundancy trend is the P7-style signal (preliminary).
  Recorded in `results/certificate_distribution.md`.
- **Blocker for the full sweep:** 3-hop is ~4.7 s/query single-threaded
  (~18 h/split); pipeline must be parallelized across queries before the
  Slurm array is practical. 1/2-hop are cheap. Next engineering step.
- Account for Slurm is `infra01`; nodes are 288-core + gpu:4 (GH200);
  certificate work is CPU-bound so no GPU needed for P4.
