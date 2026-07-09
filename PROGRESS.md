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

## 2026-07-06 — P4 — full MetaQA certificate sweep COMPLETE

- Parallelized the pipeline (multiprocessing Pool; workers build the KB
  graph once, never pickled) and switched the hot path to
  `local_edge_connectivity` (max-flow value, ~2.6x faster than
  materializing paths — same Menger count). Confirmed the user's point that
  the [HUMAN-REQUIRED] items don't block the critical path; resumed the
  sweep instead of waiting on them.
- Ran `slurm/certify_metaqa.sbatch` (array 2692989, 6 cells = {1,2,3}-hop x
  {dev,test}, 64 CPUs/cell, partition normal). All cells completed; 3-hop
  cells took ~30 min each (dense subgraphs), 1/2-hop minutes.
- **Full-split result** (results/certificate_distribution.md, generated by
  scripts/make_certificate_report.py):
    1-hop:  0.5% k>=1  (n~10k)   mean k 0.01
    2-hop: 31.3/31.4% k>=1 (n=14,872)  mean k 0.73
    3-hop: 96.9/96.7% k>=1 (n=14,274)  mean k ~6.4
  Monotonic hop-count -> path-redundancy, tight dev/test agreement. This is
  the P7 H1 signal on full data.
- Next unblocked critical-path step: 2WikiMultiHopQA KG (P2) using its native
  gold triples + Wikidata/context neighborhood, then certify it. After that,
  P6 attacks + P7 correlation (certified vs empirical robustness).

## 2026-07-07 — P2/P4 — 2WikiMultiHopQA certified (Wikidata-grounded)

- Confirmed with Marko that the [HUMAN-REQUIRED] items don't block the
  critical path; proceeded to 2Wiki.
- Built a Wikidata fetcher (`kg/wikidata.py`, batched/cached/rate-safe) and
  fetched all entity-valued claims for the 60,487 entities in 2Wiki
  compositional/inference gold chains → **1,296,716 triples** cached to
  `$SCRATCH/data/2wiki_kg` (snapshot 2026-07-07, `context/SOURCES.md`).
  Raw graph 366,039 nodes / 1,218,282 edges.
- Hub pruning (evidential-independence control) + role-aware endpoint
  reconnection: block transit through type/attribute hubs (human, male, a
  country) but keep them when they are a query's own answer. This fixed a
  real bug where 25% of questions (common-entity answers) were wrongly
  unsupported — supported 72% → 96%.
- **2Wiki certificate result** (dev 6,107 / train 49,772 compositional+
  inference queries, added to results/certificate_distribution.md):
    dev:   frac(k≥1)=0.675  supported=0.964  mean k=4.01
    train: frac(k≥1)=0.639  supported=0.957  mean k=4.10
  Tight train/dev agreement. Strong H1 support on a second, real,
  Wikidata-grounded dataset (not just MetaQA's synthetic KB).
- Next: P6 attacks (RAG-Safety triple insertion first) + P7 gate — correlate
  certified k with empirical robustness under attack. Also 2Wiki test lacks
  answer_id so isn't certifiable; train is cached for P8 retriever work.

## 2026-07-09 — P6 + P7 — attack, reasoner, and the GO gate

- Retired the top technical risk: Qwen2.5-7B runs on GH200 (torch
  2.13+cu130, aarch64 CUDA). Fixed a batched-decode left-padding bug; smoke
  test answers correctly. Added optional `[llm]` extra (torch/transformers);
  required-environments pins aarch64 so torch resolves the sbsa CUDA wheel.
  NOTE: use `.venv/bin/python` or `uv run --extra llm` consistently — mixing
  base and `--extra llm` `uv run` calls thrashes the env (slow).
- Built P6: RAG-Safety insertion attack (attacks/rag_safety.py), KG reasoner
  (models/reasoner.py), directed-evidence retrieval (retrieval/evidence.py).
- P7 experiment iteration (honest): first run had too weak an attack (fake
  ids ignored, even k=0 flipped ~6%). Fixed to a faithful attack —
  type-consistent sibling wrong-answers, real intermediates, shuffled context.
- **P7 GATE = GO.** MetaQA 2-hop, Qwen2.5-7B (results/stage1_gate.md):
    budget 8: k=0 flip 0.37 vs k>=1 ~0.07-0.10 (>4x, monotonic)
    budget 2: k=0 flip 0.12 vs k>=1 0.02-0.06
  Clean acc 0.85-0.96 across bins. Certificate predicts robustness; gap
  scales with attack strength. Decision recorded in DECISIONS.md.
- Next (GO path): broaden P7 (2Wiki obscure entities + a second reasoner,
  full budget sweep, seeds) for P9; then P8 certificate-maximizing retriever;
  P5 insertion certificate; P10 selective prediction.

## 2026-07-09 (cont.) — P9 seed — P7 effect confirmed robust

- Hardened the P7 GO against the memorization caveat with two controls, both
  on GPU (results/stage1_gate.md, k=0 vs certified k>=1 flip, budget 8):
    Qwen2.5-7B            0.37 vs 0.08  (4.5x)
    Qwen2.5-14B           0.19 vs 0.03  (7.5x)  <- cross-model, effect holds
    Qwen2.5-7B anonymized 0.42 vs 0.07  (5.6x)  <- no memorization possible
- Anonymization (opaque entity tokens) makes the gradient *steeper* and drops
  clean acc most at k=0 (0.85->0.52): memorization was masking, not creating,
  the effect. The certificate dependence is graph-structural. Caveat resolved.
- Pushed nothing to remote yet: this machine has no GitHub credentials
  (no gh/SSH/token); Marko to push or supply a token.
- Next: 2Wiki obscure-entity P7 (needs Wikidata labels for readable triples);
  multi-seed for mean±std; then P8 retriever.
