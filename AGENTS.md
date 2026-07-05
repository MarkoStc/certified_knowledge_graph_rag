# AGENTS.md — Menger-Certified Graph-RAG

**Project:** Menger-Certified Graph-RAG: Provable Per-Answer Robustness for Multi-Hop Knowledge-Graph Reasoning via Path Redundancy
**Role of this file:** This is the authoritative operating brief for the autonomous coding agent executing this project. Read this file top to bottom before doing anything. Everything the project requires is specified here or pointed to from here. Do not improvise scope.
**Human owner:** Marko (contact for all escalations).
**Compute:** CSCS Clariden vCluster, multi-node NVIDIA GH200 (96 GB HBM3/GPU, 4 GPUs/node, Grace+Hopper unified memory), Slurm-managed.

---

## 0. NORTH STAR — THE SINGLE TARGET

> **Produce the complete, reproducible empirical contribution of Menger-Certified Graph-RAG: a working system plus the full body of experimental evidence that would constitute the entire results of a top-venue paper — every claim below established by a real, re-runnable artifact on real benchmarks and real attacks. No manuscript is written (that comes later), but nothing scientific is deferred, stubbed, or demo-scaled. Do not stop, do not declare the project finished, and do not idle until every gate in the Definition of Done (§8) is green.**

This is the ultimate target, not a milestone. "Done" means the science is finished and airtight — the kind of result set a strong reviewer at NeurIPS/ICML/ICLR could not dismiss as preliminary — sitting in the repo as tables, figures, logs, and code. The only remaining step afterward would be to write English around it.

**Explicitly rejected as targets:** a demo, a proof-of-concept, a toy-scale run, "the pipeline executes," a subset of datasets "to show it works," or any placeholder result. Every experiment in the matrix (§6) is run at full scale, on all specified datasets, models, attacks, budgets, and seeds, with statistical reporting. If a result is missing, the project is not done.

The four contributions that must be **fully empirically established** (this is the entire evidentiary basis of the future paper):
1. A **per-answer certified robustness budget** `k` for multi-hop KG reasoning via edge-disjoint path counting / min-cut (Menger). Deletion threat model, proven, and computed across all datasets.
2. An **insertion-aware certificate** extending beyond textbook Menger to adversarial insertion of competing paths (KEPo-style), with a conservative, explicitly stated threat model, implemented and evaluated.
3. A **certificate-maximizing agentic retrieval policy** that actively finds independent supporting paths, demonstrated to raise mean certified `k` at modest clean-accuracy cost, with the full ablation against single-best-chain retrieval.
4. A **benchmark + metric suite** (Certified Accuracy@k, certified coverage, certified-vs-empirical curves) and **certified selective prediction** (provably-grounded abstention when `k=0`), computed and compared against confidence/self-consistency baselines.

"Done" is defined operationally in §8. Until then, the standing instruction is: pick the next unchecked task in the highest-priority incomplete phase, do it at full scale, verify it empirically, commit it, log it, and continue.

---

## 1. RULES OF ENGAGEMENT (how you operate)

- **Autonomy first.** Proceed without asking for confirmation on anything specified in this document. Only stop to ask Marko when you hit an explicit escalation trigger (§10). If a decision is genuinely underspecified and not covered here, make the most defensible choice, record it in `DECISIONS.md` with your reasoning, and continue.
- **Empirical honesty is absolute.** Never fabricate, extrapolate, or "expect" a result. A number appears in the paper only if it was produced by a script in this repo that can be re-run. If an experiment failed or was not run, say so explicitly. A wrong-but-real result is acceptable; a fabricated result is a project-ending failure.
- **Everything is reproducible.** Every experiment is driven by a versioned config (`configs/*.yaml`), a fixed seed, and a logged git commit hash. No magic numbers in code. No manual one-off runs that can't be reproduced from a config.
- **Commit continuously.** Small, atomic commits with clear messages. Commit after every task reaches its acceptance criterion. Never leave the repo in a broken state overnight.
- **Test before you trust.** Any non-trivial function (graph construction, max-flow, certificate computation, attack injection, metric calculation) gets a unit test with a hand-checkable case before it is used to produce paper numbers. The certificate math especially: build a tiny toy graph by hand, compute `k` by hand, assert the code agrees.
- **Log a running narrative.** Maintain `PROGRESS.md` as an append-only log: date, phase, what was attempted, what the result was, what's next. This is how Marko audits the run without reading every commit.
- **Checkpoint expensive state.** Cache built KGs, retrieved paths, and model outputs to `$SCRATCH` so a crashed job never forces a full recompute. Make every stage resumable.
- **Respect the gates.** The go/no-go gate in P7 is real. If Stage-1 de-risking fails its threshold, execute the pivot in §9 — do not push forward on a dead hypothesis. "Don't stop" means "don't stop until the paper exists," not "ignore the evidence."
- **Network is available on compute nodes.** Compute nodes have outbound internet, so downloading datasets/models, resolving dependencies, and syncing logs may all happen at run time on the GPU node. Still cache aggressively to `$SCRATCH` (§2) so a crash never forces a re-download and so runs are reproducible — but you are not forced to pre-stage everything.

---

## 2. ENVIRONMENT: CSCS CLARIDEN / GH200

> Verify every cluster-specific detail below against the **current CSCS user documentation** before relying on it; operational specifics change. The design here is deliberately robust to the strictest assumption (offline compute nodes) so it works regardless.

**Node types.**
- **Login / access nodes:** for lightweight interactive work, editing, job submission, and quick checks. Do not run heavy compute here.
- **Compute nodes (GH200):** have outbound internet. All training, inference, and evaluation run here via Slurm, and may download datasets/models and install dependencies at run time. Cache everything to `$SCRATCH` so repeated jobs don't re-fetch.

**Storage.**
- `$SCRATCH` — large, fast, **periodically purged**. Stage datasets, model weights, caches, and experiment outputs here. Treat as ephemeral; anything precious gets copied to project/home.
- `$HOME` / project store — persistent, smaller. Keep the git repo, configs, and final artifacts here.
- Set `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `UV_CACHE_DIR`, and `WANDB_DIR` to paths under `$SCRATCH` so nothing lands in a size-limited home quota.

**Python env (uv).**
- Dependencies are locked with `uv`. Compute nodes can `uv sync` directly. Keep the venv on `$SCRATCH` and set `UV_CACHE_DIR` under `$SCRATCH` so installs are cached across jobs.
- `/python-setup-demo` (Marko's command) handles `uv lock`, `git init`, and dummy tests. P0 assumes it has been run and verifies the result.

**Job submission.**
- Interactive smoke tests: `srun --pty ...` for a few minutes on one GPU.
- Real runs: `sbatch` scripts in `slurm/`. One script per experiment class. Parameterize by config path. Log stdout/stderr to `logs/slurm/%x_%j.out`.
- The workload is embarrassingly parallel across (dataset × model × attack × budget × seed). Fan these out as independent Slurm jobs / a job array; do not serialize them.

**Experiment tracking.**
- Use `wandb` online directly from compute nodes (set `WANDB_DIR` under `$SCRATCH`). Keep a structured CSV + TensorBoard log under `$SCRATCH/runs/` as a local backup regardless.

---

## 3. PREREQUISITES

Compute nodes have internet, so **the agent can download datasets, models, and dependencies itself** as part of P0/P1 — these are not human-blocking. Only two things genuinely require a human and must be done by Marko first; they are called out as **[HUMAN-REQUIRED]** below. The agent must **verify** every item (human-required or not) is satisfied in P0 via `check_prereqs.py`, and must escalate only if a **[HUMAN-REQUIRED]** item is missing. For agent-downloadable items, if something is absent the agent simply fetches it rather than escalating.

### 3.1 Secrets and tokens — [HUMAN-REQUIRED]
- [ ] **Hugging Face token** (needed for the gated Llama model in §3.4). Create at huggingface.co → Settings → Access Tokens (read scope). Persist it where the agent can read it but git cannot commit it:
  ```bash
  export HF_TOKEN=hf_xxx                       # put in ~/.config/hf/token, NOT in the repo
  echo 'HF_TOKEN=hf_xxx' >> .env               # .env is in .gitignore
  ```
- [ ] **Git identity / remote** configured so the agent can push (or confirm the agent works against a local repo only).
- [ ] (Optional) **wandb API key** for online tracking.

### 3.2 Datasets — agent downloads to `$SCRATCH/data/` (P1); no human needed
Primary (lead with this one):
- [ ] **2WikiMultiHopQA** — provides explicit gold reasoning paths at triple level; this is the backbone dataset. Source: the official 2Wiki release (Alab-NII "2wikimultihop" repo / mirror) or the Hugging Face copy. Download train/dev, including the evidence/path annotations, not just Q–A pairs.

Standard KG-RAG attack surface:
- [ ] **WebQSP** — download the Microsoft WebQSP release (question–SPARQL–answer). Needed to reproduce the RAG-Safety attack setting and the RoG baseline.
- [ ] **ComplexWebQuestions (CWQ)** — download from the official CWQ site / mirror. Genuine compositional multi-hop.

Controlled ablations + generalization:
- [ ] **MetaQA** — 1/2/3-hop movie-domain KG splits (clean, for hop-count vs path-count ablations).
- [ ] **HotpotQA** — multi-hop text QA (for the text-constructed-KG generalization).
- [ ] **MuSiQue** — harder multi-hop text QA (StonyBrookNLP release).

> Record exact download URLs, versions/commit hashes, and SHA256 of each archive in `context/SOURCES.md`. The agent will assert these files exist by checksum.

### 3.3 Freebase for WebQSP / CWQ — the heavy one — [HUMAN-REQUIRED to confirm the endpoint]
WebQSP and CWQ are grounded in **Freebase**, which is not a simple file download — it needs a running SPARQL endpoint. The agent can attempt the download and Virtuoso setup, but standing up and confirming a reachable endpoint is the single biggest setup burden and is the most likely thing to need a human. Marko should confirm a working endpoint (or hand the agent a reachable endpoint URL) early. Start this first.
- [ ] Obtain a preprocessed **Freebase subset dump** suitable for WebQSP/CWQ (the commonly used "Freebase-Setup" style Virtuoso dump used by GraftNet / RoG-family work). Find the current canonical source and record it in `context/SOURCES.md`.
- [ ] Stand up a **Virtuoso** SPARQL endpoint loaded with that dump (follow the standard Freebase-Setup instructions). Confirm a test SPARQL query returns results.
- [ ] Record the endpoint URL/port so the agent can query it from compute nodes (verify network reachability from a compute node — if the endpoint must live on a reachable host, arrange that).
- If standing up Freebase proves infeasible in time, that is an **escalation** (§10): the fallback is to lead entirely with 2WikiMultiHopQA + MetaQA (self-contained KGs) and treat WebQSP/CWQ as secondary, reporting the RAG-Safety comparison on whatever subset is available.

### 3.4 Models — agent downloads to `$SCRATCH/models/` (P0 smoke test); one gate is human
- [ ] **Qwen2.5-14B-Instruct** and **Qwen2.5-7B-Instruct** (open weights, no gate) — reasoner/generator. Agent runs `huggingface-cli download Qwen/Qwen2.5-14B-Instruct`.
- [ ] **Llama-3.1-8B-Instruct** — **GATED, [HUMAN-REQUIRED]**: Marko must accept the license on its Hugging Face model page once; after that the agent downloads it with `HF_TOKEN`. Used as a second reasoner for cross-model robustness of results. (If the license isn't accepted, the agent proceeds with the Qwen models and flags Llama as pending.)
- [ ] **An NLI / entailment model** for edge/claim consistency checks — e.g. a DeBERTa-v3-large MNLI checkpoint. Agent downloads.
- [ ] Agent verifies each available model loads and generates on one GH200 in a 5-minute `srun` smoke test before building anything on top.

### 3.5 Dependency environment — agent handles; no human needed
- [ ] `uv lock` (already done by `/python-setup-demo`; agent re-verifies the lockfile is current).
- [ ] Agent runs `uv sync --frozen` directly on a compute node (they're online). Set `UV_CACHE_DIR=$SCRATCH/.uv-cache` so installs are cached across jobs and runs stay reproducible.

### 3.6 Baseline repos — agent clones into `third_party/`; no human needed
- [ ] Agent clones the official implementations to compare against and adapt: **RoG (Reasoning-on-Graphs)**, **SubgraphRAG**, and a **vanilla GraphRAG** reference (e.g. the Microsoft GraphRAG library). Record commit hashes in `context/SOURCES.md`. Do not vendor blindly — read each README; these are called as baselines.
- [ ] Agent locates the official attack implementations (**RAG Safety**, **GraphRAG-under-Fire / GragPoison**, **KEPo**) if released and clones them. If code is unavailable for one, note it — reimplement the attack from the paper's description (P6).

**Agent gate:** P0 does not pass until every §3 item is verified present. For agent-downloadable items (§3.2, §3.4 non-gated, §3.5, §3.6), the agent fetches whatever is missing rather than escalating. Escalate to Marko only for a missing **[HUMAN-REQUIRED]** item (§3.1 token, §3.3 Freebase endpoint, §3.4 Llama license), with the exact missing item and how to obtain it.

---

## 4. TARGET REPOSITORY LAYOUT (end state)

```
menger-certified-graphrag/
├── AGENTS.md                  # this file
├── README.md                  # human-facing overview, how to reproduce
├── PROGRESS.md                # append-only run log (agent maintains)
├── DECISIONS.md               # non-obvious choices + rationale (agent maintains)
├── pyproject.toml / uv.lock   # locked env
├── .env / .gitignore          # secrets out of git
├── configs/                   # one yaml per experiment; no hardcoded params
├── src/mcgr/
│   ├── data/                  # loaders, cleaning, dataset-quality filtering
│   ├── kg/                    # KG construction / ingestion / SPARQL access
│   ├── retrieval/             # baseline retrievers + agentic path-diversifying policy
│   ├── certify/               # max-flow / edge-disjoint paths / certificate (deletion + insertion)
│   ├── attacks/               # RAG-Safety, GragPoison, KEPo adapters/reimpls
│   ├── nli/                   # edge/claim consistency checker
│   ├── eval/                  # metrics: Certified Accuracy@k, coverage, curves, selective prediction
│   └── models/                # reasoner/generator wrappers, LoRA training
├── tests/                     # unit tests (esp. certificate math on toy graphs)
├── slurm/                     # sbatch scripts, job arrays
├── scripts/                   # end-to-end pipeline entrypoints
├── results/                   # generated tables/figures (from scripts, reproducible)
│   └── FINAL/                 # consolidated evidence pack: tables, figures, RESULTS.md, claim↔evidence map
├── threat_model.md            # formal, conservative threat model (P5) — spec, not paper prose
└── context/                   # the dossier: SOURCES.md + papers/ + the research brief
    ├── SOURCES.md
    └── papers/                # downloaded PDFs of every cited work
```

---

## 5. WORK BREAKDOWN — PHASES P0–P12

Each phase has a **Goal**, concrete **Steps**, **Deliverables**, and an **Acceptance gate**. Do phases roughly in order; P4/P5 (certificate) and P8 (retriever) are the scientific core. Parallelize within a phase wherever the compute allows.

### P0 — Scaffold & environment sanity
- **Goal:** A clean, tested, offline-capable skeleton with all prerequisites verified.
- **Steps:**
  - Confirm `/python-setup-demo` ran: lockfile present, git initialized, dummy tests green.
  - Create the §4 directory structure; add `.gitignore` (`.env`, `$SCRATCH` paths, model weights, data).
  - Implement `scripts/check_prereqs.py`: asserts every dataset archive (by SHA256), every model dir, the Freebase endpoint reachability, and `uv sync --offline` all succeed. Wire it as a pytest.
  - Set up config system (yaml + a small loader), seeding utility, and a structured logger.
  - Set up offline wandb / CSV logging.
- **Deliverables:** green `check_prereqs`, passing CI-style `pytest`, `README` skeleton.
- **Gate:** `pytest` green AND `check_prereqs` reports all §3 items present. Do not proceed otherwise — escalate the specific gap.

### P1 — Data acquisition & cleaning
- **Goal:** Every dataset loaded into a common internal schema with quality filtering applied.
- **Steps:**
  - Loaders in `src/mcgr/data/` returning a uniform record: `{question, answers, anchor_entities, gold_paths (if any), source_dataset}`.
  - **Apply the dataset-quality caveat:** WebQSP/CWQ have known factual-correctness issues in a large fraction of sampled answers. Implement a cleaned-subset filter and report on cleaned subsets; **lead with 2WikiMultiHopQA's triple-level gold paths**.
  - For MetaQA, expose the 1/2/3-hop split cleanly for controlled ablations.
  - Deterministic train/dev/test splits with fixed seeds; cache parsed data to `$SCRATCH`.
- **Deliverables:** `data/` loaders + tests on a few hand-checked records per dataset; a short `results/data_stats.md` (counts, hop distributions, cleaned-subset sizes).
- **Gate:** Each loader passes a unit test on hand-verified examples; stats table generated.

### P2 — KG construction / ingestion
- **Goal:** For every query, a well-defined graph over which certificates are computed.
- **Steps:**
  - For KG-native datasets (2Wiki via Wikidata-grounded triples, MetaQA, WebQSP/CWQ via Freebase/Virtuoso): ingest the relevant subgraph around each query's anchor entities (bounded k-hop neighborhood).
  - For text datasets (HotpotQA, MuSiQue): construct a KG from the supporting text — extract entities/relations, map supporting sentences to edges. Record the extraction method; this doubles as the "generalizes to text-constructed KG" result in P11.
  - Represent each query's reasoning space as a directed/undirected multigraph with edges = triples; store as a serializable object keyed by query id in `$SCRATCH`.
- **Deliverables:** `kg/` module + tests; per-dataset cached query subgraphs; `results/kg_stats.md` (avg nodes/edges per query subgraph, connectivity between anchors and gold answers).
- **Gate:** For 2Wiki, gold reasoning paths are recoverable as actual paths in the constructed graph for a hand-checked sample.

### P3 — Baselines
- **Goal:** Reproduce the systems you claim to improve on / attack, at reported ballpark.
- **Steps:**
  - Run **RoG**, **SubgraphRAG**, and **vanilla GraphRAG** on WebQSP (+2Wiki where applicable) with your reasoner models. Report clean EM/F1.
  - Sanity-check against published numbers (they need not match exactly, but must be in the right regime). Document any deviation in `DECISIONS.md`.
- **Deliverables:** `results/baselines.md` with clean accuracy per (system × dataset × model).
- **Gate:** Baseline clean accuracy is within a defensible range of published results; discrepancies explained.

### P4 — Certificate engine (deletion threat model) — CORE
- **Goal:** The deterministic per-answer certificate `k` via edge-disjoint paths / min-cut.
- **Steps:**
  - Implement edge-disjoint path counting and min-cut between anchor set and candidate answer node (NetworkX `maximum_flow` / edge-disjoint paths; switch to igraph if needed for speed on larger subgraphs). Unit-capacity edges → max-flow = number of edge-disjoint paths = min-cut (Menger).
  - Define the certificate precisely: `k(q) = (#edge-disjoint anchor→predicted-answer paths) − 1`. Deletion certificate: no adversary removing `≤ k` triples can destroy all support.
  - **Test hard:** hand-build toy graphs (1 path, 2 disjoint paths, 2 paths sharing one edge), compute `k` by hand, assert the engine matches. This is the most important test in the repo.
  - Compute `k` distribution over each dataset. **This produces the P7 go/no-go evidence.**
- **Deliverables:** `certify/` module + toy-graph tests; `results/certificate_distribution.md` (histogram of `k` per dataset; fraction of queries with `k≥1`).
- **Gate:** Toy-graph tests pass exactly; `k` computed for all queries on ≥2 datasets.

### P5 — Threat model formalization + insertion-aware certificate — CORE / hardest
- **Goal:** Extend beyond deletion-only Menger to adversarial **insertion** of competing paths.
- **Steps:**
  - Write the threat model formally and **conservatively**: adversary may insert ≤ b triples; state exactly what they can connect (e.g. only via existing entities), and any detectability/boundedness assumptions. Overclaiming here is the most likely reviewer attack — be explicit about what is and isn't covered.
  - Define the insertion-aware certificate: certify predicted answer `a` only when its supporting connectivity provably dominates any competing connectivity to a wrong answer `a'` achievable within budget `b`. Prove the bound under the stated assumptions.
  - Implement the computation; test on toy graphs where you can enumerate adversary insertions by hand.
  - Clearly separate **structural** edge-disjointness from **evidential** independence (two edge-disjoint paths can still share a corrupted source); discuss and, where possible, measure the gap.
- **Deliverables:** `certify/insertion.py` + tests; a formal `threat_model.md` (specification of the threat model and the proven bound — not paper prose).
- **Gate:** Insertion certificate matches hand-computed toy cases; threat model written and internally reviewed for overclaiming.

### P6 — Attack integration
- **Goal:** Real, citable adversaries wired in to stress-test both certificates.
- **Steps:**
  - Integrate/reimplement **RAG-Safety** triple insertion (cleanest edge-budget match), **GragPoison** (shared-relation injection), and **KEPo** (connected knowledge-evolution paths). Reuse released code where available; reimplement from the paper otherwise and note it.
  - Parameterize each attack by injected-triple budget; make attacks deterministic under seed.
  - Reproduce a known headline effect as a sanity check (e.g. RoG EM on WebQSP collapsing under triple insertion) to confirm the attack is faithful.
- **Deliverables:** `attacks/` adapters + tests; `results/attack_sanity.md` showing an expected accuracy collapse under attack.
- **Gate:** At least one attack reproduces a published-magnitude accuracy drop on a baseline.

### P7 — STAGE-1 DE-RISK GATE (go / no-go) — decision point
- **Goal:** Prove the core hypothesis H1 before investing in P8+.
- **Steps:**
  - On 2WikiMultiHopQA (gold paths) + one more dataset: show certified `k` **correlates with empirical robustness** under RAG-Safety triple insertion (certified queries survive; uncertified `k=0` queries flip).
  - Measure the **fraction of multi-hop queries admitting a non-trivial certificate** (`k≥1`).
- **GO criterion:** ≥25–30% of multi-hop queries admit `k≥1` **AND** certified queries are empirically, markedly more robust than uncertified ones.
- **NO-GO:** if essentially all queries are `k=0` (single fragile chain), **execute the pivot in §9** — the certificate-maximizing retriever (P8) becomes the headline contribution ("making graphs certifiable") rather than a secondary result.
- **Deliverables:** `results/stage1_gate.md` with the correlation evidence and the `k≥1` fraction, and an explicit GO/PIVOT decision recorded in `DECISIONS.md`.
- **Gate:** A decision is made and written down. The project continues on the GO path or the PIVOT path — never ambiguously.

### P8 — Certificate-maximizing agentic retrieval policy (+ LoRA) — CORE
- **Goal:** An active retriever that finds **independent** supporting paths, raising mean certified `k`.
- **Steps:**
  - Baseline retriever: single highest-scoring chain (this is what you beat).
  - Agentic policy: iteratively seek additional **edge-disjoint** anchor→answer paths (e.g. penalize edges already used, search for alternative routes), stopping on a budget or diminishing returns.
  - Optionally **LoRA-fine-tune** the path-selection policy on a reasoner model to prefer diverse independent paths. LoRA (not full fine-tune) keeps this within one GH200's memory; train on the datasets' gold/derived paths.
  - Measure mean certified `k` and certified coverage: agentic policy vs single-best-chain, and the clean-accuracy cost.
- **Deliverables:** `retrieval/agentic.py`, optional `models/lora_train.py` + sbatch; `results/retriever_ablation.md`.
- **Gate:** Agentic policy **materially** raises mean certified `k` and/or certified coverage over single-best-chain at **modest** clean-accuracy cost (target: a clear margin, ≤ a few points EM drop). If not, iterate on the policy before proceeding.

### P9 — Full evaluation & sweeps
- **Goal:** The complete results grid backing the paper.
- **Steps:**
  - Run the full matrix (§6): datasets × reasoner models × attacks × budgets × seeds, for both baseline and certified/agentic systems.
  - Compute the metric suite (§7): Certified Accuracy@k, certified coverage, empirical robustness under each attack at matched budgets, and certified-vs-empirical robustness curves.
  - Fan out as a Slurm job array; aggregate into tables/figures via scripts (no manual assembly).
- **Deliverables:** `results/main_tables.md`, `results/figures/` (all generated by scripts), raw run logs on `$SCRATCH`.
- **Gate:** Every headline number in the paper traces to a re-runnable config + commit. Figures regenerate from `scripts/make_figures.py`.

### P10 — Certified selective prediction (H3)
- **Goal:** Show certified-path coverage is a better abstention signal than confidence/self-consistency.
- **Steps:**
  - Abstain when `k < τ`; sweep τ. Build risk–coverage curves.
  - Compare against LLM confidence and self-consistency as abstention signals (a model can be consistently wrong through a poisoned chain — that's the point).
  - Report AURC (area under risk–coverage) for each signal.
- **Deliverables:** `results/selective_prediction.md` + risk–coverage figures.
- **Gate:** The `k`-based signal beats confidence/self-consistency on AURC on at least the primary dataset, or the negative result is clearly characterized.

### P11 — Generalization (text-constructed KG) + optional multimodal teaser
- **Goal:** Show the method isn't tied to a hand-built KG.
- **Steps:**
  - Run the full pipeline on HotpotQA/MuSiQue via the P2 text-constructed KGs; report certified robustness there.
  - (Optional, only if time remains after P12 is on track) a small multimodal result: path redundancy across text+table (or text+image) evidence, as a forward-looking result establishing the follow-up direction.
- **Deliverables:** `results/generalization.md`.
- **Gate:** At least the text-constructed-KG generalization runs end-to-end and is reported.

### P12 — Ablations & final results consolidation (NO manuscript)
- **Goal:** Complete the remaining science and consolidate everything into a paper-ready evidence pack. Do **not** write the paper — produce the artifacts a paper would be written from.
- **Steps:**
  - **Ablations (science — required):** hop-count vs number-of-independent-paths (MetaQA is ideal); structural vs evidential independence; effect of the LoRA policy; sensitivity to attack budget; per-model results (Qwen-14B/7B, Llama-3.1-8B) to show robustness of findings across reasoners.
  - **Consolidate results** into `results/FINAL/`: every headline table as a CSV **and** a rendered table; every figure as a script-regenerated file; a single `results/FINAL/RESULTS.md` that states each finding as a one-line factual claim with a pointer to the exact table/figure and the config+commit that produced it. This is the evidentiary spine of the future paper — claims and numbers, not prose.
  - **Claim ↔ evidence map:** a table linking each of the four contributions (and each hypothesis H1–H3) to the specific results that establish it, so the future write-up is pure transcription.
  - **Reproduction:** `README` with one command per table/figure; confirm a cold re-run of the full pipeline reproduces `results/FINAL/` (within seed variance).
  - **Do NOT** produce LaTeX, a manuscript, an abstract, or written sections. If the North Star ever tempts you toward prose, stop — that is explicitly out of scope.
- **Deliverables:** `results/FINAL/` (all tables + figures + `RESULTS.md` + claim↔evidence map), full reproduction `README`.
- **Gate:** §8 Definition of Done fully green.

---

## 6. EXPERIMENT MATRIX

| Axis | Values |
|---|---|
| Datasets | 2WikiMultiHopQA (primary), MetaQA (1/2/3-hop ablation), WebQSP, CWQ, HotpotQA, MuSiQue |
| Reasoner models | Qwen2.5-14B-Instruct (primary), Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct |
| Systems | vanilla GraphRAG, RoG, SubgraphRAG (baselines); single-best-chain (retriever baseline); **certified + agentic** (ours) |
| Attacks | RAG-Safety triple insertion, GragPoison, KEPo; plus clean (no attack) |
| Budgets | injected-triple budget swept (e.g. 0,1,2,5,10,20 per query) |
| Seeds | ≥3 seeds for anything with variance; report mean ± std |
| Certificate | deletion (P4) and insertion-aware (P5) |

Run as independent Slurm jobs / a job array. Never serialize the grid.

---

## 7. METRICS (precise definitions — implement exactly)

- **Certified budget `k(q)`** = (number of edge-disjoint anchor→predicted-answer paths) − 1. Deletion certificate: proven robust to removal of ≤ `k` triples.
- **Insertion-certified** (P5): predicted answer is insertion-certified at budget `b` iff, under the stated threat model, no insertion of ≤ `b` triples makes any wrong answer's connectivity dominate.
- **Certified Accuracy@k** = fraction of queries where the predicted answer is correct AND `k(q) ≥ k`.
- **Certified coverage@k** = fraction of queries with `k(q) ≥ k` (independent of correctness).
- **Empirical robustness@b** = accuracy (EM/F1) under a given attack at injected budget `b`.
- **Certified-vs-empirical curve** = for each budget on the x-axis, plot the certified guarantee (lower bound) against measured empirical accuracy.
- **Selective prediction:** abstain when `k < τ`; report the **risk–coverage curve** and **AURC**, comparing the `k` signal vs LLM confidence vs self-consistency.

Every metric gets a unit test on a tiny hand-computed example.

---

## 8. DEFINITION OF DONE (all must be green)

Scope reminder: this targets the **complete empirical contribution, no written paper**. Every item below is at **full scale** — all datasets, models, attacks, budgets, and seeds in §6 — not a demo or subset.

- [ ] `pytest` green; `check_prereqs` green; environment reproduces from the lockfile via `uv sync --frozen`.
- [ ] Certificate engine (deletion) passes exact toy-graph tests; `k` computed on **every** dataset in §6.
- [ ] Insertion-aware certificate implemented, proven under a written conservative threat model (as a `threat_model.md`, not paper prose), toy-tested, and evaluated on all applicable datasets.
- [ ] P7 gate decided and recorded (GO or PIVOT), with evidence.
- [ ] Agentic retriever materially raises certified `k`/coverage over single-best-chain at modest accuracy cost — established with numbers across all reasoner models, not one.
- [ ] **Full experiment matrix (§6) run at full scale** — every (dataset × model × attack × budget × seed) cell either has a result or a logged, justified reason it was skipped. No cell is silently missing.
- [ ] Every reported number traces to a config + commit; all figures regenerate from `scripts/make_figures.py`; a cold re-run reproduces `results/FINAL/` within seed variance.
- [ ] Certified selective prediction evaluated vs confidence/self-consistency (AURC), result characterized either way, on all applicable datasets.
- [ ] Generalization to a text-constructed KG (HotpotQA/MuSiQue) run and reported end-to-end.
- [ ] Ablations done at full scale (hop-count vs path-count; structural vs evidential independence; LoRA effect; budget sensitivity; per-model).
- [ ] `results/FINAL/` complete: all tables (CSV + rendered), all figures, `RESULTS.md` stating each finding with pointers, and the claim↔evidence map linking the four contributions + H1–H3 to their supporting results.
- [ ] Reproduction README: one command per table/figure.
- [ ] `context/SOURCES.md` complete; every referenced work's PDF in `context/papers/`; every link verified to resolve.
- [ ] `PROGRESS.md` and `DECISIONS.md` complete and current.
- [ ] **No manuscript artifacts exist** (no LaTeX, abstract, or written sections) — confirming scope was respected.

When every box is checked, and not before, the project is done. The science is then complete and paper-ready; only the writing remains, as a separate later effort.

---

## 9. RISK REGISTER & PIVOTS

- **`k` is vacuous (most queries `k=0`).** Real KGs may rarely contain multiple genuinely independent paths. **Pivot:** promote the certificate-maximizing retriever (P8) to the headline finding — the contribution becomes *making* answers certifiable, not just measuring it — and establish it with the full evidence set to the same standard. Triggered by the P7 gate.
- **Insertion certificate overclaims.** Textbook Menger handles deletion cleanly; insertion needs careful assumptions. **Mitigation:** keep the threat model conservative and explicit; if the general insertion bound doesn't hold, scope the claim to the assumptions you can defend and say so.
- **Structural ≠ evidential independence.** Two edge-disjoint paths can share a corrupted source. **Mitigation:** distinguish them explicitly; measure the gap where possible; do not claim evidential independence you haven't shown.
- **Freebase setup blocks WebQSP/CWQ.** **Mitigation/fallback:** lead entirely with 2Wiki + MetaQA (self-contained), report WebQSP/CWQ on whatever is available with the caveat stated.
- **Dataset quality confounds (WebQSP/CWQ).** **Mitigation:** cleaned subsets + lead with 2Wiki gold paths.
- **"It's just a combination of known parts" (reviewer framing).** **Mitigation:** foreground the genuinely new object (per-answer compositional certificate), the insertion-aware extension, and the retriever that optimizes it — not the plumbing.
- **Contradiction in the empirical literature** (GraphRAG reported both more robust and easily poisoned). This is an argument *for* the paper: a provable budget replaces contradictory empirical claims. Use it in the motivation.

---

## 10. ESCALATION TRIGGERS (stop and ask Marko)

Escalate — do not silently work around — when:
- A **[HUMAN-REQUIRED]** prerequisite is missing or unobtainable (HF token, Llama license acceptance, Freebase endpoint down). Agent-downloadable prerequisites are not escalations — fetch them.
- Compute nodes turn out to be offline, or connectivity is otherwise different from assumed (changes the download/staging strategy).
- The P7 gate is genuinely ambiguous (near threshold) and the GO/PIVOT choice is unclear.
- A result would require fabrication to make the paper's story work — report the real result and ask how to proceed.
- Any cluster policy / quota / allocation issue that blocks the run.
- Anything requiring a human credential, license acceptance, or external account.

For everything else covered in this document: proceed autonomously, log the decision, and keep going toward the North Star.

---

## 11. REFERENCES & SOURCES

All source PDFs live in `context/papers/`; all links and versions live in `context/SOURCES.md`. **Verify every link resolves and open every PDF before citing it — never cite a source you have not read.** Anchor identifiers below are from the project research brief; confirm each on arXiv.

**Certified-RAG line (SOTA to beat):** RobustRAG (Xiang et al., arXiv:2405.15556); ReliabilityRAG (Shen et al., NeurIPS 2025, arXiv:2509.23519).
**KG-poisoning adversaries (reuse):** RAG Safety (Zhao et al., arXiv:2507.08862); GraphRAG under Fire / GragPoison (Liang et al., arXiv:2501.14050); KEPo (Chen et al., WWW 2026, arXiv:2603.11501).
**Certified-graph roots (adapt):** PGNNCert (CVPR 2025, arXiv:2503.18503); Provably Robust Explainable GNNs (arXiv:2502.04224); Randomized Smoothing (Cohen et al., ICML 2019, arXiv:1902.02918); deep partition aggregation / certified bagging (Levine & Feizi; Jia et al.).
**Graph-theoretic foundation:** Menger's theorem (1927); max-flow–min-cut (Ford–Fulkerson).
**Graph-RAG / KGQA roots:** RAG (Lewis et al., NeurIPS 2020, arXiv:2005.11401); Microsoft GraphRAG (Edge et al., 2024, arXiv:2404.16130); Reasoning-on-Graphs / RoG (Luo et al., ICLR 2024, arXiv:2310.01061); SubgraphRAG (ICLR 2025 — locate on arXiv).
**Benchmarks/KGs:** 2WikiMultiHopQA (Ho et al., COLING 2020); WebQSP (Yih et al., 2016); ComplexWebQuestions (Talmor & Berant, 2018); MetaQA; HotpotQA (Yang et al., 2018, arXiv:1809.09600); MuSiQue (Trivedi et al., TACL 2022). Dataset-quality caveat: arXiv:2505.23495.
**Discarded-direction evidence (do not re-do these):** Graph-R1 (arXiv:2507.21892); Efficient/Transferable Agentic KG-RAG via RL (arXiv:2509.26383); Plan-Then-Retrieve RL (arXiv:2510.20691); KGs are Implicit Reward Models (arXiv:2601.15160); Graph of Verification (arXiv:2506.12509); C²RAG (arXiv:2603.14828).
**Compute grounding:** CSCS Clariden / Alps docs; Apertus engineering paper (arXiv:2604.12973).
**Framing:** asymmetry of verification / verifier's law (Jason Wei, 2025).

---

*End of brief. The standing instruction is §0. Work the phases, respect the gates, keep the log, and do not stop until the Definition of Done is green.*
