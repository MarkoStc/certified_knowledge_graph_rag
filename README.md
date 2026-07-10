# Menger-Certified Graph-RAG (mcgr)

Provable per-answer robustness for multi-hop knowledge-graph reasoning via
path redundancy. Each answer gets a certified budget `k` = (edge-disjoint
anchor→answer supporting paths) − 1, so no adversary deleting ≤ `k` triples
can destroy all support (Menger's theorem / max-flow–min-cut). The project
shows this structural certificate **predicts empirical robustness** of an LLM
reasoner under adversarial triple insertion, and builds a retriever that
actively raises it.

**`AGENTS.md`** is the authoritative research brief (phases, matrix,
definition of done). **`results/FINAL/RESULTS.md`** is the consolidated
evidence pack (every finding + a pointer to its table/figure + the script
that regenerates it). This README is the reproduction entry point.

## Headline results

- **The certificate is computable at scale** on curated, real, and
  text-constructed KGs. Certified coverage (k≥1): MetaQA 1/2/3-hop =
  0.5% / 31% / 97%; 2WikiMultiHopQA (Wikidata) = 68%; HotpotQA (text) = 19%.
- **`k` predicts robustness (P7 gate = GO).** Under RAG-Safety insertion
  (budget 8), uncertified (k=0) answers flip far more than certified ones:
  MetaQA 0.324 ± 0.034 vs 0.088 ± 0.007 (3 seeds); holds cross-model (14B),
  under entity anonymization (no memorization), and on 2Wiki's obscure
  entities.
- **A certificate-maximizing retriever makes answers certifiable.** MetaQA
  3-hop: certified coverage 0%→82%, attack flip 0.18→0.035, at no
  clean-accuracy cost.
- **`k` is a better abstention signal than self-consistency** (lower AURC),
  and an **insertion-aware certificate** (`threat_model.md`) extends the
  guarantee to forged competing paths.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # CPU: certificates, metrics, figures, tests
uv sync --extra llm     # + GPU reasoner stack (torch CUDA, aarch64/GH200)
```

On CSCS compute nodes keep caches off the home quota:
`export UV_CACHE_DIR=$SCRATCH/.uv-cache`. Datasets/models live under
`$SCRATCH` (see `context/SOURCES.md`); `scripts/check_prereqs.py` verifies
prerequisites. Note: `uv sync` (no extra) removes torch — re-run with
`--extra llm` before any GPU job.

## Reproduce (one command per artifact)

CPU / Slurm (no GPU):

```sh
sbatch slurm/certify_metaqa.sbatch          # MetaQA certificate sweep
sbatch slurm/certify_2wiki.sbatch           # 2Wiki certificate sweep
uv run python scripts/make_certificate_report.py      # -> results/certificate_distribution.md
uv run python scripts/compute_insertion_certificates.py --dataset metaqa-2hop --split dev
uv run python scripts/compute_text_kg_certificates.py                 # HotpotQA text-KG
uv run python scripts/make_figures.py                 # -> results/FINAL/figures/
```

GPU (GH200; `srun --gres=gpu:1 ...`, use `.venv/bin/python` or `uv run --extra llm`):

```sh
python scripts/run_p7_gate.py --dataset metaqa-2hop --split dev --budget 8 --out-dir <d>
python scripts/make_stage1_report.py --runs <d> ...          # -> results/stage1_gate.md
python scripts/run_retriever_ablation.py --dataset metaqa-3hop --split dev --out-dir <d>
python scripts/run_selective_prediction.py --dataset metaqa-2hop --split dev --out-dir <d>
python scripts/run_p7_gate_2wiki.py --split dev --budget 8 --out-dir <d>   # prefetch labels off-GPU first
```

## Repo layout

```
src/mcgr/
  certify/     deletion (menger.py) + insertion (insertion.py) certificates
  kg/          graph store, MetaQA/2Wiki/text KG builders, Wikidata fetcher
  data/        dataset loaders (uniform QARecord)
  retrieval/   evidence retrieval + single-best vs certificate-maximizing
  attacks/     RAG-Safety triple-insertion attack
  models/      Qwen reasoner over retrieved triples
  eval/        risk-coverage / AURC
scripts/       compute + report + figure generators
slurm/         certificate sweeps
results/       generated tables; results/FINAL/ = evidence pack
threat_model.md  formal, conservative threat model (deletion + insertion)
```

## Development

```sh
uv run pytest              # hermetic tests (data/prereqs marked, deselected)
uv run ruff format . ; uv run ruff check .
uv run mcgr-demo           # certificate on hand-checked toy graphs
```

Lint/format/test config in `pyproject.toml`; `uv.lock` is committed.
