# RESULTS — Menger-Certified Graph-RAG (evidence pack)

Consolidated evidence for the empirical contribution (AGENTS.md §0/§8). Each
finding is a one-line factual claim with a pointer to the exact table/figure
and the script that regenerates it. **Status: the core is complete on two
self-contained datasets (MetaQA, 2WikiMultiHopQA); WebQSP/CWQ, Llama, and the
full §6 matrix remain (see "Not yet done").**

Reasoner: Qwen2.5-7B-Instruct (+ 14B for cross-model). Attack: RAG-Safety
triple insertion. Certificate graph: undirected k-hop fact subgraph; 2Wiki
adds role-aware hub pruning. Figures: `results/FINAL/figures/` via
`scripts/make_figures.py`.

## Findings

1. **A per-answer deletion certificate `k` is computable at scale on real KGs.**
   MetaQA 1/2/3-hop certified coverage (k≥1) = 0.5% / 31% / 97%; 2Wiki
   (compositional/inference) = 68%. → `results/certificate_distribution.md`,
   fig `certificate_coverage.png`; `scripts/compute_certificates.py`,
   `slurm/certify_metaqa.sbatch`, `slurm/certify_2wiki.sbatch`.

2. **`k` predicts empirical robustness to insertion attacks (H1 / P7 = GO).**
   MetaQA 2-hop, budget 8: k=0 flip 0.37 vs certified 0.09. → `stage1_gate.md`;
   `scripts/run_p7_gate.py`.

3. **The effect is graph-structural, not memorization.** Holds cross-model
   (14B: 0.19 vs 0.03) and under entity anonymization (0.42 vs 0.07, steeper;
   clean acc drops most at k=0). → `stage1_gate.md` "Robustness of the effect",
   fig `cross_condition.png`; `run_p7_gate.py --anonymize`,
   `--model Qwen2.5-14B-Instruct`.

4. **Stable across seeds.** 3 seeds, budget 8: k=0 flip 0.324 ± 0.034 vs
   certified 0.088 ± 0.007. → `stage1_gate.md` "Multi-seed";
   `scripts/make_multiseed_report.py`.

5. **Certified-vs-empirical curve.** k=0 flip rises 0→0.49 over budgets
   0…20 while certified stays ≤0.10; budget 0 = 0/0 sanity check. →
   `results/certified_vs_empirical.md`, fig `certified_vs_empirical.png`;
   `scripts/make_curve_report.py`.

6. **Generalizes to a second real-world KG (2Wiki, obscure entities).**
   budget 8: k=0 flip 0.189 vs certified 0.012; budget 2: 0.054 vs 0.002. →
   `stage1_gate.md` "Cross-dataset"; `scripts/run_p7_gate_2wiki.py`.

7. **Insertion-aware certificate (contribution #2) is distinct from deletion.**
   b_ins = paths(answer) − max competitor paths − 1; MetaQA 2-hop b_ins≥1
   15.3% vs deletion k≥1 16.8%, strictly tighter for 22% of queries. →
   `results/insertion_certificate.md`, `threat_model.md`;
   `scripts/compute_insertion_certificates.py`.

8. **The certificate-maximizing retriever makes answers certifiable
   (contribution #3).** MetaQA 3-hop: certified coverage 0%→82%, clean acc
   32%→78%, attack flip 0.18→0.035 vs single-best-chain. →
   `results/retriever_ablation.md`, fig `retriever_ablation.png`;
   `scripts/run_retriever_ablation.py`.

9. **`k` beats self-consistency for certified selective prediction (H3 /
   contribution #4).** Under attack, AURC(k) < AURC(self-consistency):
   2-hop 0.305 vs 0.374, 3-hop 0.236 vs 0.277. →
   `results/selective_prediction.md`; `scripts/run_selective_prediction.py`.

## Claim ↔ evidence map

| Brief item | Established by | Evidence |
|---|---|---|
| Contribution 1 — deletion certificate | Findings 1, 2, 4 | certificate_distribution.md, stage1_gate.md |
| Contribution 2 — insertion certificate | Finding 7 | insertion_certificate.md, threat_model.md |
| Contribution 3 — certificate-maximizing retriever | Finding 8 | retriever_ablation.md |
| Contribution 4 — metric suite + selective prediction | Findings 5, 9 | certified_vs_empirical.md, selective_prediction.md |
| H1 — k ⇒ robustness | Findings 2, 3, 4, 5, 6 | stage1_gate.md |
| H3 — k is a better abstention signal | Finding 9 | selective_prediction.md |
| P7 gate decision | Findings 2–4 | DECISIONS.md (GO) |

## Reproduction

```sh
uv sync                              # CPU: certificates, metrics, figures
uv sync --extra llm                  # + GPU reasoner (GH200)
# certificates (CPU/Slurm)
sbatch slurm/certify_metaqa.sbatch ; sbatch slurm/certify_2wiki.sbatch
uv run python scripts/make_certificate_report.py
# P7 gate + conditions (GPU)
uv run python scripts/run_p7_gate.py --dataset metaqa-2hop --split dev --budget 8 --out-dir <d>
uv run python scripts/make_stage1_report.py --runs <d> ...
# retriever, selective prediction, insertion, figures
uv run python scripts/run_retriever_ablation.py --dataset metaqa-3hop --split dev --out-dir <d>
uv run python scripts/run_selective_prediction.py --dataset metaqa-2hop --split dev --out-dir <d>
uv run python scripts/compute_insertion_certificates.py --dataset metaqa-2hop --split dev
uv run python scripts/make_figures.py
```

## Not yet done (full §6 matrix)

- **Datasets**: WebQSP, CWQ (need the Freebase endpoint — `context/BLOCKED.md`);
  P11 text-KG generalization (HotpotQA, MuSiQue).
- **Models**: Llama-3.1-8B (gated — token + license, `context/BLOCKED.md`).
- **Baselines**: P3 RoG / SubgraphRAG / GraphRAG reproductions.
- **Scale**: reasoning experiments are single-model / single-split samples;
  the full dataset × model × attack × budget × seed grid is not yet run.
- **Confound to close**: 2Wiki clean accuracy falls with k (dense-subgraph
  context length); a length-controlled retriever would remove it.
