# Certified-vs-empirical curve (AGENTS.md §7)

Attack flip rate vs injected-triple budget (Qwen2.5-7B-Instruct, MetaQA 2-hop
dev, over clean-correct queries). The certificate predicts that certified
(k>=1) answers stay robust as the budget grows while k=0 answers degrade.

| budget | k=0 flip | certified k>=1 flip | gap |
|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | +0.00 |
| 1 | 0.08 | 0.03 | +0.05 |
| 2 | 0.12 | 0.03 | +0.09 |
| 5 | 0.25 | 0.06 | +0.20 |
| 8 | 0.37 | 0.08 | +0.29 |
| 10 | 0.31 | 0.08 | +0.23 |
| 20 | 0.49 | 0.10 | +0.39 |

At budget 0 (no attack) both are ~0 (a sanity check). As the adversary
spends more inserted triples, the k=0 flip rate climbs steeply (to ~0.49
at budget 20) while certified queries stay low (~0.10) — the widening gap
is the certificate's empirical protection. Small non-monotonicities
(e.g. budget 8 vs 10) are finite-sample noise at single seed (~60 k=0
queries per point); the multi-seed point in stage1_gate.md pins budget 8.
