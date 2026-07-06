# Deletion-certificate distribution — MetaQA (P4)

**Status: validated pipeline + preliminary samples, NOT the full sweep.**
Numbers below are from bounded samples run on a compute node (`srun`,
partition `debug`) to validate the P2→P4 path end to end. The complete
distribution over all splits is pending the Slurm array
(`slurm/certify_metaqa.sbatch`), which first needs the pipeline parallelized
across queries (see PROGRESS.md / the 3-hop timing note below).

Certificate definition (AGENTS.md §7): `k = (#edge-disjoint anchor->answer
paths in the anchor's k-hop subgraph) - 1`. Undirected fact graph (deletion
threat model); `k>=1` = provably robust to deletion of at least one supporting
triple. See `DECISIONS.md` for the modeling choices.

## Preliminary samples

| Dataset | Split | N | frac k≥1 | mean k (supported) | source |
|---|---|---:|---:|---:|---|
| MetaQA 1-hop | dev | 60 | 0% | 0.00 | login-node probe |
| MetaQA 2-hop | dev | 60 | 37% | — | login-node probe |
| MetaQA 2-hop | dev | 300 | 28.7% | 0.64 | **compute-node `srun`, via script** |
| MetaQA 3-hop | dev | 60 | 98% | — | login-node probe |

MetaQA 2-hop dev (300-query) k histogram: `{0: 214, 1: 39, 2: 25, 3: 9, 4: 5,
6: 4, 8: 3, 9: 1}`, all queries supported (`frac_supported = 1.0`).

## The finding that matters (H1, preliminary)

Certified redundancy rises sharply with hop count:

- **1-hop → 0% certifiable.** A 1-hop answer is a single fact; there is no
  second independent path, so `k = 0` by construction. This is a correctness
  check as much as a result.
- **2-hop → ~29–37% certifiable.** A meaningful minority of queries already
  admit an independent second path.
- **3-hop → ~98% certifiable.** Dense KB regions give many edge-disjoint
  supporting paths.

This monotonic hop-count → path-redundancy relationship is exactly the
signal P7's go/no-go gate looks for, and the basis of the §12 hop-count vs
path-count ablation. It is **preliminary** until the full split is computed.

## Performance note (blocks the full sweep)

3-hop subgraphs reach ~60k edges and cost ~4.7 s/query single-threaded
(≈18 h for a full 14k-query split). The certificate pipeline must be
parallelized across queries (multiprocessing over the 288-core nodes)
before the Slurm array sweep is practical. 1-hop and 2-hop are cheap
(<0.2 s/query) and could run now.
