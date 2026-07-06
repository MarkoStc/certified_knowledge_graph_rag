#!/usr/bin/env python3
"""Compute deletion-certificate distributions for a dataset split (P4).

Reproducible and Slurm-friendly:

    uv run python scripts/compute_certificates.py --dataset metaqa-2hop \\
        --split dev --limit 200 --out-dir $SCRATCH/runs/cert_metaqa2_dev

Writes a per-query metrics.csv (via RunLogger) and a JSON summary. The full
sweep over all splits is heavy (3-hop subgraphs reach ~60k edges); run it as
a Slurm job array, one cell per (dataset, split), not on the login node.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mcgr.certify.pipeline import certify_dataset
from mcgr.logging_utils import RunLogger, get_logger
from mcgr.seeding import seed_everything

log = get_logger("mcgr.certify")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. metaqa-1hop / metaqa-2hop / metaqa-3hop")
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap #queries (for smoke runs)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    runlog = RunLogger(args.out_dir)
    kdist: Counter[int] = Counter()
    n = 0
    for cq in certify_dataset(args.dataset, args.split, limit=args.limit):
        runlog.log(
            {
                "qid": cq.qid,
                "k": cq.k,
                "n_anchors": cq.n_anchors,
                "n_answers_supported": cq.n_answers_supported,
                "subgraph_edges": cq.subgraph_edges,
            }
        )
        kdist[cq.k] += 1
        n += 1
        if n % 200 == 0:
            log.info("%s/%s: %d queries certified", args.dataset, args.split, n)

    certifiable = sum(v for k, v in kdist.items() if k >= 1)
    supported = sum(v for k, v in kdist.items() if k >= 0)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_queries": n,
        "k_distribution": dict(sorted(kdist.items())),
        "frac_k_ge_1": certifiable / n if n else 0.0,
        "frac_supported": supported / n if n else 0.0,
        "mean_k_supported": (
            sum(k * v for k, v in kdist.items() if k >= 0) / supported if supported else 0.0
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", json.dumps(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
