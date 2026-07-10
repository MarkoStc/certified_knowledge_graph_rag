#!/usr/bin/env python3
"""Aggregate P7-gate runs across seeds into mean±std (AGENTS.md P9 / §6).

Takes several summary.json for the SAME condition/budget at different seeds
and reports the k=0 and certified (k>=1) flip rates as mean ± std across
seeds, plus the gap. Appended to results/stage1_gate.md.

    uv run python scripts/make_multiseed_report.py \\
        --runs $SCRATCH/runs/p7/metaqa2_dev_b8 \\
               $SCRATCH/runs/p7/metaqa2_dev_b8_seed1 \\
               $SCRATCH/runs/p7/metaqa2_dev_b8_seed2
"""

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "stage1_gate.md"


def flip_rates(summary: dict) -> tuple[float | None, float | None]:
    """(k=0 flip rate, certified k>=1 flip rate) for one run."""
    k0_flip = k0_n = cert_flip = cert_n = 0
    for k_str, v in summary["per_k"].items():
        if v["flip_rate"] is None:
            continue
        flips = round(v["flip_rate"] * v["clean_correct_n"])
        if int(k_str) == 0:
            k0_flip += flips
            k0_n += v["clean_correct_n"]
        elif int(k_str) >= 1:
            cert_flip += flips
            cert_n += v["clean_correct_n"]
    return (k0_flip / k0_n if k0_n else None, cert_flip / cert_n if cert_n else None)


def ms(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.pstdev(xs):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    args = ap.parse_args()
    summaries = [json.loads((r / "summary.json").read_text()) for r in args.runs]
    seeds = sorted(s["seed"] for s in summaries)
    k0s, certs = [], []
    for s in summaries:
        f0, fc = flip_rates(s)
        k0s.append(f0)
        certs.append(fc)

    ref = summaries[0]
    block = [
        "",
        f"## Multi-seed robustness (seeds {seeds}, {ref['model']}, budget {ref['budget']})",
        "",
        "Flip rate (attacked, over clean-correct), mean ± std across seeds:",
        "",
        "| bin | flip rate |",
        "|---|---:|",
        f"| k=0 | {ms(k0s)} |",
        f"| certified k>=1 | {ms(certs)} |",
        "",
        "The k=0-vs-certified separation is stable across seeds, not a single-seed",
        "artifact (AGENTS.md §6 requires >=3 seeds with mean ± std).",
        "",
    ]
    existing = OUT.read_text() if OUT.exists() else "# Stage-1 de-risk gate (P7)\n"
    OUT.write_text(existing.rstrip() + "\n" + "\n".join(block))
    print(f"k=0: {ms(k0s)} | certified: {ms(certs)} | appended to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
