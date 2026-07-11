#!/usr/bin/env python3
"""Regenerate the paper figures from cached run data (AGENTS.md P12 / §8).

Every figure is produced from the JSON summaries under $SCRATCH/runs, so the
figures are reproducible from the logged experiments. Missing runs are
skipped with a note.

    uv run python scripts/make_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGDIR = REPO_ROOT / "results" / "FINAL" / "figures"
RUNS = Path(os.environ["SCRATCH"]) / "runs"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def k0_vs_cert(summary):
    k0f = k0n = cf = cn = 0
    for ks, v in summary["per_k"].items():
        if v["flip_rate"] is None:
            continue
        fl = round(v["flip_rate"] * v["clean_correct_n"])
        if int(ks) == 0:
            k0f, k0n = k0f + fl, k0n + v["clean_correct_n"]
        elif int(ks) >= 1:
            cf, cn = cf + fl, cn + v["clean_correct_n"]
    return (k0f / k0n if k0n else None, cf / cn if cn else None)


def fig_certificate_coverage():
    cells = [
        ("MetaQA\n1-hop", "certificates/metaqa-1hop_dev"),
        ("MetaQA\n2-hop", "certificates/metaqa-2hop_dev"),
        ("MetaQA\n3-hop", "certificates/metaqa-3hop_dev"),
        ("2Wiki", "certificates/2wikimultihopqa_dev"),
    ]
    labels, vals = [], []
    for name, rel in cells:
        s = load(RUNS / rel / "summary.json")
        if s:
            labels.append(name)
            vals.append(100 * s["frac_k_ge_1"])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, vals, color="#3b7dd8")
    ax.set_ylabel("certified coverage (% k≥1)")
    ax.set_title("Certified coverage by dataset")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIGDIR / "certificate_coverage.png", dpi=140)
    plt.close(fig)


def fig_budget_curve():
    budgets = [0, 1, 2, 5, 8, 10, 20]
    xs, k0, cert = [], [], []
    for b in budgets:
        s = load(RUNS / f"p7/metaqa2_dev_b{b}" / "summary.json")
        if not s:
            continue
        f0, fc = k0_vs_cert(s)
        xs.append(b)
        k0.append(f0)
        cert.append(fc)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(xs, k0, "o-", color="#d8453b", label="uncertified (k=0)")
    ax.plot(xs, cert, "s-", color="#3b7dd8", label="certified (k≥1)")
    ax.set_xlabel("attack budget (inserted triples)")
    ax.set_ylabel("attack flip rate")
    ax.set_title("Certified-vs-empirical (MetaQA 2-hop, Qwen2.5-7B)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "certified_vs_empirical.png", dpi=140)
    plt.close(fig)


def fig_cross_condition():
    conds = [
        ("MetaQA 7B", "p7/metaqa2_dev_b8"),
        ("MetaQA 14B", "p7/metaqa2_dev_14b_b8"),
        ("MetaQA\nLlama-8B", "p7/metaqa2_dev_llama_b8"),
        ("MetaQA 7B\nanon", "p7/metaqa2_dev_anon_b8"),
        ("2Wiki 7B", "p7/2wiki_dev_b8"),
    ]
    labels, k0s, certs = [], [], []
    for name, rel in conds:
        s = load(RUNS / rel / "summary.json")
        if not s:
            continue
        f0, fc = k0_vs_cert(s)
        labels.append(name)
        k0s.append(f0 or 0)
        certs.append(fc or 0)
    if not labels:
        return
    import numpy as np

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - w / 2, k0s, w, label="k=0", color="#d8453b")
    ax.bar(x + w / 2, certs, w, label="certified k≥1", color="#3b7dd8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("attack flip rate (budget 8)")
    ax.set_title("k=0 vs certified across model / anonymization / dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "cross_condition.png", dpi=140)
    plt.close(fig)


def fig_retriever():
    import numpy as np

    rows = []
    for name, rel in [("2-hop", "p8/metaqa2_dev"), ("3-hop", "p8/metaqa3_dev")]:
        s = load(RUNS / rel / "summary.json")
        if s:
            rows.append((name, s["policies"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(rows))
    w = 0.38
    sb = [100 * r[1]["single_best"]["certified_coverage"] for r in rows]
    ag = [100 * r[1]["agentic"]["certified_coverage"] for r in rows]
    ax.bar(x - w / 2, sb, w, label="single-best-chain", color="#999999")
    ax.bar(x + w / 2, ag, w, label="certificate-maximizing", color="#3b9d5b")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("certified coverage (%)")
    ax.set_title("Retriever ablation (MetaQA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "retriever_ablation.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_certificate_coverage()
    fig_budget_curve()
    fig_cross_condition()
    fig_retriever()
    made = sorted(p.name for p in FIGDIR.glob("*.png"))
    print(f"wrote {len(made)} figures to {FIGDIR.relative_to(REPO_ROOT)}: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
