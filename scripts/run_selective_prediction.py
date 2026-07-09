#!/usr/bin/env python3
"""P10 certified selective prediction (AGENTS.md P10 / H3).

Under the RAG-Safety attack, compares abstention signals by AURC (area under
the risk-coverage curve, lower = better):
  - certificate `k` (structural grounding),
  - self-consistency agreement (sample the reasoner, majority fraction),
  - clean confidence proxy (greedy-vs-sample agreement).

H3: `k` is the better abstention signal, because a model can be
*consistently wrong* through a forged single chain (k=0) that self-
consistency rates as confident but `k` flags.

    srun --gres=gpu:1 ... uv run --extra llm python \\
        scripts/run_selective_prediction.py --dataset metaqa-2hop \\
        --split dev --per-k 60 --budget 8 --out-dir ...
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from mcgr.attacks.rag_safety import InsertionAttack
from mcgr.data import load_dataset
from mcgr.eval.selective import aurc, risk_coverage
from mcgr.kg.graph_store import khop_subgraph
from mcgr.kg.metaqa_kg import metaqa_graph
from mcgr.models.reasoner import Reasoner
from mcgr.retrieval.evidence import (
    metaqa_pair_index,
    metaqa_relation_objects,
    path_evidence,
)
from mcgr.seeding import seed_everything


def load_k_by_qid(path):
    with open(path) as f:
        return {r["qid"]: int(r["k"]) for r in csv.DictReader(f)}


def stratified(records, k_by_qid, per_k, seed):
    """Up to ``per_k`` supported (k>=0) queries per k value, as (record, k)."""
    buckets = defaultdict(list)
    for r in records:
        k = k_by_qid.get(r.qid)
        if k is not None and k >= 0:
            buckets[k].append(r)
    rng = random.Random(seed)
    out = []
    for k, rs in buckets.items():
        rng.shuffle(rs)
        out.extend((r, k) for r in rs[:per_k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa-2hop")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-k", type=int, default=60)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    seed_everything(args.seed)
    hops = int(args.dataset.removeprefix("metaqa-").removesuffix("hop"))
    radius = args.radius or hops

    cert_csv = (
        Path(os.environ["SCRATCH"])
        / "runs"
        / "certificates"
        / f"{args.dataset}_{args.split}"
        / "metrics.csv"
    )
    k_by_qid = load_k_by_qid(cert_csv)
    records = list(load_dataset(args.dataset, args.split))
    sample = stratified(records, k_by_qid, args.per_k, args.seed)

    graph = metaqa_graph()
    pair_index = metaqa_pair_index()
    rel_objects = metaqa_relation_objects()
    entity_pool = tuple(sorted({s for v in pair_index.values() for s, _r, _o in v})[:20000])
    attack = InsertionAttack(budget=args.budget, seed=args.seed, entity_pool=entity_pool)

    prepared = []  # (qid, k, question, gold, attacked_ctx)
    for r, k in sample:
        anchors = list(r.anchor_entities)
        if not anchors:
            continue
        answer = r.answers[0]
        gold = set(r.answers)
        sub = khop_subgraph(graph, anchors, radius)
        ev = path_evidence(sub, anchors[0], answer, pair_index)
        if not ev:
            continue
        rels = [rr for _s, rr, o in ev if o == answer]
        rel = rels[0] if rels else None
        sib = [o for o in rel_objects.get(rel, ()) if o not in gold] if rel else []
        wrong = random.Random(f"{args.seed}:{r.qid}").choice(sib) if sib else "__none__"
        atk = attack.apply(ev, anchors[0], ev[:2], wrong)
        random.Random(f"shuf:{r.qid}").shuffle(atk)
        prepared.append((r.qid, k, r.question, gold, atk))

    reasoner = Reasoner(model_name=args.model).load()
    items = [(q, ctx) for _qid, _k, q, _g, ctx in prepared]

    # greedy answer under attack -> correctness (the risk outcome)
    greedy = []
    for i in range(0, len(items), args.batch_size):
        greedy.extend(reasoner.answer_batch(items[i : i + args.batch_size]))
    # self-consistency signal under attack
    consistency = []
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        consistency.extend(reasoner.self_consistency(batch, args.n_samples))

    k_sig, cons_sig, correct = [], [], []
    for (_qid, k, _q, gold, _c), ga, (_maj, agree) in zip(
        prepared, greedy, consistency, strict=True
    ):
        gold_l = {g.lower() for g in gold}
        ok = any(g in ga.lower() for g in gold_l)
        k_sig.append(float(k))
        cons_sig.append(agree)
        correct.append(ok)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "budget": args.budget,
        "n": len(correct),
        "attacked_accuracy": sum(correct) / len(correct) if correct else 0,
        "aurc": {
            "certificate_k": aurc(k_sig, correct),
            "self_consistency": aurc(cons_sig, correct),
        },
        "risk_coverage": {
            "certificate_k": risk_coverage(k_sig, correct),
            "self_consistency": risk_coverage(cons_sig, correct),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("n", "attacked_accuracy", "aurc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
