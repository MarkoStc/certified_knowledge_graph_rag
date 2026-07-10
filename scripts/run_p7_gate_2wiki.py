#!/usr/bin/env python3
"""P7 gate on 2WikiMultiHopQA (AGENTS.md P9 — cross-dataset generalization).

Same certified-vs-empirical test as run_p7_gate.py, but over the 2Wiki
Wikidata KG (real, obscure entities — a strong memorization control by
construction) with human-readable triples produced from Wikidata labels.

    srun --gres=gpu:1 ... uv run --extra llm python \\
        scripts/run_p7_gate_2wiki.py --split dev --per-k 40 --budget 8 \\
        --out-dir ...
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from mcgr.attacks.rag_safety import InsertionAttack
from mcgr.kg.graph_store import khop_subgraph, reconnect_endpoints
from mcgr.kg.twowiki_kg import (
    seed_qids,
    twowiki_graph_and_hubs,
    twowiki_pair_index,
    twowiki_relation_objects,
)
from mcgr.kg.wikidata import get_labels
from mcgr.models.reasoner import Reasoner
from mcgr.retrieval.evidence import path_evidence
from mcgr.seeding import seed_everything

RADIUS = 2


def load_k(path):
    with open(path) as f:
        return {r["qid"]: int(r["k"]) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-k", type=int, default=40)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    seed_everything(args.seed)

    cert_csv = (
        Path(os.environ["SCRATCH"])
        / "runs"
        / "certificates"
        / f"2wikimultihopqa_{args.split}"
        / "metrics.csv"
    )
    k_by_qid = load_k(cert_csv)
    _, queries = seed_qids(splits=(args.split,))  # (qid, split, anchor_qid, answer_qid)

    # stratify by k (supported only)
    buckets = defaultdict(list)
    for qid, _sp, anchor, answer in queries:
        k = k_by_qid.get(qid)
        if k is not None and k >= 0:
            buckets[k].append((qid, anchor, answer))
    rng = random.Random(args.seed)
    sample = []
    for k, qs in buckets.items():
        rng.shuffle(qs)
        sample.extend((q, k) for q in qs[: args.per_k])

    graph, hub_adj = twowiki_graph_and_hubs()
    pair_index = twowiki_pair_index()
    rel_objects = twowiki_relation_objects()
    entity_pool = tuple(sorted({s for v in pair_index.values() for s, _p, _o in v})[:20000])
    attack = InsertionAttack(budget=args.budget, seed=args.seed, entity_pool=entity_pool)

    prepared = []  # (qid, k, anchor, gold_qid, wrong_qid, evidence, attacked)
    for (qid, anchor, answer), k in sample:
        sub = khop_subgraph(graph, [anchor], RADIUS)
        reconnect_endpoints(sub, hub_adj, [anchor, answer])
        ev = path_evidence(sub, anchor, answer, pair_index)
        if not ev:
            continue
        rels = [p for _s, p, o in ev if o == answer]
        rel = rels[0] if rels else None
        sibs = [o for o in rel_objects.get(rel, ()) if o != answer] if rel else []
        wrong = rng.choice(sibs) if sibs else "__none__"
        atk = attack.apply(ev, anchor, ev[:2], wrong)
        random.Random(f"shuf:{qid}").shuffle(atk)
        prepared.append((qid, k, anchor, answer, wrong, ev, atk))

    # fetch labels for every Q-id / P-id we will render
    ids = set()
    for _qid, _k, _a, gold, wrong, ev, atk in prepared:
        ids.update({gold, wrong})
        for s, p, o in [*ev, *atk]:
            ids.update({s, p, o})
    labels = get_labels(sorted(ids))

    def render(triples):
        return [(labels.get(s, s), labels.get(p, p), labels.get(o, o)) for s, p, o in triples]

    # 2Wiki questions already contain the entity names; render triples only
    from mcgr.data import load_dataset

    q_text = {r.qid: r.question for r in load_dataset("2wikimultihopqa", args.split)}

    reasoner = Reasoner(model_name=args.model).load()

    def run(items):
        out = []
        for i in range(0, len(items), args.batch_size):
            out.extend(reasoner.answer_batch(items[i : i + args.batch_size]))
        return out

    clean_items = [(q_text[p[0]], render(p[5])) for p in prepared]
    atk_items = [(q_text[p[0]], render(p[6])) for p in prepared]
    clean_out = run(clean_items)
    atk_out = run(atk_items)

    per_k = defaultdict(lambda: {"n": 0, "clean_n": 0, "flip": 0, "clean": 0})
    for (_qid, k, _a, gold, wrong, _e, _t), ca, aa in zip(
        prepared, clean_out, atk_out, strict=True
    ):
        gl = labels.get(gold, gold).lower()
        wl = labels.get(wrong, wrong).lower()
        clean_ok = gl in ca.lower()
        flipped = wl in aa.lower() and gl not in aa.lower()
        d = per_k[k]
        d["n"] += 1
        d["clean"] += clean_ok
        if clean_ok:
            d["clean_n"] += 1
            d["flip"] += flipped

    summary = {
        "dataset": "2wikimultihopqa",
        "split": args.split,
        "model": args.model,
        "budget": args.budget,
        "n": len(prepared),
        "per_k": {
            str(k): {
                "n": d["n"],
                "clean_correct_n": d["clean_n"],
                "flip_rate": (d["flip"] / d["clean_n"]) if d["clean_n"] else None,
            }
            for k, d in sorted(per_k.items())
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
