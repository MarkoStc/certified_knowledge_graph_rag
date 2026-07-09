#!/usr/bin/env python3
"""P8 retriever ablation (AGENTS.md P8).

Compares two retrieval policies over the same MetaQA KG:
  - single-best-chain (standard KG-RAG baseline): one supporting chain,
    retrieved certificate k = 0 always;
  - certificate-maximizing (agentic): up to `max_paths` edge-disjoint
    anchor->answer paths, retrieved certificate k = n_paths - 1.

For a random query sample it reports, per policy: mean retrieved k,
certified coverage (frac k>=1), clean accuracy, and flip rate under the
RAG-Safety attack. Claim: the agentic policy raises certified coverage and
robustness at modest clean-accuracy cost.

    srun --gres=gpu:1 ... uv run --extra llm python \\
        scripts/run_retriever_ablation.py --dataset metaqa-3hop --split dev \\
        --limit 400 --budget 8 --out-dir ...
"""

import argparse
import json
import random
from pathlib import Path

from mcgr.attacks.rag_safety import InsertionAttack
from mcgr.data import load_dataset
from mcgr.kg.graph_store import khop_subgraph
from mcgr.kg.metaqa_kg import metaqa_graph
from mcgr.logging_utils import RunLogger, get_logger
from mcgr.models.reasoner import Reasoner
from mcgr.retrieval.agentic import certificate_maximizing, single_best_chain
from mcgr.retrieval.evidence import (
    metaqa_pair_index,
    metaqa_relation_objects,
    path_evidence,
)
from mcgr.seeding import seed_everything

log = get_logger("mcgr.p8")

POLICIES = {"single_best": single_best_chain, "agentic": certificate_maximizing}


def final_relation(evidence, answer):
    rels = [r for _s, r, o in evidence if o == answer]
    return rels[0] if rels else None


def pick_sibling_wrong(gold, rel, rel_objects, seed_key):
    rng = random.Random(seed_key)
    sib = [o for o in rel_objects.get(rel, ()) if o not in gold] if rel else []
    return rng.choice(sib) if sib else "__no_such_entity__"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa-3hop")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--radius", type=int, default=None, help="default = hop count")
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--max-paths", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    seed_everything(args.seed)
    hops = int(args.dataset.removeprefix("metaqa-").removesuffix("hop"))
    radius = args.radius or hops

    records = list(load_dataset(args.dataset, args.split))
    random.Random(args.seed).shuffle(records)
    graph = metaqa_graph()
    pair_index = metaqa_pair_index()
    rel_objects = metaqa_relation_objects()
    attack = InsertionAttack(
        budget=args.budget,
        seed=args.seed,
        entity_pool=tuple(sorted({s for v in pair_index.values() for s, _r, _o in v})[:20000]),
    )

    # prepare per-query, per-policy retrieved evidence
    prepared = []  # (qid, policy, n_paths, question, gold, wrong, clean_ctx, atk_ctx)
    for r in records:
        if len({p[0] for p in prepared}) >= args.limit:
            break
        anchors = list(r.anchor_entities)
        if not anchors:
            continue
        answer = r.answers[0]
        gold = set(r.answers)
        sub = khop_subgraph(graph, anchors, radius)
        # relation of the true final hop, for a coherent wrong answer
        ref = path_evidence(sub, anchors[0], answer, pair_index)
        if not ref:
            continue
        wrong = pick_sibling_wrong(
            gold, final_relation(ref, answer), rel_objects, f"{args.seed}:{r.qid}"
        )
        for pname, fn in POLICIES.items():
            triples, n_paths = fn(sub, anchors[0], answer, pair_index)
            if not triples:
                continue
            atk = attack.apply(triples, anchors[0], triples[:2], wrong)
            random.Random(f"{args.seed}:{r.qid}:{pname}").shuffle(atk)
            prepared.append((r.qid, pname, n_paths, r.question, gold, wrong, triples, atk))

    log.info("prepared %d (query,policy) items", len(prepared))
    reasoner = Reasoner(model_name=args.model).load()

    def run(items):
        out = []
        for i in range(0, len(items), args.batch_size):
            out.extend(reasoner.answer_batch(items[i : i + args.batch_size]))
        return out

    clean_out = run([(p[3], p[6]) for p in prepared])  # (question, clean context)
    atk_out = run([(p[3], p[7]) for p in prepared])  # (question, attacked context)

    runlog = RunLogger(args.out_dir)
    agg = {
        p: {"n": 0, "kpaths": 0, "cov": 0, "clean": 0, "flip": 0, "clean_n": 0} for p in POLICIES
    }
    for (qid, pname, n_paths, _q, gold, wrong, _c, _a), ca, aa in zip(
        prepared, clean_out, atk_out, strict=True
    ):
        gold_l = {g.lower() for g in gold}
        clean_ok = any(g in ca.lower() for g in gold_l)
        flipped = wrong.lower() in aa.lower() and not any(g in aa.lower() for g in gold_l)
        k = n_paths - 1
        a = agg[pname]
        a["n"] += 1
        a["kpaths"] += k
        a["cov"] += k >= 1
        a["clean"] += clean_ok
        if clean_ok:
            a["clean_n"] += 1
            a["flip"] += flipped
        runlog.log(
            {
                "qid": qid,
                "policy": pname,
                "retrieved_k": k,
                "clean_correct": int(clean_ok),
                "attacked_flipped": int(flipped),
            }
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "budget": args.budget,
        "policies": {
            p: {
                "n": a["n"],
                "mean_retrieved_k": a["kpaths"] / a["n"] if a["n"] else 0,
                "certified_coverage": a["cov"] / a["n"] if a["n"] else 0,
                "clean_acc": a["clean"] / a["n"] if a["n"] else 0,
                "flip_rate": a["flip"] / a["clean_n"] if a["clean_n"] else None,
            }
            for p, a in agg.items()
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
