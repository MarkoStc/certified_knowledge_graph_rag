#!/usr/bin/env python3
"""P7 Stage-1 de-risk gate (AGENTS.md P7).

Tests H1: does the certificate ``k`` predict empirical robustness under the
RAG-Safety insertion attack? For MetaQA queries stratified by ``k``, the
reasoner answers from (a) clean supporting evidence and (b) evidence plus a
forged competing chain to a wrong answer. We report clean accuracy, and —
among queries answered correctly when clean — the attack flip rate per ``k``.
Prediction: flip rate falls as ``k`` rises.

Run on a GH200:
    srun --gres=gpu:1 ... uv run --extra llm python scripts/run_p7_gate.py \\
        --dataset metaqa-2hop --split dev --per-k 120 --budget 2 --out-dir ...
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
from mcgr.kg.graph_store import khop_subgraph
from mcgr.kg.metaqa_kg import metaqa_graph
from mcgr.logging_utils import RunLogger, get_logger
from mcgr.models.reasoner import Reasoner
from mcgr.retrieval.evidence import (
    metaqa_pair_index,
    metaqa_relation_objects,
    path_evidence,
)
from mcgr.seeding import seed_everything

log = get_logger("mcgr.p7")


def load_k_by_qid(cert_csv: Path) -> dict[str, int]:
    with cert_csv.open() as f:
        return {row["qid"]: int(row["k"]) for row in csv.DictReader(f)}


def stratified_sample(records, k_by_qid, per_k, seed):
    """Up to ``per_k`` supported (k>=0) queries per k value."""
    buckets: dict[int, list] = defaultdict(list)
    for r in records:
        k = k_by_qid.get(r.qid)
        if k is not None and k >= 0:
            buckets[k].append(r)
    rng = random.Random(seed)
    sample = []
    for k, rs in buckets.items():
        rng.shuffle(rs)
        sample.extend((r, k) for r in rs[:per_k])
    return sample


def final_relation(evidence, answer):
    """The relation of the evidence hop(s) that land on the gold answer."""
    rels = [r for _s, r, o in evidence if o == answer]
    return rels[0] if rels else None


def pick_sibling_wrong_answer(gold, rel, rel_objects, fallback_pool, seed_key):
    """A type-consistent wrong answer: another object of the same relation
    (a language for a language question), so the forged chain is coherent."""
    rng = random.Random(seed_key)
    siblings = [o for o in rel_objects.get(rel, ()) if o not in gold] if rel else []
    if siblings:
        return rng.choice(siblings)
    for _ in range(10):
        cand = rng.choice(fallback_pool)
        if cand not in gold:
            return cand
    return "__no_such_entity__"


def anonymize(question, anchor, evidence, attacked, gold, wrong):
    """Rename every entity to an opaque token, consistently within this query.

    Forces the reasoner to rely on the provided triples rather than
    parametric memory of the real entities — a control for the memorization
    confound. Relations are kept (they carry the reasoning structure)."""
    entities: dict[str, str] = {}

    def label(e: str) -> str:
        if e not in entities:
            entities[e] = f"Entity_{len(entities) + 1}"
        return entities[e]

    # anchor first so it maps deterministically; then evidence, then forged
    label(anchor)
    for s, _r, o in [*evidence, *attacked]:
        label(s)
        label(o)
    for g in gold:
        label(g)
    label(wrong)

    def remap(triples):
        return [(entities[s], r, entities[o]) for s, r, o in triples]

    q = question.replace(anchor, entities[anchor])
    return (
        q,
        remap(evidence),
        remap(attacked),
        {entities[g] for g in gold},
        entities[wrong],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa-2hop")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-k", type=int, default=120)
    ap.add_argument("--budget", type=int, default=2, help="inserted triples")
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--anonymize",
        action="store_true",
        help="rename entities to opaque tokens (memorization control)",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    seed_everything(args.seed)

    cert_csv = (
        Path(os.environ["SCRATCH"])
        / "runs"
        / "certificates"
        / f"{args.dataset}_{args.split}"
        / "metrics.csv"
    )
    k_by_qid = load_k_by_qid(cert_csv)
    records = list(load_dataset(args.dataset, args.split))
    sample = stratified_sample(records, k_by_qid, args.per_k, args.seed)
    log.info("sampled %d queries across %d k-values", len(sample), len({k for _, k in sample}))

    graph = metaqa_graph()
    pair_index = metaqa_pair_index()
    rel_objects = metaqa_relation_objects()
    answer_pool = sorted({a for r in records for a in r.answers})
    # plausible real intermediates for forged chains (KB entities, not answers)
    entity_pool = tuple(sorted({s for objs in pair_index.values() for s, _r, _o in objs})[:20000])
    attack = InsertionAttack(budget=args.budget, seed=args.seed, entity_pool=entity_pool)

    # build per-query evidence, wrong answer, clean+attacked contexts
    prepared = []
    for r, k in sample:
        anchors = list(r.anchor_entities)
        gold = set(r.answers)
        answer = r.answers[0]
        sub = khop_subgraph(graph, anchors, args.radius)
        evidence = path_evidence(sub, anchors[0], answer, pair_index) if anchors else []
        if not evidence:
            continue
        rel = final_relation(evidence, answer)
        wrong = pick_sibling_wrong_answer(
            gold, rel, rel_objects, answer_pool, f"{args.seed}:{r.qid}"
        )
        gold_chain = evidence[:2]  # mirror the relation structure of one real path
        attacked = attack.apply(evidence, anchors[0], gold_chain, wrong)
        # shuffle so forged triples are not positionally distinguishable
        random.Random(f"shuf:{args.seed}:{r.qid}").shuffle(attacked)
        question = r.question
        if args.anonymize:
            question, evidence, attacked, gold, wrong = anonymize(
                question, anchors[0], evidence, attacked, gold, wrong
            )
        prepared.append((r.qid, k, question, gold, wrong, evidence, attacked))

    log.info("prepared %d queries with evidence", len(prepared))
    reasoner = Reasoner(model_name=args.model).load()

    def run(items):
        outs = []
        for i in range(0, len(items), args.batch_size):
            outs.extend(reasoner.answer_batch(items[i : i + args.batch_size]))
            if (i // args.batch_size) % 10 == 0:
                log.info("  %d/%d", i, len(items))
        return outs

    clean_items = [(q, ev) for _, _, q, _, _, ev, _ in prepared]
    atk_items = [(q, at) for _, _, q, _, _, _, at in prepared]
    clean_out = run(clean_items)
    atk_out = run(atk_items)

    runlog = RunLogger(args.out_dir)
    per_k_clean_correct: dict[int, int] = defaultdict(int)
    per_k_total: dict[int, int] = defaultdict(int)
    per_k_flip: dict[int, int] = defaultdict(int)
    per_k_clean_n: dict[int, int] = defaultdict(int)
    for (qid, k, _q, gold, wrong, _, _), ca, aa in zip(prepared, clean_out, atk_out, strict=True):
        gold_l = {g.lower() for g in gold}
        clean_correct = any(g in ca.lower() for g in gold_l)
        flipped = wrong.lower() in aa.lower() and not any(g in aa.lower() for g in gold_l)
        per_k_total[k] += 1
        per_k_clean_correct[k] += clean_correct
        if clean_correct:
            per_k_clean_n[k] += 1
            per_k_flip[k] += flipped
        runlog.log(
            {
                "qid": qid,
                "k": k,
                "clean_correct": int(clean_correct),
                "attacked_flipped": int(flipped),
                "clean_answer": ca.replace("\n", " ")[:60],
                "attacked_answer": aa.replace("\n", " ")[:60],
            }
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "anonymized": args.anonymize,
        "budget": args.budget,
        "seed": args.seed,
        "n": len(prepared),
        "per_k": {
            str(k): {
                "n": per_k_total[k],
                "clean_acc": per_k_clean_correct[k] / per_k_total[k],
                "clean_correct_n": per_k_clean_n[k],
                "flip_rate": (per_k_flip[k] / per_k_clean_n[k]) if per_k_clean_n[k] else None,
            }
            for k in sorted(per_k_total)
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
