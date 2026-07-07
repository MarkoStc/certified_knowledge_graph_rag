#!/usr/bin/env python3
"""Fetch and cache the 2WikiMultiHopQA Wikidata KG (AGENTS.md P2).

Collects the seed Q-ids from compositional/inference gold chains, fetches
each entity's claims from Wikidata (batched, cached), and writes the
entity-valued triples edgelist to $SCRATCH/data/2wiki_kg/triples.tsv.

    uv run python scripts/build_2wiki_kg.py            # full (~84k entities)
    uv run python scripts/build_2wiki_kg.py --limit 2000   # smoke

Network-bound; run on a node with internet. Idempotent: cached batches are
reused, so re-running resumes rather than refetching.
"""

import argparse
import time

from mcgr.kg.twowiki_kg import seed_qids, triples_path, write_triples
from mcgr.kg.wikidata import BATCH, entity_triples, get_entities
from mcgr.logging_utils import get_logger

log = get_logger("mcgr.build_2wiki")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #seed entities (smoke)")
    args = ap.parse_args()

    seeds, queries = seed_qids()
    if args.limit:
        seeds = seeds[: args.limit]
    log.info("seeds=%d, queries=%d", len(seeds), len(queries))

    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    t0 = time.time()
    for i in range(0, len(seeds), BATCH):
        chunk = seeds[i : i + BATCH]
        entities = get_entities(chunk)
        for ent in entities.values():
            for tr in entity_triples(ent):
                if tr not in seen:
                    seen.add(tr)
                    triples.append(tr)
        done = i + len(chunk)
        if done % (BATCH * 20) == 0:
            rate = done / (time.time() - t0)
            log.info("fetched %d/%d (%.0f/s), %d triples", done, len(seeds), rate, len(triples))

    path = write_triples(triples)
    log.info("wrote %d triples to %s", len(triples), path)
    print(f"{len(triples)} triples -> {triples_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
