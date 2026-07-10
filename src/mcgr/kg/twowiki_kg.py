"""2WikiMultiHopQA Wikidata KG (AGENTS.md P2).

Nodes are Wikidata Q-ids (canonical, avoiding 2Wiki's name ambiguity). The
graph is the union of entity-valued claims fetched for every entity that
appears in a compositional/inference gold chain, cached as a triples
edgelist under ``$SCRATCH/data/2wiki_kg``. Built by
``scripts/build_2wiki_kg.py``; loaded here for certification.
"""

import os
from functools import cache
from pathlib import Path

import networkx as nx

from mcgr.kg.graph_store import build_graph, prune_hubs

# Degree above which a node is treated as a type/attribute hub (human, male,
# a country, ...) rather than a specific reasoning entity, and dropped to
# enforce evidential independence. See prune_hubs. Tunable per snapshot.
HUB_MAX_DEGREE = 200


def kg_dir() -> Path:
    base = os.environ.get("MCGR_DATA_ROOT") or (Path(os.environ["SCRATCH"]) / "data")
    return Path(base) / "2wiki_kg"


def triples_path() -> Path:
    return kg_dir() / "triples.tsv"


def seed_qids(splits: tuple[str, ...] = ("train", "dev")) -> tuple[list[str], list[tuple]]:
    """Return (unique seed Q-ids, per-query (qid, anchor, answer) tuples) for
    compositional/inference 2Wiki questions with Wikidata annotations."""
    import json

    data_root = os.environ.get("MCGR_DATA_ROOT") or (Path(os.environ["SCRATCH"]) / "data")
    seeds: dict[str, None] = {}
    queries: list[tuple] = []
    for split in splits:
        path = Path(data_root) / "2wikimultihopqa" / "extracted" / f"{split}.json"
        for r in json.loads(path.read_text()):
            if r.get("type") not in ("compositional", "inference"):
                continue
            eid = r.get("evidences_id") or []
            ans = r.get("answer_id")
            if not eid or not ans:
                continue
            objs = {o for _, _, o in eid}
            anchors = [s for s, _, _ in eid if s not in objs]
            anchor = anchors[0] if anchors else eid[0][0]
            for s, _, o in eid:
                seeds[s] = None
                seeds[o] = None
            seeds[ans] = None
            queries.append((r["_id"], split, anchor, ans))
    return list(seeds), queries


def write_triples(triples: list[tuple[str, str, str]]) -> Path:
    kg_dir().mkdir(parents=True, exist_ok=True)
    path = triples_path()
    with path.open("w") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    return path


def load_triples() -> list[tuple[str, str, str]]:
    with triples_path().open() as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


@cache
def twowiki_graph_and_hubs(max_degree: int = HUB_MAX_DEGREE) -> tuple[nx.Graph, dict]:
    """The cached 2Wiki Wikidata KG (hubs pruned for transit) plus the
    ``{hub: neighbors}`` adjacency needed to restore a hub as a query
    endpoint. See graph_store.prune_hubs / reconnect_endpoints."""
    graph = build_graph(load_triples())
    return prune_hubs(graph, max_degree)


def twowiki_graph(max_degree: int = HUB_MAX_DEGREE) -> nx.Graph:
    """The pruned 2Wiki Wikidata KG (transit graph)."""
    return twowiki_graph_and_hubs(max_degree)[0]


@cache
def twowiki_pair_index() -> dict:
    """Map each unordered Q-id pair to its directed (s, pid, o) triple(s)."""
    index: dict = {}
    for s, p, o in load_triples():
        index.setdefault(frozenset((s, o)), []).append((s, p, o))
    return index


@cache
def twowiki_relation_objects() -> dict:
    """Map each property (P-id) to the tuple of its object Q-ids, for picking
    a type-consistent sibling wrong answer under the same relation."""
    objs: dict = {}
    for _s, p, o in load_triples():
        objs.setdefault(p, []).append(o)
    return {p: tuple(dict.fromkeys(v)) for p, v in objs.items()}
