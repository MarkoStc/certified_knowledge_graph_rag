"""Text-constructed KG from HotpotQA (AGENTS.md P2/P11).

Shows the certificate is not tied to a curated KG. HotpotQA's context is a
list of Wikipedia articles ``(title, sentences)``; we treat article titles
(and the gold answer) as entities and add an edge whenever one entity's
article mentions another entity — i.e. a supporting sentence links two
entities. Redundancy then comes from multiple independent entity paths, and
the certificate is computed exactly as on the curated KGs.

Extraction is deliberately simple (case-insensitive whole-entity mention);
it is the *method of record*, not a claim of perfect extraction. Parallel
sentence-edges between the same pair are collapsed (conservative).
"""

import re

import networkx as nx

from mcgr.kg.graph_store import build_graph


def _mentions(text: str, entity: str) -> bool:
    if len(entity) < 3:
        return False
    return re.search(rf"\b{re.escape(entity)}\b", text, flags=re.IGNORECASE) is not None


def build_hotpot_kg(record: dict) -> tuple[nx.Graph, list[str], str | None]:
    """Return (co-mention graph, anchor entities, answer node) for one
    HotpotQA record. ``answer`` node is None if the answer is not an entity
    in the context (e.g. yes/no) and the query is not certifiable."""
    context = record.get("context") or record.get("meta", {}).get("context", {})
    titles = list(context["title"])
    sentences = list(context["sentences"])
    answer = record["answer"] if isinstance(record, dict) and "answer" in record else None
    if answer is None:
        answer = record.get("answers", (None,))[0]

    # entity vocabulary: article titles plus the answer (if entity-like)
    vocab = list(dict.fromkeys(titles + ([answer] if answer else [])))

    triples: list[tuple[str, str, str]] = []
    for title, sents in zip(titles, sentences, strict=False):
        blob = " ".join(sents)
        for other in vocab:
            if other != title and _mentions(blob, other):
                triples.append((title, "mentions", other))
    graph = build_graph(triples)

    question = record.get("question", "")
    anchors = [t for t in titles if _mentions(question, t)]
    answer_node = answer if answer in graph else None
    return graph, anchors, answer_node
