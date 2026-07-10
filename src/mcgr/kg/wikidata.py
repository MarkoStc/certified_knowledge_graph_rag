"""Wikidata claim fetcher for the 2WikiMultiHopQA KG (AGENTS.md P2).

Fetches entity claims via the public ``wbgetentities`` API in batches of 50,
caches each raw response to ``$SCRATCH/data/wikidata_cache`` (a reproducible
snapshot), and extracts entity-valued triples ``(qid, pid, qid)``. Only
entity-to-entity statements are kept — literal-valued claims (dates, strings,
external ids) are not graph edges.

The API is public and needs no token; be a polite client (descriptive
User-Agent, batched requests). See https://www.wikidata.org/w/api.php.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "mcgr-research/0.1 (Menger-Certified Graph-RAG; certificate experiments)"
BATCH = 50


def cache_dir() -> Path:
    base = os.environ.get("MCGR_DATA_ROOT") or (Path(os.environ["SCRATCH"]) / "data")
    d = Path(base) / "wikidata_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _batch_path(batch_key: str) -> Path:
    return cache_dir() / f"{batch_key}.json"


def _fetch_batch(ids: list[str], *, props: str = "claims", retries: int = 8, pause: float = 1.0):
    """Fetch ``props`` for up to 50 ids, returning the ``entities`` mapping.

    Handles Wikidata rate limiting: honours a ``Retry-After`` header on HTTP
    429/503 and otherwise backs off exponentially (capped). ``maxlag`` asks
    the API to defer when replication lag is high (polite-client protocol).
    """
    params = urllib.parse.urlencode(
        {
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": props,
            "format": "json",
            "maxlag": "5",
        }
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            if "entities" in payload:
                return payload["entities"]
            # maxlag / throttling surfaces as an error body, not an HTTP error
            last_err = RuntimeError(payload.get("error", "no entities in response"))
            time.sleep(min(pause * (2**attempt), 60.0))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else None
                time.sleep(wait if wait is not None else min(pause * (2**attempt), 60.0))
            else:
                time.sleep(min(pause * (2**attempt), 60.0))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(min(pause * (2**attempt), 60.0))
    raise RuntimeError(f"wbgetentities failed for {ids[:3]}...: {last_err}")


def get_entities(ids: list[str], *, use_cache: bool = True, polite_pause: float = 0.1) -> dict:
    """Claims for ``ids`` (cached per batch). Returns {qid: entity-json}.

    A short pause follows each *network* fetch (not cache hits) to stay well
    under Wikidata's rate limit; cached re-runs are full-speed.
    """
    out: dict = {}
    unique = list(dict.fromkeys(ids))
    for i in range(0, len(unique), BATCH):
        chunk = unique[i : i + BATCH]
        key = f"{chunk[0]}_{len(chunk)}_{hash(tuple(chunk)) & 0xFFFFFFFF:08x}"
        path = _batch_path(key)
        if use_cache and path.exists():
            out.update(json.loads(path.read_text()))
            continue
        entities = _fetch_batch(chunk)
        path.write_text(json.dumps(entities))
        out.update(entities)
        time.sleep(polite_pause)
    return out


def get_labels(ids: list[str], *, lang: str = "en", use_cache: bool = True) -> dict[str, str]:
    """English labels for Q-ids and/or P-ids (cached per batch). Missing
    labels fall back to the id itself so callers always get a string."""
    labels: dict[str, str] = {}
    unique = list(dict.fromkeys(ids))
    for i in range(0, len(unique), BATCH):
        chunk = unique[i : i + BATCH]
        key = f"labels_{chunk[0]}_{len(chunk)}_{hash(tuple(chunk)) & 0xFFFFFFFF:08x}"
        path = _batch_path(key)
        if use_cache and path.exists():
            labels.update(json.loads(path.read_text()))
            continue
        entities = _fetch_batch(chunk, props="labels")
        batch = {
            qid: e.get("labels", {}).get(lang, {}).get("value", qid) for qid, e in entities.items()
        }
        path.write_text(json.dumps(batch))
        labels.update(batch)
        time.sleep(0.1)
    return {qid: labels.get(qid, qid) for qid in unique}


def entity_triples(entity: dict) -> list[tuple[str, str, str]]:
    """Entity-valued ``(subject_qid, property_pid, object_qid)`` triples."""
    subj = entity.get("id")
    triples: list[tuple[str, str, str]] = []
    for pid, statements in entity.get("claims", {}).items():
        for st in statements:
            snak = st.get("mainsnak", {})
            if snak.get("datatype") != "wikibase-item" or snak.get("snaktype") != "value":
                continue
            obj = snak.get("datavalue", {}).get("value", {}).get("id")
            if obj:
                triples.append((subj, pid, obj))
    return triples
