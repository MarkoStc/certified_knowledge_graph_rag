# SOURCES.md — provenance of every external artifact

Machine-checkable paths + SHA256 live in `manifest.yaml` (verified by
`scripts/check_prereqs.py`). This file records the human-facing details:
where each artifact came from, which version, and any substitutions.

## Datasets (downloaded 2026-07-06 to `$SCRATCH/data/`)

| Dataset | Source | Version / notes |
|---|---|---|
| 2WikiMultiHopQA | Official Dropbox link from Alab-NII/2wikimultihop README | `data_ids_april7.zip` (April 7 2021 fix; includes `evidences_id`, `answer_id`, `id_aliases.json` — the triple-level gold-path version the brief requires) |
| MetaQA | HF mirror `camazlucas/MetaQA` | kb.txt + 1/2/3-hop vanilla splits. **Substitution:** the official Google Drive folder (yuyuz/MetaQA README) is an old-style `0B-…` id that gdown cannot fetch (401). Mirror content verified: pipe-separated KB triples, bracketed-entity QA format, kb.txt 5.2 MB — matches the original's documented format and sizes. |
| HotpotQA | HF parquet conversion of `hotpotqa/hotpot_qa` | **Substitution:** canonical host `curtis.ml.cmu.edu` unreachable (connection timeout) on 2026-07-06. Parquet is HF's auto-conversion of the official loader. Distractor train/validation + fullwiki validation. |
| MuSiQue | Official StonyBrookNLP Google Drive (id `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`, from `download_data.sh`) | `musique_v1.0.zip` |
| WebQSP | Microsoft Download Center (details.aspx?id=52763) | `WebQSP.zip`, 2016-05 release. Note: the older `F5051A19…` URL cited in third-party repos is dead (404); current URL in manifest. |
| CWQ | Official ComplexWebQuestions Dropbox folder (Talmor & Berant) | v1.1 files: train/dev/test JSON + web snippets |

## Models (downloading 2026-07-06 to `$SCRATCH/models/`)

| Model | HF id | Status |
|---|---|---|
| Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct | ungated, agent-downloaded |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | ungated, agent-downloaded |
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | **BLOCKED: gated — needs Marko's license acceptance + HF token** |
| DeBERTa-v3-large MNLI (NLI checker) | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | ungated, agent-downloaded |

## Freebase (WebQSP/CWQ grounding)

Not yet available. Needs a Virtuoso SPARQL endpoint loaded with the
Freebase-Setup style dump — **[HUMAN-REQUIRED]** to confirm (AGENTS.md §3.3).

## Baseline repos (`third_party/`, cloned 2026-07-06, depth-50)

| Repo | URL | Commit at clone |
|---|---|---|
| RoG | https://github.com/RManLuo/reasoning-on-graphs | `ccf8ec847bf61005a1b27cc9e5aff5c8ead7a24b` |
| SubgraphRAG | https://github.com/Graph-COM/SubgraphRAG | `48eb05e56d11293fa780849512310462c45b956b` |
| GraphRAG (Microsoft) | https://github.com/microsoft/graphrag | `6d02c2355c3fed4c49007572fbe951d73258a37f` |

Attack implementations (RAG-Safety, GragPoison, KEPo): located in P6.

## Wikidata KG snapshot for 2WikiMultiHopQA (P2)

- Fetched **2026-07-07** via the public `wbgetentities` API (props=claims,
  maxlag=5), for all 60,487 entities appearing in the compositional/inference
  gold chains (`evidences_id`) plus answers of train+dev.
- Raw per-batch responses cached under `$SCRATCH/data/wikidata_cache/`;
  extracted entity-valued triples in `$SCRATCH/data/2wiki_kg/triples.tsv`
  (**1,296,716 triples**; graph 366,039 nodes / 1,218,282 edges before hub
  pruning). Rebuild: `scripts/build_2wiki_kg.py` (idempotent — reuses cache).
- Wikidata is public, CC0; no token needed. Snapshot is dated because
  Wikidata is live — the cache is the reproducible pin.

## Papers

`context/papers/` — populated as sources are read and cited (AGENTS.md §11).
