# BLOCKED — [HUMAN-REQUIRED] items for Marko (AGENTS.md §3 / §10)

`scripts/check_prereqs.py` now passes 15/16; only Freebase (§3.3) remains.

## 1. Hugging Face token (§3.1) — ✅ RESOLVED 2026-07-11
Token in `~/.config/hf/token`, detected by check_prereqs.

## 2. Llama-3.1-8B-Instruct license (§3.4) — ✅ RESOLVED 2026-07-11
License accepted; model downloaded to `$SCRATCH/models/Llama-3.1-8B-Instruct`
(15G, 4 shards). Being added as a third reasoner across P7/P8/P10.

## 3. Freebase SPARQL endpoint (§3.3) — STILL BLOCKED
Blocks: WebQSP and CWQ (KG construction + their experiments). The datasets
are downloaded, but they are grounded in Freebase, which needs a running
Virtuoso endpoint — the biggest setup burden and most likely to need a human.
Do: stand up the Freebase-Setup Virtuoso dump and give the agent a reachable
URL (set `FREEBASE_ENDPOINT` or `context/manifest.yaml:freebase_endpoint`).
Fallback if infeasible (§9): lead entirely with 2WikiMultiHopQA + MetaQA
(self-contained, already working) and treat WebQSP/CWQ as secondary.

None of these block the current critical path: the certificate core runs on
MetaQA (and next 2Wiki) with zero dependency on the above.
