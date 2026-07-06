# BLOCKED — [HUMAN-REQUIRED] items for Marko (AGENTS.md §3 / §10)

Everything else is proceeding. These three need a human and gate specific
parts of the full sweep. `scripts/check_prereqs.py` reports them (13/16 pass;
these are the 3 misses).

## 1. Hugging Face token (§3.1)
Blocks: downloading the gated Llama-3.1-8B model.
Do: create a read token at huggingface.co → Settings → Access Tokens, then
```bash
mkdir -p ~/.config/hf && printf 'hf_xxx' > ~/.config/hf/token   # NOT in the repo
# or: echo 'HF_TOKEN=hf_xxx' >> .env   (.env is gitignored)
```

## 2. Llama-3.1-8B-Instruct license (§3.4)
Blocks: the second reasoner model (cross-model robustness of results).
Do: accept the license once on
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (needs the token
above). The agent then downloads it. Until then, work proceeds with the two
Qwen2.5 models (already downloaded).

## 3. Freebase SPARQL endpoint (§3.3)
Blocks: WebQSP and CWQ (KG construction + their experiments). The datasets
are downloaded, but they are grounded in Freebase, which needs a running
Virtuoso endpoint — the biggest setup burden and most likely to need a human.
Do: stand up the Freebase-Setup Virtuoso dump and give the agent a reachable
URL (set `FREEBASE_ENDPOINT` or `context/manifest.yaml:freebase_endpoint`).
Fallback if infeasible (§9): lead entirely with 2WikiMultiHopQA + MetaQA
(self-contained, already working) and treat WebQSP/CWQ as secondary.

None of these block the current critical path: the certificate core runs on
MetaQA (and next 2Wiki) with zero dependency on the above.
