# DECISIONS.md — non-obvious choices + rationale

- **2026-07-05 — Python 3.12 via uv.** System python3 is 3.6 (unusable);
  3.12.13 was already in the uv cache (no download). `requires-python >=3.12`.
- **2026-07-05 — Package named `mcgr`, repo keeps its GitHub name.** The
  remote is `certified_knowledge_graph_rag`; AGENTS.md §4 shows root
  `menger-certified-graphrag/`. The layout *contents* are authoritative;
  renaming the GitHub repo is Marko's call.
- **2026-07-06 — Deletion certificate convention: k = −1 for unsupported
  answers.** §7 defines k = (#edge-disjoint paths) − 1; with zero paths this
  yields −1, which we keep (distinguishes "no support at all" from "single
  fragile chain", k = 0). Metrics treat any k < 0 as uncertified.
- **2026-07-06 — Config/seed/log plumbing is stdlib + PyYAML only.** No
  hydra/omegaconf: configs are flat YAML files loaded into a frozen dataclass;
  fewer deps, fully lockable, sufficient for a config-per-experiment pattern.
- **2026-07-06 — wandb deferred until first real experiment.** P0 sets up the
  CSV/JSONL local logging backup (§2 requires it regardless); the wandb dep
  and API key land with the first training/eval run to keep the lockfile
  minimal until needed.
