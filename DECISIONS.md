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
- **2026-07-06 — MuSiQue gold paths are the structured-hop subset only.**
  ~34% of decomposition hops are natural-language sub-questions, not the
  `entity >> relation` template, so they don't yield clean triples. Loader
  keeps the full raw decomposition in `meta['decomposition']`; nothing lost.
- **2026-07-06 — Deletion certificate uses an UNDIRECTED fact graph.** The
  deletion adversary removes *facts*; a fact is one undirected edge, and
  edge-disjoint anchor→answer paths (Menger's classic undirected form) count
  independent fact-chains. Matches §7 (unit-capacity edges → max-flow =
  min-cut). Directedness is deferred to the harder P5 insertion certificate.
- **2026-07-06 — Parallel facts between the same entity pair are collapsed to
  one edge.** NetworkX flow needs simple graphs; collapsing can only *lower*
  the path count, so the certificate is conservative (under-certifies) — the
  safe direction for a robustness guarantee, per the brief's conservative
  stance.
- **2026-07-06 — Certificate is computed on the anchor's k-hop subgraph** (P2
  bounded neighborhood), radius = question hop count. Open subtlety to revisit
  in P5: edge-disjoint paths inside the ball may be longer than `hops` edges,
  so k partly reflects local KB connectivity, not only question-length
  reasoning. Flagged, not yet resolved.
- **2026-07-06 — Multi-anchor queries use a super-source with infinite-
  capacity edges** to each anchor, so the min cut falls only on real facts =
  min facts to disconnect the answer from the whole anchor set. Reduces to the
  single-pair count for one anchor.
