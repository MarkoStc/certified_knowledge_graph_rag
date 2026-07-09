# Threat model (AGENTS.md P5)

This is a specification, not paper prose. It states, conservatively and
explicitly, the adversary and the guarantees the certificates provide. The
guiding principle (§9 risk register): **overclaiming here is the most likely
reviewer attack, so every assumption is named and the bounds are kept
conservative.**

## 1. Objects

- A query has an **anchor set** `A` (entities named in the question) and a
  **candidate answer set** `C` (the entities the system could return).
- The **evidence graph** `G` is the bounded k-hop neighbourhood of `A` in the
  knowledge graph, undirected, one edge per KB triple (a *fact*). Parallel
  facts between the same pair are collapsed (conservative — can only lower
  path counts).
- For a candidate `a`, `paths(a)` = the number of edge-disjoint `A -> a`
  paths in `G` (max-flow with a super-source over `A`, unit edge capacities;
  Menger).

## 2. Decision-rule abstraction

We certify a **structural surrogate** of the reasoner, not the LLM directly
(we make no claim to prove properties of a neural network): the surrogate
returns the candidate with the most edge-disjoint anchor->answer support,

    decide(G) = argmax_{a in C} paths(a),  ties broken adversarially.

**Assumption A1 (plurality grounding).** The reasoner's answer is governed by
the amount of independent supporting evidence in the retrieved graph. The P7
experiments provide empirical backing (robustness rises monotonically with
`paths(a) - 1`), but A1 is an assumption, not a theorem, and every guarantee
below is *conditional on A1*.

## 3. Deletion adversary (P4)

- **Power.** Remove up to `k` edges (facts) from `G`.
- **Certificate.** `k(a) = paths(a) - 1`.
- **Guarantee (Menger, unconditional on the graph).** No deletion of `<= k(a)`
  edges disconnects `a` from `A`: the min `A`-`a` edge cut equals `paths(a)`.
  This part needs no assumption about the reasoner — it is a pure connectivity
  statement. A1 only enters when translating "still connected" into "still
  answered".

## 4. Insertion adversary (P5)

- **Power.** Insert up to `b` edges (forged facts). An inserted edge may join
  any two entities, including fresh forged intermediates. The adversary aims
  to make some wrong answer `a' != a_true` satisfy `paths(a') >= paths(a_true)`
  so that `decide` may return `a'`.

- **Assumption A2 (one path per insertion).** Each inserted edge raises the
  edge-disjoint `A -> a'` path count of at most one competitor by at most one.
  This is the adversary's *most efficient* case — a direct forged edge
  `A -> a'` is one insertion and one new edge-disjoint path — so bounding the
  adversary by A2 is conservative: any real adversary needing multi-edge
  chains does *worse*, i.e. our certified budget is a lower bound on the true
  tolerated budget.

- **Assumption A3 (competitor set).** Competitors are drawn from an explicit
  set `C' ⊆ C` of plausible wrong answers (e.g. type-consistent siblings under
  the query's final relation). A guarantee is only against `C'`; a wrong
  answer outside `C'` is out of scope and must be stated as such. Widening
  `C'` can only lower the certificate (more competitors to dominate), so a
  narrow `C'` is an *over*-claim — pick `C'` as broad as is defensible.

- **Certificate.**

    b_ins(a) = paths(a) - max_{a' in C'} paths(a') - 1 .

- **Guarantee (conditional on A1–A3).** For every `b <= b_ins(a)` and every
  insertion of `<= b` edges, `paths(a) > paths(a')` still holds for all
  `a' in C'`, so `decide` still returns `a`. Proof: by A2, `b` insertions
  raise `max_{a'} paths(a')` by at most `b`; since
  `paths(a) - max_{a'} paths(a') = b_ins(a) + 1 > b`, the strict inequality is
  preserved.

## 5. Relationship to deletion, and honesty caveats

- `b_ins(a) = paths(a) - max_{a'} paths(a') - 1` **≤** `k(a) = paths(a) - 1`,
  with equality iff no competitor in `C'` has any support. So the insertion
  certificate is never more permissive than deletion and is strictly tighter
  whenever a real competitor exists — it is *not* a relabelling of `k`.

- **Structural ≠ evidential independence.** Edge-disjoint paths may still share
  a corrupted upstream source; `paths(a)` counts structural, not evidential,
  independence. The 2Wiki hub-pruning is a partial control; a full treatment
  is open (§9 risk register). Where the two can diverge, the certificate is an
  upper bound on evidential redundancy and should be read as such.

- **A1 is the load-bearing assumption.** The certificates are graph-theoretic
  facts; their relevance to a specific reasoner rests on A1. We report A1's
  empirical support (P7) rather than asserting it.
