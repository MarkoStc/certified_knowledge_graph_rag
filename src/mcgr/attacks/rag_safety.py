"""RAG-Safety-style triple-insertion attack (AGENTS.md P6).

Threat model (deletion's dual): the adversary *inserts* up to ``budget``
triples to forge a competing reasoning chain from the query anchor to a
WRONG answer, aiming to make the reasoner output it instead of the gold
answer. Insertions are deterministic under ``seed``.

This is the empirical stressor for the P7 gate: the deletion certificate
``k`` measures how much genuine redundant support the gold answer has, and
the hypothesis (H1) is that higher-``k`` queries resist insertion better —
a forged single chain competes poorly against many independent real ones.
"""

import hashlib
from dataclasses import dataclass

Triple = tuple[str, str, str]


def _rng_index(seed: int, *parts: str, modulo: int) -> int:
    """Deterministic index in [0, modulo) from seed + string parts."""
    h = hashlib.sha256(("|".join((str(seed), *parts))).encode()).hexdigest()
    return int(h, 16) % max(modulo, 1)


@dataclass(frozen=True)
class InsertionAttack:
    budget: int
    seed: int = 0
    # plausible real entity names used as forged intermediates; a realistic
    # RAG-Safety attack does not use obviously-synthetic ids (those are
    # trivially ignored by the reasoner). Empty -> fall back to synthetic tags.
    entity_pool: tuple[str, ...] = ()

    def forge_chain(self, anchor: str, gold_chain: list[Triple], wrong_answer: str) -> list[Triple]:
        """Forge a competing chain anchor -> ... -> wrong_answer.

        Mirrors the gold chain's relation sequence with plausible
        intermediates so the forged path is as convincing as a real one. Uses
        up to ``budget`` triples: the first cover the chain to
        ``wrong_answer``; any remainder add corroborating duplicates of the
        final (…, wrong_answer) hop from other plausible intermediates, which
        is what raises an insertion attack's strength.
        """
        if self.budget <= 0 or not gold_chain:
            return []
        relations = [r for _, r, _ in gold_chain]
        n = len(relations)
        inserted: list[Triple] = []

        # forge the primary competing chain (length = min(budget, n))
        prev = anchor
        chain_len = min(self.budget, n)
        for i in range(chain_len):
            rel = relations[i]
            is_last = i == chain_len - 1
            nxt = wrong_answer if is_last else self._fake_entity(anchor, wrong_answer, i)
            inserted.append((prev, rel, nxt))
            prev = nxt

        # spend any leftover budget on corroborating final-hop assertions
        final_rel = relations[-1]
        for j in range(self.budget - chain_len):
            fake_mid = self._fake_entity(anchor, wrong_answer, n + j)
            inserted.append((fake_mid, final_rel, wrong_answer))
        return inserted

    def _fake_entity(self, anchor: str, wrong_answer: str, i: int) -> str:
        idx = _rng_index(self.seed, anchor, wrong_answer, str(i), modulo=len(self.entity_pool) or 1)
        if self.entity_pool:
            return self.entity_pool[idx]
        tag = _rng_index(self.seed, anchor, wrong_answer, str(i), modulo=1_000_000)
        return f"__forged_{tag:06d}__"

    def apply(
        self, context: list[Triple], anchor: str, gold_chain: list[Triple], wrong_answer: str
    ) -> list[Triple]:
        """Return the attacked context: original triples plus the forged
        chain. Order is deterministic (forged triples appended)."""
        return [*context, *self.forge_chain(anchor, gold_chain, wrong_answer)]
