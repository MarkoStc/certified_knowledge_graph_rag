"""LLM reasoner over retrieved KG triples (AGENTS.md P6).

Given a question and a set of context triples (the retrieved subgraph), the
model produces a short answer. Uses HF transformers; models are loaded from
``$SCRATCH/models`` (downloaded in P1). vLLM can replace this later for
throughput, behind the same ``answer_batch`` interface.
"""

import os
from dataclasses import dataclass
from pathlib import Path

Triple = tuple[str, str, str]

_SYSTEM = (
    "You answer multi-hop questions using ONLY the provided knowledge-graph "
    "triples. Reply with the answer entity and nothing else. If the triples "
    "do not support an answer, reply 'unknown'."
)


def models_root() -> Path:
    base = os.environ.get("MCGR_MODELS_ROOT") or (Path(os.environ["SCRATCH"]) / "models")
    return Path(base)


def format_prompt(question: str, triples: list[Triple]) -> str:
    lines = [f"({s}, {r}, {o})" for s, r, o in triples]
    facts = "\n".join(lines) if lines else "(no facts retrieved)"
    return f"Knowledge graph triples:\n{facts}\n\nQuestion: {question}\nAnswer:"


@dataclass
class Reasoner:
    """Thin wrapper over an instruct model with chat templating."""

    model_name: str = "Qwen2.5-7B-Instruct"
    max_new_tokens: int = 32
    _model: object = None
    _tokenizer: object = None

    def load(self) -> "Reasoner":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(models_root() / self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        return self

    def _render(self, question: str, triples: list[Triple]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": format_prompt(question, triples)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def answer_batch(self, items: list[tuple[str, list[Triple]]]) -> list[str]:
        """Answer a batch of (question, triples). Returns stripped strings."""
        import torch

        if self._model is None:
            raise RuntimeError("call .load() first")
        prompts = [self._render(q, t) for q, t in items]
        tok = self._tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = self._model.generate(**tok, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, tok["input_ids"].shape[1] :]
        return [t.strip() for t in self._tokenizer.batch_decode(gen, skip_special_tokens=True)]
