#!/usr/bin/env python3
"""GH200 inference smoke test (AGENTS.md §3.4 gate).

Loads a Qwen model and answers two hand-built KG questions, proving the
LLM stack works on the GPU before anything is built on top. Run via srun:

    srun --gres=gpu:1 ... uv run --extra llm python scripts/smoke_reasoner.py
"""

import sys

from mcgr.models.reasoner import Reasoner


def main() -> int:
    import torch

    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    reasoner = Reasoner(model_name="Qwen2.5-7B-Instruct").load()
    print("model loaded", flush=True)

    items = [
        (
            "Who is the mother of the director of the film X?",
            [("X", "director", "Bob"), ("Bob", "mother", "Alice")],
        ),
        (
            "What language is the film Y in?",
            [("Y", "in_language", "Greek")],
        ),
    ]
    answers = reasoner.answer_batch(items)
    for (q, _), a in zip(items, answers, strict=True):
        print(f"Q: {q}\nA: {a}\n", flush=True)

    ok = "alice" in answers[0].lower() and "greek" in answers[1].lower()
    print("SMOKE PASS" if ok else "SMOKE FAIL (unexpected answers)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
