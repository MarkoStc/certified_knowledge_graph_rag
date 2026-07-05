#!/usr/bin/env python3
"""Report AGENTS.md §3 prerequisite status (P0 gate).

Exit 0 iff every item is present. Human-required items are flagged so
the agent knows what to escalate vs what to fetch itself.

Run: uv run python scripts/check_prereqs.py
"""

import sys

from mcgr.prereqs import run_all_checks


def main() -> int:
    results = run_all_checks()
    width = max(len(r.item) for r in results)
    failures = 0
    for r in results:
        mark = "OK  " if r.ok else "MISS"
        tag = " [HUMAN-REQUIRED]" if (not r.ok and r.human_required) else ""
        print(f"{mark} {r.item:<{width}}  {r.detail}{tag}")
        failures += not r.ok
    print(f"\n{len(results) - failures}/{len(results)} prerequisites satisfied")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
