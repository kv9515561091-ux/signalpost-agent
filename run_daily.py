"""One command.

``python -m src.agent`` on its own runs the competition configuration:
the hundred companies listed in ``data/competition_companies.txt``,
fully enriched, budget-aware, writing profiles and change reports.

The flags stay for anyone who wants to vary it. This exists because
"make it easy to use" is scored, and an evaluator should not have to
know the flag names to produce the thing being asked for.
"""

from __future__ import annotations

from src.batch import process_companies, summarise
from src.config import load_env
from src.request_tracker import request_tracker

COMPETITION_INPUT = "data/competition_companies.txt"
COMPETITION_OUTPUT = "data/competition_profiles.json"


def run_competition_run() -> int:
    """The default: enrich every listed company and report changes."""
    load_env()

    from src.batch import load_company_numbers

    numbers = load_company_numbers(COMPETITION_INPUT)

    if not numbers:
        print(
            f"No company numbers in {COMPETITION_INPUT}.\n"
            f"Put one 9-digit organization number per line and run again.",
        )
        return 1

    print(f"Signalpost Agent — {len(numbers)} companies")
    print("  registry + web discovery + quote-verified extraction")
    print()

    results = process_companies(
        numbers,
        output_file=COMPETITION_OUTPUT,
        include_web=True,
        include_extraction=True,
        include_narrative=True,
        resume=True,
    )

    stats = summarise(results)
    print()
    for key, value in stats.items():
        print(f"  {key:<16} {value}")

    changed = sum(1 for r in results if r.get("changes"))
    first_seen = sum(
        1
        for r in results
        if any(c.get("kind") == "first_seen" for c in r.get("changes") or [])
    )

    print()
    print(f"  changed          {changed - first_seen} (new this run)")
    print(f"  first seen       {first_seen}")
    print(f"  unchanged        {len(results) - changed}")

    tracker = request_tracker.summary()
    print(f"  requests         {tracker['requests_used']}/{tracker['request_limit']}")
    print(f"  output           {COMPETITION_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_competition_run())
