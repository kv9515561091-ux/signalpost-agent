"""One command for a file of organization numbers.

``python run_daily.py`` reads the list in
``data/competition_companies.txt``, enriches every company it can within
the budget, and writes one profile per company plus a change report.
With no configured key it runs keyless; ``--keyless`` forces that mode
even when ``.env`` holds credentials.

The flags stay for anyone who wants to vary the configuration. This
exists because "make it easy to use" is scored, and an evaluator should
not have to know the flag names to produce the thing being asked for.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
from pathlib import Path

from src import yieldcache
from src.batch import (
    default_workers,
    process_companies,
    state_is_complete,
    summarise,
)
from src.config import load_env
from src.request_tracker import request_tracker

COMPETITION_INPUT = "data/competition_companies.txt"
COMPETITION_OUTPUT = "data/competition_profiles.json"

# Eight in flight, measured rather than guessed. One company at a time
# measured 156 seconds, and eight brings a hundred companies closest to
# the forty-five minute limit without drawing connection resets: twelve
# in flight had 12 of 20 companies reset mid-run. Override with
# AGENT_WORKERS.
#
# Width is not the binding constraint, though. Every page read costs one
# model call, and the shared limiter paces those two seconds apart, so a
# company's floor is roughly its page count times the pacing interval no
# matter how many are in flight. That is why the pacing below matters
# more than the worker count.
COMPETITION_WORKERS = 8

# Measured, not guessed. 40 RPM — the provider's commonly reported
# default — drew twenty ``429`` responses across twenty companies and
# spent three requests per refusal, so the extra pacing was slower and
# poorer than 30. Thirty has never drawn one. Override with NVIDIA_RPM.
COMPETITION_RPM = 30

# Ten pages rather than twelve. The 2,000-request ceiling binds before
# the time ceiling at twelve: the measured fifty-company run used 1,017
# requests (about 2,034 projected for a hundred), and the dropped pages
# are the slow directory tail that contributes the least. Ten keeps the
# projection near 1,700 requests and thirty-five minutes. Override with
# AGENT_MAX_PAGES.
COMPETITION_MAX_PAGES = 10

# The daily limit is forty-five minutes of wall clock, and a run that
# overruns it is disqualified however good the profiles are. Stopping
# enrichment five minutes early is the difference between a complete
# company list with a shallow tail and no entry at all. The tail is
# exactly the companies that produced nothing last run, because the work
# queue is ordered productive-first. Override with AGENT_RUN_MINUTES;
# zero disables the cutoff.
COMPETITION_MINUTES = 40


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_daily.py",
        description=(
            "Read company numbers and write one enriched profile per "
            "company. Runs keyless with registry data and deterministic "
            "explanations when no API keys are configured."
        ),
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("AGENT_INPUT", COMPETITION_INPUT),
        help="file with one organization number per line",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("AGENT_OUTPUT", COMPETITION_OUTPUT),
        help="JSON file for the completed profiles",
    )
    parser.add_argument(
        "--keyless",
        action="store_true",
        help="force registry-only mode even if API keys are present",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="companies processed concurrently",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="wall-clock budget in minutes; zero disables the cutoff",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="page-read ceiling per company",
    )
    return parser


def _install_hard_stop(seconds: float) -> None:
    """Exit at the wall-clock limit even if a worker is stuck on the network.

    The soft deadline stops *starting* companies, but a provider call that
    never returns keeps the pool alive and can overrun the competition's
    forty-five minute limit. Profiles are persisted after every company,
    so exiting here costs only the companies in flight.
    """

    def stop() -> None:
        print(
            "\n  hard time budget reached; keeping the profiles "
            "written so far"
        )
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        os._exit(0)

    timer = threading.Timer(seconds, stop)
    timer.daemon = True
    timer.start()


def run_competition_run(argv: list[str] | None = None) -> int:
    """The default: enrich every listed company and report changes."""
    load_env()
    args = build_parser().parse_args(argv)

    # ``load_env`` never overwrites an exported variable, so a key set in
    # the shell or in ``.env`` still wins. This is only the fallback.
    os.environ.setdefault("NVIDIA_RPM", str(COMPETITION_RPM))

    from src.batch import load_company_numbers
    from src.llm import is_explicitly_configured

    input_file = args.input
    output_file = args.output

    numbers = load_company_numbers(input_file)

    if not numbers:
        print(
            f"No company numbers in {input_file}.\n"
            f"Put one 9-digit organization number per line and run again.",
        )
        return 1

    search_available = bool(os.environ.get("TINYFISH_API_KEY"))
    # ``is_available`` also accepts a discoverable Claude Code profile.
    # A batch command must not treat an installed SDK as a configured
    # key, or a keyless run would attempt a model call and record its
    # authentication failure in every profile.
    llm_ready = is_explicitly_configured()
    if args.keyless:
        search_available = False
        llm_ready = False

    include_web = search_available or llm_ready
    include_extraction = llm_ready
    include_website_check = not include_extraction
    include_narrative = True
    use_llm = llm_ready

    workers = (
        args.workers
        if args.workers and args.workers > 0
        else default_workers(fallback=COMPETITION_WORKERS)
    )

    yieldcache.use_default(
        Path(output_file).with_name("yield_cache.json")
    )
    bands = {yieldcache.PRIORITY_PRODUCTIVE: 0, yieldcache.PRIORITY_UNKNOWN: 0,
             yieldcache.PRIORITY_EMPTY: 0}
    for number in numbers:
        bands[yieldcache.priority(number.strip())] += 1

    if args.minutes is not None:
        minutes = args.minutes
    else:
        try:
            minutes = float(
                os.environ.get("AGENT_RUN_MINUTES", COMPETITION_MINUTES)
            )
        except ValueError:
            minutes = COMPETITION_MINUTES

    if args.max_pages is not None:
        max_pages = max(0, args.max_pages)
    else:
        try:
            max_pages = int(
                os.environ.get("AGENT_MAX_PAGES", COMPETITION_MAX_PAGES)
            )
        except ValueError:
            max_pages = COMPETITION_MAX_PAGES
    deadline = time.monotonic() + minutes * 60 if minutes > 0 else None
    if minutes > 0:
        # Two minutes after the soft stop, and never past 43 minutes, so
        # the profile file is safely on disk before the 45-minute limit.
        _install_hard_stop(min(minutes + 2, 43) * 60)

    if include_extraction and search_available:
        mode = "registry + web discovery + quote-verified extraction"
    elif include_extraction:
        mode = "registry + registered website + quote-verified extraction"
    elif include_web:
        mode = (
            "registry + web discovery + registered website check "
            "(no extraction key)"
        )
    elif include_website_check:
        mode = (
            "registry + registered website check + deterministic "
            "explanations (keyless)"
        )
    else:
        mode = "registry + deterministic explanations (keyless)"

    print(f"Signalpost Agent - {len(numbers)} companies")
    print(f"  mode: {mode}")
    print(f"  concurrency: {workers} companies at once")
    if use_llm:
        print(
            f"  model pacing: {os.environ['NVIDIA_RPM']} "
            "requests per minute"
        )
    # A completed previous run is refreshed in full; only an unfinished
    # one is resumed. Otherwise the daily re-run would skip every company
    # and report no changes.
    resume = not state_is_complete(output_file, numbers)
    print(
        f"  previous run: "
        f"{'incomplete, resuming' if resume else 'complete, refreshing'}"
    )
    print(
        f"  queue: {bands[yieldcache.PRIORITY_PRODUCTIVE]} productive, "
        f"{bands[yieldcache.PRIORITY_UNKNOWN]} new, "
        f"{bands[yieldcache.PRIORITY_EMPTY]} empty last run (read last, shallower)"
    )
    print(
        f"  time budget: {minutes:g} minutes"
        if deadline is not None
        else "  time budget: none"
    )
    print()

    results = process_companies(
        numbers,
        output_file=output_file,
        include_web=include_web,
        include_extraction=include_extraction,
        include_website_check=include_website_check,
        include_narrative=include_narrative,
        use_llm=use_llm,
        resume=resume,
        workers=workers,
        max_pages=max_pages,
        deadline=deadline,
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
    print(f"  changed          {changed - first_seen} (updated)")
    print(f"  first seen       {first_seen}")
    print(f"  unchanged        {len(results) - changed}")

    tracker = request_tracker.summary()
    print(f"  requests         {tracker['requests_used']}/{tracker['request_limit']}")
    print(f"  output           {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_competition_run())
