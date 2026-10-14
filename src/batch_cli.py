"""Batch CLI: ``python -m src.batch_cli``.

The argparse body lives here rather than at the repository root so it is
importable, which is what lets ``pyproject.toml`` ship a
``signalpost-batch`` console script. The root ``run_batch.py`` is a thin
shim that preserves the documented ``python run_batch.py`` invocation.
"""

from __future__ import annotations

import argparse

from src.agent import (
    DEFAULT_MAX_CRAWL_LINKS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_SOURCES,
)
from src.batch import (
    DEFAULT_DELAY_SECONDS,
    load_company_numbers,
    process_companies,
    summarise,
)
from src.request_tracker import request_tracker

PROG = "python run_batch.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Build verified company profiles for every organization "
            "number in data/company_numbers.txt. Registry-only by "
            "default."
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Discover and verify public sources (needs a search key).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract facts from verified pages (implies --web, needs an "
             "LLM key). Costs roughly one model call per page.",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Add a written explanation per company.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Never call an LLM, even if credentials exist.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_MAX_SOURCES,
        metavar="N",
        help=f"Sources per company (default {DEFAULT_MAX_SOURCES}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        metavar="N",
        help=f"Pages per company (default {DEFAULT_MAX_PAGES}).",
    )
    parser.add_argument(
        "--max-crawl-links",
        type=int,
        default=DEFAULT_MAX_CRAWL_LINKS,
        metavar="N",
        help=(
            "Same-domain follow-ups read from a verified first-party "
            f"page (default {DEFAULT_MAX_CRAWL_LINKS}). 0 disables the "
            "crawl."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/company_numbers.txt",
        metavar="PATH",
        help="File of organization numbers, one per line.",
    )
    parser.add_argument(
        "--output",
        default="data/company_profiles.json",
        metavar="PATH",
        help="Where to write the resulting JSON.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        metavar="SECONDS",
        help=(
            "Pause between companies. This is not the LLM rate limit; "
            "model calls are paced separately."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    company_numbers = load_company_numbers(args.input)

    if not company_numbers:
        print(f"No company numbers found in {args.input}")
        return 1

    enriched = args.web or args.extract or args.narrate
    use_llm = not args.no_llm

    print(f"Building {len(company_numbers)} company profiles...")

    if enriched:
        modes = [
            name
            for name, on in (
                ("web", args.web),
                ("extract", args.extract),
                ("narrate", args.narrate),
            )
            if on
        ]
        print(f"  enrichment: {', '.join(modes)}")
        if not use_llm:
            print("  LLM calls disabled (--no-llm)")
        if args.extract:
            print(
                f"  caps: {args.max_sources} sources, {args.max_pages} pages, "
                f"{args.max_crawl_links} crawl links per company"
            )

    def report(index: int, total: int, raw: str, record: dict) -> None:
        if "error" in record:
            print(f"  [{index}/{total}] {raw}  ERROR  {record['error']}")
            return

        counts = []
        if record.get("facts"):
            counts.append(f"{len(record['facts'])} facts")
        if record.get("explanation"):
            counts.append("narrated")
        suffix = f"  ({', '.join(counts)})" if counts else ""
        name = record.get("company_name") or raw
        print(f"  [{index}/{total}] {name}{suffix}")

    results = process_companies(
        company_numbers,
        output_file=args.output,
        delay=args.delay,
        include_web=args.web,
        include_extraction=args.extract,
        include_narrative=args.narrate,
        use_llm=use_llm,
        max_sources=args.max_sources,
        max_pages=args.max_pages,
        max_crawl_links=args.max_crawl_links,
        progress=report,
    )

    print()
    for key, value in summarise(results).items():
        print(f"  {key:<16} {value}")

    tracker = request_tracker.summary()
    print(f"  requests        {tracker['requests_used']}/{tracker['request_limit']}")
    print(f"  output          {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
