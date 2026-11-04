"""Contract-compliant batch runner: ``python run_contract.py``.

This wraps the existing research pipeline rather than replacing it.  The
pipeline still writes one rich profile per company; this command adapts
those records to the evaluator's terminal envelope, writes JSONL, and
produces a report that fails loudly when an input is missing.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from src.batch import (
    default_workers,
    process_companies,
)
from src.config import load_env
from src.contract import (
    build_report,
    company_rows,
    envelope_for,
    missing_results,
    read_inputs,
    utc_now,
    write_json,
    write_jsonl,
)
from src.request_tracker import request_tracker

PROG = "run_contract.py"
DEFAULT_INPUT = "data/competition_companies.txt"
DEFAULT_OUTPUT = "data/contract_envelopes.jsonl"
DEFAULT_PROFILES = "data/contract_profiles.json"
DEFAULT_REPORT = "data/contract_report.json"
DEFAULT_WORKERS = 8
DEFAULT_MINUTES = 40.0
AUTO_BULK_MIN = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Run the agent over an organisation-number file and emit one "
            "contract-compliant JSONL envelope per company."
        ),
    )
    parser.add_argument(
        "--input",
        "--organisations",
        dest="input",
        default=os.environ.get("AGENT_INPUT", DEFAULT_INPUT),
        metavar="PATH",
        help="JSON, JSONL or text file of organisation numbers.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("AGENT_OUTPUT", DEFAULT_OUTPUT),
        metavar="PATH",
        help="JSONL file for terminal envelopes.",
    )
    parser.add_argument(
        "--profiles-output",
        default=DEFAULT_PROFILES,
        metavar="PATH",
        help="Rich profile array, kept for inspection and refresh history.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        metavar="PATH",
        help="JSON run report and contract checks.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier; defaults to the UTC start timestamp.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        metavar="N",
        help="Fail the contract report unless exactly N envelopes are written.",
    )
    parser.add_argument(
        "--keyless",
        action="store_true",
        help="Force registry-only mode even if API keys are present.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Companies processed concurrently.",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=DEFAULT_MINUTES,
        metavar="MINUTES",
        help="Wall-clock budget; zero disables the cutoff.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Page-read ceiling per company.",
    )
    parser.add_argument(
        "--bulk",
        default=os.environ.get("AGENT_BULK"),
        metavar="PATH",
        help=(
            "Frozen Brreg CSV or CSV.GZ snapshot, or 'auto'. Skips the "
            "live identity request and makes a 1,000-company run fit the "
            "request ceiling. Runs of 500+ default to auto."
        ),
    )
    parser.add_argument(
        "--underenheter",
        default=os.environ.get("AGENT_UNDERENHETER"),
        metavar="PATH",
        help=(
            "Frozen Brreg underenhet CSV or CSV.GZ snapshot, or 'auto'. "
            "Adds registered workplaces without a live request per "
            "company. Runs of 500+ default to auto."
        ),
    )
    parser.add_argument(
        "--verify-websites",
        action="store_true",
        help="Keep the registry-website fetch when running from --bulk.",
    )
    return parser


def _mode(
    args: argparse.Namespace,
    *,
    bulk: bool = False,
) -> tuple[bool, bool, bool, bool, bool]:
    from src.llm import is_explicitly_configured

    search_available = bool(os.environ.get("TINYFISH_API_KEY"))
    llm_ready = is_explicitly_configured()
    if args.keyless:
        search_available = False
        llm_ready = False

    include_web = search_available or llm_ready
    include_extraction = llm_ready
    verify_websites = getattr(args, "verify_websites", False)
    include_website_check = not include_extraction and (
        not bulk or verify_websites
    )
    include_narrative = True
    return (
        include_web,
        include_extraction,
        include_website_check,
        include_narrative,
        llm_ready,
    )


def _ordered_records(
    inputs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly one record per input, in input order."""
    by_organisation: dict[str, dict[str, Any]] = {}
    for result in results:
        org = str(
            result.get("company_number")
            or result.get("organisation_number")
            or ""
        )
        if org and org not in by_organisation:
            by_organisation[org] = result

    for missing in missing_results(inputs, results):
        by_organisation[missing["company_number"]] = missing

    ordered: list[dict[str, Any]] = []
    for item in inputs:
        org = item["organisation_number"]
        selected = by_organisation.get(org)
        if selected is None:
            selected = {
                "company_number": org,
                "error": "SubmissionError: no profile returned for input",
            }
        record = selected
        record["organisation_number"] = org
        if item.get("evaluation_split") is not None:
            record["evaluation_split"] = item["evaluation_split"]
        if item.get("sample_slice") is not None:
            record["sample_slice"] = item["sample_slice"]
        ordered.append(record)
    return ordered


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)

    try:
        inputs = read_inputs(args.input)
    except (OSError, ValueError) as exc:
        print(f"Invalid input: {exc}")
        return 2

    if not inputs:
        print(f"No company numbers in {args.input}")
        return 2

    expected = (
        args.expected_count
        if args.expected_count is not None
        else len(inputs)
    )
    if args.expected_count is not None and expected != len(inputs):
        print(
            f"Expected {expected} companies, but input contains {len(inputs)}"
        )

    bulk_records: dict[str, dict[str, Any]] | None = None
    bulk_meta: dict[str, Any] | None = None
    bulk_path = args.bulk
    if bulk_path is None and len(inputs) >= AUTO_BULK_MIN:
        bulk_path = "auto"
    if bulk_path:
        from src.bulk import ensure_bulk, load_bulk

        explicit = bulk_path != "auto"
        if bulk_path == "auto":
            try:
                bulk_path = str(ensure_bulk())
            except Exception as exc:  # noqa: BLE001 - live fallback is valid
                print(f"Bulk download unavailable: {exc}")
                bulk_path = None
        if bulk_path:
            try:
                bulk_records, bulk_meta = load_bulk(
                    bulk_path,
                    [item["organisation_number"] for item in inputs],
                    retrieved_at=utc_now(),
                )
            except (OSError, ValueError) as exc:
                print(f"Invalid bulk snapshot: {exc}")
                if explicit:
                    return 2
            else:
                if bulk_meta["missing"]:
                    print(
                        "  bulk misses "
                        f"{len(bulk_meta['missing'])} companies; they will "
                        "use the live identity endpoint"
                    )

    subunit_records: dict[str, list[dict[str, Any]]] | None = None
    subunit_meta: dict[str, Any] | None = None
    subunit_path = args.underenheter
    if subunit_path is None and len(inputs) >= AUTO_BULK_MIN:
        subunit_path = "auto"
    if subunit_path:
        from src.bulk import ensure_underenheter, load_underenheter

        explicit = subunit_path != "auto"
        if subunit_path == "auto":
            try:
                subunit_path = str(ensure_underenheter())
            except Exception as exc:  # noqa: BLE001 - workplaces are optional
                print(f"Underenhet download unavailable: {exc}")
                subunit_path = None
        if subunit_path:
            try:
                subunit_records, subunit_meta = load_underenheter(
                    subunit_path,
                    [item["organisation_number"] for item in inputs],
                    retrieved_at=utc_now(),
                )
            except (OSError, ValueError) as exc:
                print(f"Invalid underenhet snapshot: {exc}")
                if explicit:
                    return 2
            else:
                print(
                    "  workplaces "
                    f"{subunit_meta['workplaces']} across "
                    f"{subunit_meta['selected']} companies"
                )

    (
        include_web,
        include_extraction,
        include_website_check,
        include_narrative,
        use_llm,
    ) = _mode(args, bulk=bool(bulk_records))

    workers = (
        args.workers
        if args.workers is not None and args.workers > 0
        else default_workers(fallback=DEFAULT_WORKERS)
    )
    minutes = max(0.0, args.minutes)
    deadline = time.monotonic() + minutes * 60 if minutes > 0 else None

    run_id = args.run_id or utc_now()
    started_at = utc_now()
    started_clock = time.monotonic()

    print(f"Signalpost contract run - {len(inputs)} companies")
    print(f"  run: {run_id}")
    print(f"  input: {args.input}")
    if bulk_meta:
        print(
            f"  bulk: {bulk_meta['selected']}/{bulk_meta['requested']} "
            f"identities from {bulk_meta['rows_scanned']} rows "
            f"(sha256 {bulk_meta['snapshot_sha256'][:12]})"
        )
    print(f"  concurrency: {workers} companies at once")

    options: dict[str, Any] = {}
    if args.max_pages is not None:
        options["max_pages"] = max(0, args.max_pages)

    results = process_companies(
        [item["organisation_number"] for item in inputs],
        output_file=args.profiles_output,
        include_web=include_web,
        include_extraction=include_extraction,
        include_website_check=include_website_check,
        include_narrative=include_narrative,
        use_llm=use_llm,
        company_records=bulk_records,
        subunit_records=subunit_records,
        resume=False,
        workers=workers,
        deadline=deadline,
        persist_every=25,
        **options,
    )

    completed_at = utc_now()
    runtime_ms = int((time.monotonic() - started_clock) * 1000)
    records = _ordered_records(inputs, results)

    # The pipeline writes after every company. Rewrite its array once in
    # exact input order, including explicit error records for anything
    # that never returned.
    write_json(records, args.profiles_output)

    envelopes = [
        envelope_for(
            record,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
        )
        for record in records
    ]
    write_jsonl(envelopes, args.output)

    tracker = request_tracker.summary()
    report = build_report(
        envelopes,
        expected_count=expected,
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        runtime_ms=runtime_ms,
        requests=tracker["requests_used"],
        input_path=str(Path(args.input)),
    )
    report["request_limit"] = tracker["request_limit"]
    report["workers"] = workers
    report["bulk"] = bulk_meta
    report["underenheter"] = subunit_meta
    report["companies"] = company_rows(envelopes)
    write_json(report, args.report)

    print()
    print(f"  returned         {report['returned_count']}/{expected}")
    print(f"  claims           {report['committed_claims']}")
    print(f"  evidence         {report['committed_evidence']}")
    print(f"  requests         {report['requests']}/{report['request_limit']}")
    print(f"  runtime          {runtime_ms} ms")
    print(f"  contract         {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  envelopes        {args.output}")
    print(f"  report           {args.report}")
    if not report["passed"]:
        for name, passed in report["checks"].items():
            if not passed:
                print(f"  failed check    {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
