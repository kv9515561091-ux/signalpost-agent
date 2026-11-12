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
# A vacancy index is worth reading for any batch large enough to amortise
# the harvest, which costs no per-company request.
AUTO_NAV_MIN = 100
# One page fetch per company does not fit beside identity, roles and
# accounts inside the request ceiling, so a large batch reads the
# richest companies and leaves the rest registry-only.
#
# A 1,000-company run spends ~1,427 registry requests (roles and filed
# accounts) plus retry overhead on failed fetches. Measured at 300
# website checks and 150 news lookups the run hit 2,000 and dropped the
# last seven companies, so both defaults now leave headroom: website
# checks verify ~53% of companies and are the better per-request buy,
# news ~17%.
AUTO_WEBSITE_CHECKS = 260
# A news lookup is one request per company, so it shares the same
# shrinking slice of the ceiling.
AUTO_NEWS_CHECKS = 90
# Wikidata resolves the whole batch in one SPARQL request, so any batch
# large enough to amortise the ~10 s call is worth it.
AUTO_WIKIDATA_MIN = 100

# The enrichment defaults above were tuned on a 1,000-company cold run
# that spent 1,427 requests on roles and filed accounts. A larger batch
# (the contract says the official set may grow to 1,100) has to take its
# extra spine requests out of enrichment, or the final companies reach
# the ceiling. Up to the verified size the proven defaults are left
# untouched; above it they shrink to fit.
VERIFIED_BATCH = 1000
# Measured registry cost per company on the cold run, rounded up so a
# batch with a higher filing rate still fits.
LIVE_REGISTRY_PER_COMPANY = 1.45
# Failed page fetches cost two requests each (one retry). The cold run
# spent ~139 on retries; leave room for the larger batch.
RETRY_MARGIN = 150
SAFETY_MARGIN = 20


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
        "--html-output",
        default=None,
        metavar="DIR",
        help=(
            "Also render a self-contained static site (index.html plus "
            "one page per company) into DIR. Searchable, offline, mobile-"
            "friendly; every fact links to its source."
        ),
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
        help="Keep the registry-website fetch for every company in --bulk.",
    )
    parser.add_argument(
        "--website-checks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Companies whose registry website is fetched. 0 disables the "
            "check, a negative value fetches every company. Defaults to "
            "260 for runs of 500+ and to every company below that; the "
            "default shrinks automatically for larger batches."
        ),
    )
    parser.add_argument(
        "--news-checks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Companies whose press mentions are looked up in the keyless "
            "Google News RSS feed. 0 disables the lookup, a negative value "
            "looks up every company. Defaults to 90 for runs of 500+ and "
            "to every company below that; the default shrinks automatically "
            "for larger batches."
        ),
    )
    parser.add_argument(
        "--nav-jobs",
        default=os.environ.get("AGENT_NAV_JOBS"),
        metavar="PATH",
        help=(
            "Frozen NAV Arbeidsplassen vacancy snapshot, or 'auto' for the "
            "cached snapshot refreshed when stale, or 'off'. Runs of 100+ "
            "default to auto."
        ),
    )
    parser.add_argument(
        "--wikidata",
        default=os.environ.get("AGENT_WIKIDATA"),
        metavar="auto|off",
        help=(
            "Resolve external identifiers (QID, LEI) from Wikidata in one "
            "batch request. 'off' disables it. Runs of 100+ default to auto."
        ),
    )
    return parser


def _website_check_limit(count: int, requested: int | None) -> int:
    if requested is None:
        return count if count < AUTO_BULK_MIN else AUTO_WEBSITE_CHECKS
    return requested


def _news_check_limit(count: int, requested: int | None) -> int:
    if requested is None:
        return count if count < AUTO_BULK_MIN else AUTO_NEWS_CHECKS
    return requested


def _fit_enrichment_to_ceiling(
    inputs: list[dict[str, Any]],
    requested_website: int | None,
    requested_news: int | None,
    website_limit: int,
    news_limit: int,
    company_records: dict[str, dict[str, Any]] | None,
    ceiling: int,
) -> tuple[int, int]:
    """Shrink the automatic enrichment caps so a large batch still fits.

    Explicit ``--website-checks`` / ``--news-checks`` are honoured: a
    caller that names a number owns the budget. Only the defaults
    shrink, and only above the batch size they were verified against.
    """
    count = len(inputs)
    if count <= VERIFIED_BATCH:
        return website_limit, news_limit

    if company_records:
        # One roles request per company plus one accounts request for
        # every company with a filing. Bulk misses default to a filing,
        # which keeps the estimate conservative.
        filings = sum(
            1
            for item in inputs
            if (company_records.get(str(item["organisation_number"])) or {}).get(
                "last_filed_annual_accounts", True
            )
        )
        registry_cost = count + filings
    else:
        registry_cost = int(count * LIVE_REGISTRY_PER_COMPANY)

    room = max(0, ceiling - registry_cost - RETRY_MARGIN - SAFETY_MARGIN)
    # Website checks verify ~53% of companies and news ~17%, so the
    # scarcer room goes to the better per-request buy first.
    if requested_website is None:
        website_limit = min(website_limit, room * 74 // 100)
    if requested_news is None:
        news_limit = min(news_limit, max(0, room - website_limit))
    return website_limit, news_limit


def _news_name_is_usable(record: dict[str, Any]) -> bool:
    from src.news import MIN_KEY_LENGTH, key_length

    return key_length(str(record.get("company_name") or "")) >= MIN_KEY_LENGTH


def _select_news_checks(
    inputs: list[dict[str, Any]],
    company_records: dict[str, dict[str, Any]] | None,
    limit: int,
) -> set[str] | None:
    """Pick the companies whose press coverage is worth a request.

    Unlike the website check there is no precondition: press coverage is
    about the company, not its site. Names too short to match safely are
    skipped, because a three-letter query returns unrelated articles and
    the identity gate would have to reject every one of them.
    """
    if limit < 0:
        return None
    if limit == 0:
        return set()

    known = company_records or {}
    ranked: list[tuple[int, int, int, str]] = []
    for position, item in enumerate(inputs):
        org = str(item["organisation_number"])
        record = known.get(org)
        if record is None:
            # No bulk snapshot means no name to judge; the connector
            # applies its own length gate once the registry answers.
            ranked.append((0, 0, position, org))
            continue
        if not _news_name_is_usable(record):
            continue
        employees = int(record.get("employee_count") or 0)
        filing = 1 if record.get("last_filed_annual_accounts") else 0
        ranked.append((-employees, -filing, position, org))
    ranked.sort()
    return {entry[3] for entry in ranked[:limit]}


def _select_website_checks(
    inputs: list[dict[str, Any]],
    company_records: dict[str, dict[str, Any]] | None,
    limit: int,
) -> set[str] | None:
    """Pick the companies whose website is worth a request.

    A registered website is the precondition, so companies without one
    are never selected. Among the rest, size first and a filed annual
    account second: those are the profiles a reader is most likely to
    look at, and the ones where a verified site adds the most.
    """
    if limit < 0:
        return None
    if limit == 0:
        return set()

    known = company_records or {}
    ranked: list[tuple[int, int, int, str]] = []
    for position, item in enumerate(inputs):
        org = str(item["organisation_number"])
        record = known.get(org)
        if record is None:
            # Without a bulk snapshot the planner cannot see the
            # registered website, so it selects in input order and lets
            # the per-company check decide.
            ranked.append((0, 0, position, org))
            continue
        if not record.get("website"):
            continue
        employees = int(record.get("employee_count") or 0)
        filing = 1 if record.get("last_filed_annual_accounts") else 0
        ranked.append((-employees, -filing, position, org))
    ranked.sort()
    return {entry[3] for entry in ranked[:limit]}


def _mode(
    args: argparse.Namespace,
    *,
    bulk: bool = False,
    website_checks: int = 1,
) -> tuple[bool, bool, bool, bool, bool]:
    from src.llm import is_explicitly_configured

    search_available = bool(os.environ.get("TINYFISH_API_KEY"))
    llm_ready = is_explicitly_configured()
    if args.keyless:
        search_available = False
        llm_ready = False

    include_web = search_available or llm_ready
    include_extraction = llm_ready
    include_website_check = (
        not include_extraction
        and (not bulk or website_checks != 0)
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

    requested_checks = args.website_checks
    if getattr(args, "verify_websites", False):
        requested_checks = -1
    website_limit = _website_check_limit(len(inputs), requested_checks)
    news_limit = _news_check_limit(len(inputs), args.news_checks)
    website_limit, news_limit = _fit_enrichment_to_ceiling(
        inputs,
        requested_checks,
        args.news_checks,
        website_limit,
        news_limit,
        bulk_records,
        request_tracker.limit,
    )
    website_check_orgs = _select_website_checks(
        inputs, bulk_records, website_limit
    )

    news_orgs = _select_news_checks(inputs, bulk_records, news_limit)
    include_news = news_orgs is None or bool(news_orgs)

    (
        include_web,
        include_extraction,
        include_website_check,
        include_narrative,
        use_llm,
    ) = _mode(
        args, bulk=bool(bulk_records), website_checks=website_limit
    )

    nav_index = None
    nav_meta: dict[str, Any] | None = None
    nav_path = args.nav_jobs
    if nav_path is None and len(inputs) >= AUTO_NAV_MIN:
        nav_path = "auto"
    if nav_path and nav_path != "off":
        from src.nav_jobs import DEFAULT_NAV_PATH, ensure_snapshot, load

        if nav_path == "auto":
            nav_index, nav_meta = ensure_snapshot(DEFAULT_NAV_PATH)
        else:
            nav_index = load(nav_path)
            nav_meta = {
                "path": nav_path,
                "postings": len(nav_index.postings) if nav_index else 0,
                "retrieved_at": nav_index.retrieved_at if nav_index else None,
            }
            if nav_index is None:
                nav_meta["error"] = "snapshot missing or unreadable"
        if nav_meta and nav_meta.get("error"):
            print(f"  nav jobs unavailable: {nav_meta['error']}")
        elif nav_meta:
            print(
                f"  nav jobs {nav_meta['postings']} postings "
                f"({nav_meta['retrieved_at']})"
            )

    wikidata_entities: dict[str, Any] = {}
    wikidata_meta: dict[str, Any] | None = None
    wikidata_mode = args.wikidata
    if wikidata_mode is None and len(inputs) >= AUTO_WIKIDATA_MIN:
        wikidata_mode = "auto"
    if wikidata_mode and wikidata_mode != "off":
        from src import wikidata as wikidata_module
        from src.agent import _wikidata_fetch

        try:
            wikidata_entities = wikidata_module.resolve(
                [item["organisation_number"] for item in inputs],
                fetch=_wikidata_fetch,
            )
            resolved = len(wikidata_entities)
            with_lei = sum(
                1 for entity in wikidata_entities.values() if entity.lei
            )
            wikidata_meta = {"resolved": resolved, "with_lei": with_lei}
            print(
                f"  wikidata {resolved}/{len(inputs)} entities "
                f"({with_lei} with LEI)"
            )
        except Exception as exc:  # noqa: BLE001 - identity is an enrichment
            wikidata_entities = {}
            error = f"{type(exc).__name__}: {exc}"
            wikidata_meta = {"error": error}
            print(f"  wikidata unavailable: {error}")

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
        nav_index=nav_index,
        website_check_orgs=website_check_orgs,
        include_news=include_news,
        news_orgs=news_orgs,
        wikidata_entities=wikidata_entities,
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
    report["nav_jobs"] = nav_meta
    report["website_checks"] = {
        "enabled": bool(include_website_check),
        "limit": website_limit,
        "selected": (
            None if website_check_orgs is None else len(website_check_orgs)
        ),
    }
    report["news"] = {
        "enabled": include_news,
        "limit": news_limit,
        "selected": None if news_orgs is None else len(news_orgs),
    }
    report["wikidata"] = wikidata_meta
    report["companies"] = company_rows(envelopes)
    report["html_output"] = args.html_output
    write_json(report, args.report)
    if args.html_output:
        from src.html_report import render_site

        rendered = render_site(envelopes, args.html_output)
        print()
        print(f"  html             {rendered['pages']} pages in {args.html_output}")

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
