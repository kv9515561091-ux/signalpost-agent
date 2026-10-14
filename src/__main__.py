"""
Command-line interface.

    python -m src.agent 989061593

Prints a verified company profile. The registry spine always runs; web
discovery, extraction and narration are opt-in because they cost money
or need API keys.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.agent import build_profile
from src.cache import cache
from src.brreg import CompanyNotFound
from src.http import NotFound
from src.orgnumber import InvalidCompanyNumber
from src.request_tracker import request_tracker

RULE = "=" * 62
THIN = "-" * 62


def _money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "not reported"
    unit = currency or "NOK"
    return f"{amount:,.0f} {unit}"


def render(profile, *, verbose: bool = False) -> str:
    lines: list[str] = []

    lines.append(RULE)
    lines.append("COMPANY PROFILE")
    lines.append(RULE)
    lines.append("")
    lines.append(f"Company:            {profile.company_name or '-'}")
    lines.append(f"Organization No.:   {profile.company_number}")
    if profile.legal_form:
        lines.append(f"Legal form:         {profile.legal_form}")
    if profile.status:
        lines.append(f"Status:             {profile.status}")
    if profile.industry:
        code = f" ({profile.industry_code})" if profile.industry_code else ""
        lines.append(f"Industry:           {profile.industry}{code}")
    if profile.registration_date:
        lines.append(f"Registration date:  {profile.registration_date}")
    if profile.address:
        lines.append(f"Registered address: {profile.address}")
    if profile.website:
        lines.append(f"Website:            {profile.website}")
    if profile.employee_count is not None:
        as_of = (
            f" (as of {profile.employee_count_as_of})"
            if profile.employee_count_as_of
            else ""
        )
        lines.append(f"Employees:          {profile.employee_count}{as_of}")

    lines.append("")
    lines.append(THIN)
    lines.append("MANAGEMENT")
    lines.append(THIN)
    if profile.ceo_name:
        lines.append(f"Chief executive:    {profile.ceo_name}")
    if profile.owner_name:
        lines.append(f"Sole owner:         {profile.owner_name}")
    if profile.board_chair_name:
        lines.append(f"Board chair:        {profile.board_chair_name}")
    if profile.board_size:
        lines.append(f"Board members:      {profile.board_size}")
    if profile.auditor_name:
        lines.append(f"Auditor:            {profile.auditor_name}")
    if not any(
        [profile.ceo_name, profile.owner_name, profile.board_chair_name,
         profile.board_size, profile.auditor_name]
    ):
        lines.append("No management roles on file.")

    lines.append("")
    lines.append(THIN)
    lines.append("FINANCIALS")
    lines.append(THIN)
    if profile.financials:
        for year in profile.financials:
            scope = "consolidated" if year.statement_scope == "KONSERN" else "standalone"
            lines.append(f"FY{year.fiscal_year} ({scope}, {year.currency})")
            lines.append(
                f"   Revenue:      {_money(year.revenue, year.currency)}"
            )
            lines.append(
                f"   Net result:   {_money(year.net_result, year.currency)}"
            )
            lines.append(
                f"   Total assets: {_money(year.total_assets, year.currency)}"
            )
            if year.operating_margin_pct is not None:
                lines.append(
                    f"   Op. margin:   {year.operating_margin_pct}%"
                )
    else:
        lines.append("No annual accounts on file with the registry.")

    if profile.facts:
        lines.append("")
        lines.append(THIN)
        lines.append("ADDITIONAL VERIFIED FACTS")
        lines.append(THIN)
        for fact in profile.facts:
            lines.append(f"{fact.field}: {fact.value}")
            lines.append(
                f"   source: {fact.source_url} "
                f"(tier {fact.source_tier}, confidence {fact.confidence})"
            )
            if verbose and fact.evidence_text:
                quoted = fact.evidence_text.replace("\n", " ")
                lines.append(f"   quote:  \"{quoted[:220]}\"")

    lines.append("")
    lines.append(THIN)
    lines.append("EVIDENCE")
    lines.append(THIN)
    lines.append(f"{len(profile.evidence)} sourced claims recorded.")
    if verbose:
        for item in profile.evidence:
            lines.append(f"  {item.field} = {str(item.value)[:60]}")
            lines.append(f"     {item.source_url}")
            lines.append(
                f"     retrieved {item.retrieved_at} | "
                f"tier {item.source_tier} | confidence {item.confidence}"
            )

    if profile.conflicts:
        lines.append("")
        lines.append(THIN)
        lines.append("UNRESOLVED DISAGREEMENTS")
        lines.append(THIN)
        for conflict in profile.conflicts:
            period = conflict.get("reporting_period")
            suffix = f" (FY{period})" if period else ""
            lines.append(f"{conflict['field']}{suffix}:")
            for value in conflict["values"]:
                lines.append(
                    f"   {value['value']} <- {value['source_type']} "
                    f"(tier {value['tier']}) {value['source_url']}"
                )
            lines.append(f"   published: {conflict['published_value']}")

    if profile.extraction_rejected:
        rejected = sum(
            len(entry["rejected"]) for entry in profile.extraction_rejected
        )
        lines.append("")
        lines.append(
            f"{rejected} extracted candidate(s) were refused during "
            "verification and are not published."
        )

    if profile.retrieval_errors:
        lines.append("")
        lines.append(THIN)
        lines.append("RETRIEVAL GAPS")
        lines.append(THIN)
        for error in profile.retrieval_errors:
            lines.append(f"  - {error}")
        lines.append(
            "  A gap means 'not retrieved', not 'does not exist'."
        )

    if profile.explanation:
        lines.append("")
        lines.append(THIN)
        lines.append("EXPLANATION")
        lines.append(THIN)
        lines.append(profile.explanation["text"])
        if profile.explanation.get("note"):
            lines.append(f"[{profile.explanation['note']}]")

    lines.append("")
    lines.append(THIN)
    lines.append("SOURCES")
    lines.append(THIN)
    for index, source in enumerate(profile.sources, start=1):
        lines.append(f"{index}. {source}")

    lines.append("")
    stats = cache.stats()
    if stats["hits"] or stats["writes"]:
        lines.append(
            f"Cache: {stats['hits']} hit(s), {stats['writes']} written"
        )
    if profile.retrieved_from_cache:
        lines.append(
            "Note: identity was served from the local cache for this run."
        )
    lines.append(
        f"Requests used: {request_tracker.count}/{request_tracker.limit}"
    )
    lines.append(RULE)

    return "\n".join(lines)


def _force_utf8_output() -> None:
    """Emit UTF-8 regardless of the platform's console encoding.

    Every name, address and description this agent prints is Norwegian
    text containing æ, ø and å. On Windows, stdout defaults to a legacy
    code page, which would turn those into replacement characters on the
    way out — the same corruption the agent exists to eliminate, applied
    to its own output.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass


def _check_setup() -> int:
    """Report whether everything is configured, and what to do about it.

    Exists so nobody has to type Python into a shell to find out whether
    their setup works.
    """
    import os

    from src import llm
    from src.cache import DEFAULT_CACHE_DIR

    lines = []
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        mark = "OK  " if passed else "WARN"
        if not passed:
            ok = ok and passed
        lines.append(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))

    lines.append(RULE)
    lines.append("SETUP CHECK")
    lines.append(RULE)

    # Python
    lines.append("")
    lines.append("Python")
    check(f"version {sys.version.split()[0]}", True)

    # Dependencies
    lines.append("")
    lines.append("Dependencies")
    for module, package in (
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("dotenv", "python-dotenv"),
    ):
        try:
            __import__(module)
            check(package, True)
        except ImportError:
            check(package, False, "pip install -r requirements.txt")

    # .env
    lines.append("")
    lines.append("Configuration")
    env_file = "C:\\Users\\X1 Carbon\\Downloads\\Signalpost-Agent\\.env"
    check(".env file", os.path.exists(".env"), env_file if not os.path.exists(".env") else "")

    check("TINYFISH_API_KEY", bool(os.getenv("TINYFISH_API_KEY")),
          "needed for --web" if not os.getenv("TINYFISH_API_KEY") else "")

    anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    nvidia_key = bool(os.getenv("NVIDIA_API_KEY"))

    if anthropic_key or nvidia_key:
        check("LLM key found", True,
              "Anthropic" if anthropic_key else "NVIDIA")
    else:
        check("LLM key", False, "set ANTHROPIC_API_KEY or NVIDIA_API_KEY in .env")

    provider = llm.provider_name()
    if provider:
        lines.append(f"  [OK  ] LLM provider: {llm.describe()}")
    else:
        reason = llm.unavailable_reason() or "not configured"
        check("LLM provider", False, reason)

    # Cache
    lines.append("")
    lines.append("Storage")
    cache_dir = DEFAULT_CACHE_DIR
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check(f"cache directory ({cache_dir})", True)
    except OSError as exc:
        check(f"cache directory ({cache_dir})", False, str(exc))

    check("batch output path (data/company_profiles.json)", True)

    lines.append("")
    lines.append(RULE)
    if ok:
        lines.append("Everything needed for a full run is configured.")
        lines.append("")
        lines.append("Try:")
        lines.append("  python -m src.agent 989061593 --web --extract --narrate")
    else:
        lines.append("Some things above need attention (WARN).")
        lines.append("The registry spine works regardless:")
        lines.append("  python -m src.agent 989061593")
    lines.append(RULE)

    print("\n".join(lines))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()

    parser = argparse.ArgumentParser(
        prog="python -m src.agent",
        description=(
            "Build a verified company profile from a Norwegian "
            "organization number."
        ),
    )
    parser.add_argument(
        "company_number",
        nargs="?",
        help="9-digit Norwegian organization number, e.g. 989061593",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the setup and report what is missing.",
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
             "Anthropic or NVIDIA key).",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Add a written explanation (falls back to a deterministic "
             "summary).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Never call an LLM, even if credentials exist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw profile as JSON instead of a report.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full provenance."
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cached registry responses and exit.",
    )

    args = parser.parse_args(argv)

    if args.clear_cache:
        removed = cache.clear()
        print(f"Removed {removed} cached response(s).")
        return 0

    if args.check:
        return _check_setup()

    if not args.company_number:
        parser.error(
            "an organization number is required "
            "(or use --check, or --clear-cache)"
        )

    try:
        profile = build_profile(
            args.company_number,
            include_web=args.web or args.extract,
            include_extraction=args.extract,
            include_narrative=args.narrate,
            use_llm=not args.no_llm,
        )
    except InvalidCompanyNumber as exc:
        print(f"Invalid organization number: {exc}", file=sys.stderr)
        return 2
    except (CompanyNotFound, NotFound):
        print(
            f"No record found for organization number "
            f"{args.company_number}.",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(render(profile, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())