"""
Batch profile generation.

Runs the agent over a list of organization numbers and writes the
profiles as JSON. Each company is independent: one failure records an
error for that entry and the batch continues.

Enrichment is opt-in. The default run is registry-only, which is fast,
free, and safe to repeat. Web discovery and extraction cost money per
call and are slow enough that a large batch should be a deliberate
choice rather than the thing that happens by default.

Results are written after every company rather than once at the end. A
fifty-company enriched run is minutes of paid calls, and losing all of it
because something failed at number thirty is not an acceptable failure
mode for a tool that bills per page.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src import yieldcache
from src.agent import (
    DEFAULT_MAX_CRAWL_LINKS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_SOURCES,
    CompanyProfile,
    build_profile,
)
from src.budget import REGISTRY_REQUESTS_PER_COMPANY, RequestBudget
from src.changes import diff_profiles, summarise_changes
from src.orgnumber import InvalidCompanyNumber
from src.request_tracker import request_tracker

DEFAULT_DELAY_SECONDS = 0.4

# Companies are independent, and the wall clock is scored: a fully
# enriched company measured ~156 seconds, so a hundred sequentially is
# over four hours against a forty-five minute budget. One worker keeps
# the original sequential behaviour and is the default, because
# concurrency is the kind of change that should be opted into.
DEFAULT_WORKERS = 1
MAX_WORKERS = 16


def default_workers(fallback: int = DEFAULT_WORKERS) -> int:
    """The configured worker count, from ``AGENT_WORKERS`` or ``fallback``."""
    raw = (os.getenv("AGENT_WORKERS") or "").strip()
    if not raw:
        return fallback

    try:
        value = int(raw)
    except ValueError:
        return fallback

    return max(1, min(MAX_WORKERS, value))

# Suffix for the sidecar recording which companies are finished and
# under which settings. Kept out of the results file so that file stays
# a plain list of profiles for anything consuming it.
STATE_SUFFIX = ".state.json"

# Measured: a fully enriched company costs about 17 requests. Used as the
# divisor when smoothing what is left across the companies still to do.
PER_COMPANY_ENRICHMENT_COST = 17


def load_company_numbers(input_file: str = "data/company_numbers.txt") -> list[str]:
    path = Path(input_file)
    if not path.exists():
        return []
    return [
        line.strip().lstrip("\ufeff")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _fact_hosts(profile: CompanyProfile) -> set[str]:
    """Hosts that produced at least one fact, for the yield cache."""
    return {
        urlparse(fact.source_url).netloc
        for fact in profile.facts
        if fact.source_url
    }


def mode_signature(
    *,
    include_web: bool,
    include_extraction: bool,
    include_narrative: bool,
    use_llm: bool,
    include_website_check: bool = False,
    bulk: bool = False,
    max_sources: int,
    max_pages: int,
    max_crawl_links: int,
) -> str:
    """A fingerprint of the settings a run was produced under.

    Resume reuses previous results, and a registry-only profile is a
    valid answer to a different question than an enriched one. Without
    this, ``run_batch.py`` followed by ``run_batch.py --resume --extract``
    would return half the companies enriched and half not, with nothing
    in the output saying so.
    """
    modes = []
    if include_web:
        modes.append("web")
    if include_extraction:
        modes.append("extract")
    if include_website_check and not include_extraction:
        modes.append("website")
    if include_narrative:
        modes.append("narrate")
    if use_llm:
        modes.append("llm")
    if bulk:
        modes.append("bulk")

    return (
        f"{'+'.join(modes) or 'registry'}"
        f"|s{max_sources}|p{max_pages}|c{max_crawl_links}"
    )


def _state_path(output_file: str | Path) -> Path:
    return Path(f"{output_file}{STATE_SUFFIX}")


def _replace_with_retry(tmp_name: str, path: Path, attempts: int = 6) -> None:
    """Rename the finished file into place, tolerating a transient lock.

    Windows refuses ``os.replace`` with *Access denied* whenever another
    handle has the destination open without FILE_SHARE_DELETE. Real-time
    scanners do exactly that, and because the results file is rewritten
    after every company the odds of catching one rise with the corpus
    size -- a thousand-company run lost 74 profiles to it before this
    was added.

    The handle is already closed by the time we get here, so any failure
    is somebody else's brief hold on the file. Retrying with a short back
    off clears it; a real permission problem still raises, after the
    budget of attempts, rather than silently losing the write.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp_name, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename.

    The results file is rewritten after every company, so a crash during
    the write itself would otherwise leave truncated JSON -- losing the
    very work the incremental write exists to protect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        _replace_with_retry(tmp_name, path)
    except BaseException:
        # The temporary file is only cleaned up on failure; on success
        # os.replace has already consumed it.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _persist(
    output_file: str,
    results: list[dict[str, Any]],
    completed: set[str],
    signature: str,
) -> None:
    _atomic_write(
        Path(output_file),
        json.dumps(results, indent=2, ensure_ascii=False),
    )
    _atomic_write(
        _state_path(output_file),
        json.dumps(
            {"mode": signature, "completed": sorted(completed)},
            indent=2,
        ),
    )


def load_state(output_file: str, signature: str) -> tuple[list[dict], set[str]]:
    """Previous results, if they were produced under the same settings.

    Returns ``([], set())`` when there is nothing usable, including when
    the previous run used different settings. Treating that as a fresh
    run is the safe reading: redoing work costs money, but mixing modes
    in one output file costs correctness.
    """
    state_path = _state_path(output_file)
    results_path = Path(output_file)

    if not state_path.is_file() or not results_path.is_file():
        return [], set()

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], set()

    if state.get("mode") != signature:
        return [], set()

    if not isinstance(results, list):
        return [], set()

    completed = {
        str(number) for number in state.get("completed", []) if number
    }
    return results, completed


def state_is_complete(
    output_file: str, company_numbers: list[str]
) -> bool:
    """Whether the stored state marks every requested company complete.

    Used by the daily command to tell two situations apart. An
    interrupted run has unfinished companies, so resuming is right. A
    finished run has none, and re-running it must not skip every company
    -- the daily test re-runs the same list to find real changes, and
    skipping would report a quiet day no matter what the registry said.
    """
    path = _state_path(output_file)

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    completed = {
        str(number) for number in state.get("completed", []) if number
    }
    wanted = {str(number) for number in company_numbers}
    return bool(wanted) and wanted.issubset(completed)


def _load_history(
    output_file: str, signature: str
) -> dict[str, dict[str, Any]]:
    """Stored profiles from an earlier run, keyed by org number.

    Read directly rather than through :func:`load_state`, which is about
    deciding what to skip. A run with no ``--resume`` still needs to know
    what it published last time.

    A history written under different settings is still useful -- a
    registry field does not stop being a change because enrichment
    levels differ -- so the mode check is deliberately not applied here.
    """
    path = Path(output_file)

    if not path.is_file():
        return {}

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(records, list):
        return {}

    _ = signature
    return {
        str(record.get("company_number")): record
        for record in records
        if isinstance(record, dict) and record.get("company_number")
    }


def process_companies(
    company_numbers: list[str],
    output_file: str = "data/company_profiles.json",
    delay: float = DEFAULT_DELAY_SECONDS,
    *,
    include_web: bool = False,
    include_extraction: bool = False,
    include_website_check: bool = False,
    include_narrative: bool = False,
    use_llm: bool = True,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_crawl_links: int = DEFAULT_MAX_CRAWL_LINKS,
    company_records: dict[str, dict[str, Any]] | None = None,
    subunit_records: dict[str, list[dict[str, Any]]] | None = None,
    nav_index: Any | None = None,
    website_check_orgs: set[str] | None = None,
    include_news: bool = False,
    news_orgs: set[str] | None = None,
    wikidata_entities: dict[str, Any] | None = None,
    resume: bool = False,
    workers: int = DEFAULT_WORKERS,
    deadline: float | None = None,
    persist_every: int = 1,
    progress=None,
) -> list[dict[str, Any]]:
    """Build a profile for each org number and persist the results.

    Args:
        include_web: Discover and verify public sources per company.
        include_extraction: Extract quote-backed facts from those
            sources. Implies ``include_web``.
        include_website_check: Verify the registry-listed website with a
            page fetch, without a model. Ignored with extraction.
        include_narrative: Add a written explanation per company.
        use_llm: Whether LLM calls are permitted at all. ``False`` forces
            deterministic behaviour even with credentials present, which
            is the safe way to measure enrichment's cost.
        max_sources: Source ceiling per company.
        max_pages: Page ceiling per company.
        max_crawl_links: Same-domain follow-ups read from each verified
            first-party page. Zero disables the crawl.
        company_records: Pre-fetched identity records keyed by organisation
            number, normally from a frozen bulk export. Saves one live
            registry request per company.
        subunit_records: Pre-fetched underenhet records keyed by parent
            organisation number, normally from a frozen export. Adds
            registered workplaces without a live request per company.
        nav_index: Pre-harvested NAV vacancy index shared by the whole
            batch. Read-only and thread-safe; costs no request per
            company.
        website_check_orgs: Restrict the registry-website fetch to these
            organisation numbers. ``None`` checks every company. A large
            batch cannot afford one fetch per company inside the request
            ceiling, so the caller chooses who gets the richer read.
        resume: Reuse completed companies from a previous run under the
            same settings. Companies that errored are retried, since a
            transient failure should not be permanent.
        workers: Companies profiled concurrently. ``1`` keeps the
            sequential behaviour. Results stay in input order regardless,
            and the request ceiling is still enforced process-wide.
        deadline: ``time.monotonic()`` value after which companies stop
            being enriched and are built from the registry alone. They
            are still profiled and still appear in the output, so the
            company list is complete even when the clock is not.
        persist_every: Write the growing profile array and resume state
            after this many completed companies. ``1`` keeps the
            crash-resilient behaviour; larger values avoid rewriting a
            multi-megabyte array a thousand times in a large run. The
            final array is always written.
        progress: Optional callback ``(index, total, raw, record)`` for
            per-company reporting. With concurrency, calls arrive in
            completion order and ``index`` is the company's input
            position.
        include_news: Look up recent press mentions in the keyless
            Google News RSS feed. Costs one request per company.
        news_orgs: Restrict the news lookup to these organisation
            numbers. ``None`` looks up every company.
        wikidata_entities: Batch-wide Wikidata resolution keyed by
            organisation number. Costs one request for the whole run and
            publishes external identifiers where Wikidata knows the
            entity.

    Note:
        The inter-company delay is *not* sufficient rate limiting once
        extraction is on, because a single company issues several model
        calls in quick succession. Those are paced by the LLM limiter,
        which is why it matters that the batch reuses the same process.
    """
    workers = max(1, min(MAX_WORKERS, workers))

    # Extraction reads pages, so it cannot work without discovery.
    if include_extraction:
        include_web = True
        include_website_check = False

    # The yield cache sits beside the output it describes, so a fresh
    # output starts with no history and two runs cannot share a lesson
    # by accident.
    yieldcache.use_default(Path(output_file).with_name("yield_cache.json"))

    signature = mode_signature(
        include_web=include_web,
        include_extraction=include_extraction,
        include_narrative=include_narrative,
        include_website_check=include_website_check,
        use_llm=use_llm,
        bulk=bool(company_records),
        max_sources=max_sources,
        max_pages=max_pages,
        max_crawl_links=max_crawl_links,
    )

    results: list[dict[str, Any]] = []
    completed: set[str] = set()
    persist_every = max(1, persist_every)
    since_persist = 0

    wanted = {
        str(number).strip()
        for number in company_numbers
        if str(number).strip()
    }

    if resume:
        results, completed = load_state(output_file, signature)
        results = [
            record
            for record in results
            if str(record.get("company_number", "")).strip() in wanted
        ]
        completed.intersection_update(wanted)

    # History is loaded whether or not --resume was asked for. --resume
    # decides whether to *skip* companies that are already done; it has
    # no bearing on whether a fresh run knows what it looked like last
    # time. Loading only under --resume meant an ordinary re-run
    # overwrote the very history that makes change detection possible.
    previous = _load_history(output_file, signature)

    if not resume and previous:
        # A refresh rewrites every profile, but an interrupted refresh
        # must not erase the profiles it has not reached yet: keep the
        # earlier record as a placeholder until its company is done.
        # Otherwise one slow run costs the next run its history for the
        # whole tail.
        results = [
            dict(record)
            for number, record in previous.items()
            if number in wanted
        ]

    total = len(company_numbers)
    skipped = 0

    # Smoothing the remaining budget across the companies still to do is
    # what turns "the run died at company 80" into "the last twenty got
    # shallower but every one of them finished".
    #
    # The reserve is capped by the ceiling. Reserving three requests per
    # company for a thousand companies asks for three thousand, which is
    # more than the daily limit, and the result was that enrichment was
    # switched off from the very first company -- three hundred and
    # forty-three registry-only profiles produced, none of them carrying
    # a fact, and nothing in the output said enrichment had been
    # requested. A reserve larger than the budget cannot protect
    # anything.
    ceiling = request_tracker.limit
    registry_per_company = 0 if company_records else REGISTRY_REQUESTS_PER_COMPANY
    reserve = min(
        registry_per_company * total,
        max(0, ceiling - registry_per_company),
    )
    run_budget = RequestBudget.from_tracker(
        request_tracker,
        protected=reserve,
    )
    # The order the caller supplied, so concurrency cannot reorder the
    # output file. A finished company is merged by number, not appended,
    # and the list is then sorted back into input order.
    order: dict[str, int] = {}
    for position, number in enumerate(company_numbers):
        cleaned = number.strip()
        if cleaned:
            order.setdefault(cleaned, position)

    work: list[tuple[int, str]] = []
    for index, raw in enumerate(company_numbers, start=1):
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned in completed:
            skipped += 1
            continue
        work.append((index, cleaned))

    # Productive companies first, unknown next, empty ones last. The
    # daily constraint is wall clock, so if the run has to stop
    # enriching, it should stop on the companies that produced nothing
    # last time rather than on the ones that were paying. The sort is
    # stable and the bands only reorder the queue: the output file is
    # still written in input order.
    work.sort(key=lambda item: yieldcache.priority(item[1]))

    # Walks down as each company is *granted* its depth, not as it
    # finishes. Under concurrency several companies are in flight at
    # once, and the divisor has to be the number still owed a grant, or
    # the in-flight ones get promised the same remainder twice.
    companies_left = len(work)

    # Guards the budget grant and the result merge. Two acquisitions
    # rather than one long hold: deciding a page count takes no time, and
    # persistence must not keep a worker from starting its company.
    state_lock = threading.Lock()

    def run_one(
        raw: str, index: int
    ) -> tuple[int, str, dict[str, Any], str | None]:
        nonlocal companies_left

        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.monotonic()
        requests_before = request_tracker.summary()["requests_used"]

        depth = max_pages
        if yieldcache.priority(raw) == yieldcache.PRIORITY_EMPTY:
            # Same sources, fewer of them. The walk is tier-ordered, so
            # the pages most likely to carry a fact are still read; this
            # drops only the tail that produced nothing last time.
            depth = min(depth, yieldcache.ZERO_YIELD_PAGES)

        skip_reason: str | None = None
        with state_lock:
            # The tracker is the authority on what has been spent; the
            # budget re-reads it rather than counting separately, so the
            # two cannot drift.
            run_budget.sync(request_tracker)

            # Enrichment stops entirely once only the registry reserve
            # remains, or once the promised time has run out. Either way
            # the company is still built, so the output list stays whole.
            if run_budget.registry_only():
                skip_reason = "outbound request budget"
            elif deadline is not None and time.monotonic() >= deadline:
                skip_reason = "daily time budget"

            enrichment_allowed = skip_reason is None
            affordable = run_budget.pages_allowed(
                depth, PER_COMPANY_ENRICHMENT_COST, max(1, companies_left)
            )
            # Promised is spent. Debiting here is what stops the tail of
            # the run from being granted the same remainder as the head.
            run_budget.commit(affordable)
            companies_left = max(0, companies_left - 1)

        profile = None
        clock_exhausted = skip_reason == "daily time budget"
        company_record = (company_records or {}).get(raw)
        has_filing = company_record is None or bool(
            company_record.get("last_filed_annual_accounts")
        )
        include_financials = not clock_exhausted and has_filing
        try:
            profile = build_profile(
                raw,
                include_web=include_web and enrichment_allowed,
                include_extraction=include_extraction and enrichment_allowed,
                include_website_check=(
                    include_website_check
                    and enrichment_allowed
                    and (
                        website_check_orgs is None
                        or raw in website_check_orgs
                    )
                ),
                include_narrative=include_narrative and enrichment_allowed,
                use_llm=use_llm,
                max_sources=max_sources,
                max_pages=affordable,
                max_crawl_links=max_crawl_links,
                company_record=company_record,
                subunits=(subunit_records or {}).get(raw),
                nav_index=nav_index,
                include_news=(
                    include_news
                    and enrichment_allowed
                    and (news_orgs is None or raw in news_orgs)
                ),
                include_roles=not clock_exhausted,
                include_financials=include_financials,
                wikidata_entity=(wikidata_entities or {}).get(raw),
            )
            record = profile.model_dump(mode="json")
            if clock_exhausted:
                record.setdefault("retrieval_errors", []).append(
                    "roles: skipped after the daily time budget ran out"
                )
                if has_filing:
                    record["retrieval_errors"].append(
                        "financials: skipped after the daily time budget ran out"
                    )
        except InvalidCompanyNumber as exc:
            record = {"company_number": raw, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one bad entity must not
            # abort a batch of good ones.
            record = {
                "company_number": raw,
                "error": f"{type(exc).__name__}: {exc}",
            }

        requests_used = (
            request_tracker.summary()["requests_used"] - requests_before
        )
        record["_operations"] = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "runtime_ms": int((time.monotonic() - started_clock) * 1000),
            # With concurrent workers the tracker is process-wide, so a
            # delta can include another company's requests. It is exact
            # for the sequential path and a harmless over-estimate
            # otherwise; the run report carries the authoritative total.
            "requests": max(0, requests_used),
        }

        if (
            profile is not None
            and skip_reason is None
            and include_extraction
            and affordable > 0
        ):
            # Only a company that was actually read teaches the cache
            # anything. One skipped for the budget has no outcome, and
            # recording a zero for it would poison the next run.
            yieldcache.record(
                raw,
                facts=len(profile.facts),
                domains=_fact_hosts(profile),
            )

        return index, raw, record, skip_reason

    def finish(item: tuple[int, str, dict[str, Any], str | None]) -> None:
        nonlocal since_persist
        index, raw, record, skip_reason = item

        with state_lock:
            # Compared against whatever the last run stored for this
            # company, before the new record replaces it. Done for
            # failures too, so a company that stops erroring is reported
            # as recovered.
            record["changes"] = diff_profiles(previous.get(raw), record)
            record["changes_summary"] = summarise_changes(record["changes"])

            # A retried company replaces its earlier record rather than
            # appending beside it, otherwise a company that failed once
            # and then succeeded appears twice -- once as an error, once
            # as a profile -- and the summary counts both.
            # A company that ran out of clock was not finished, whatever
            # its record says. Marking it complete would make --resume
            # skip it forever, turning one slow run into permanently
            # shallow profiles for the tail.
            if "error" not in record and skip_reason is None:
                completed.add(raw)

            if skip_reason is not None and include_extraction and "error" not in record:
                # Enrichment was asked for and not delivered. That is a
                # materially different profile from one that was simply
                # requested and completed, and a caller reading the
                # output needs to be able to tell the two apart -- a run
                # can otherwise produce hundreds of registry-only
                # profiles while looking entirely healthy. Naming the
                # budget that ran out matters: a request ceiling and a
                # clock are fixed by different things.
                record.setdefault("retrieval_errors", []).append(
                    f"enrichment skipped: the {skip_reason} was "
                    "insufficient for this company"
                )

            results[:] = [
                existing
                for existing in results
                if existing.get("company_number") != raw
            ]
            results.append(record)
            results.sort(
                key=lambda row: order.get(
                    str(row.get("company_number")), len(order)
                )
            )

            since_persist += 1
            if since_persist >= persist_every:
                _persist(output_file, results, completed, signature)
                since_persist = 0

        if progress is not None:
            progress(index, total, raw, record)

    if workers <= 1:
        for index, raw in work:
            time.sleep(delay)
            finish(run_one(raw, index))
    else:
        # The inter-company delay is dropped: with several companies in
        # flight the requests are already spaced, and sleeping here would
        # add forty seconds to a hundred-company run for nothing.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, raw, index): (index, raw)
                for index, raw in work
            }
            for future in as_completed(futures):
                index, raw = futures[future]
                try:
                    item = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker that
                    # dies unexpectedly must cost one company, not the
                    # whole batch.
                    item = (
                        index,
                        raw,
                        {
                            "company_number": raw,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        None,
                    )
                finish(item)

    _persist(output_file, results, completed, signature)

    _ = skipped  # reported through the return value contract
    return results


def summarise(results: list[dict[str, Any]]) -> dict[str, int]:
    succeeded = [r for r in results if "error" not in r]
    return {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(results) - len(succeeded),
        "with_financials": sum(1 for r in succeeded if r.get("financials")),
        "with_ceo": sum(1 for r in succeeded if r.get("ceo_name")),
        "with_facts": sum(1 for r in succeeded if r.get("facts")),
        "with_narrative": sum(1 for r in succeeded if r.get("explanation")),
    }
