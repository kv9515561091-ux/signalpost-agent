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
import time
from pathlib import Path
from typing import Any

from src.agent import (
    DEFAULT_MAX_CRAWL_LINKS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_SOURCES,
    build_profile,
)
from src.budget import REGISTRY_REQUESTS_PER_COMPANY, RequestBudget
from src.changes import diff_profiles, summarise_changes
from src.orgnumber import InvalidCompanyNumber
from src.request_tracker import request_tracker

DEFAULT_DELAY_SECONDS = 0.4

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
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mode_signature(
    *,
    include_web: bool,
    include_extraction: bool,
    include_narrative: bool,
    use_llm: bool,
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
    if include_narrative:
        modes.append("narrate")
    if use_llm:
        modes.append("llm")

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
    include_narrative: bool = False,
    use_llm: bool = True,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_crawl_links: int = DEFAULT_MAX_CRAWL_LINKS,
    resume: bool = False,
    progress=None,
) -> list[dict[str, Any]]:
    """Build a profile for each org number and persist the results.

    Args:
        include_web: Discover and verify public sources per company.
        include_extraction: Extract quote-backed facts from those
            sources. Implies ``include_web``.
        include_narrative: Add a written explanation per company.
        use_llm: Whether LLM calls are permitted at all. ``False`` forces
            deterministic behaviour even with credentials present, which
            is the safe way to measure enrichment's cost.
        max_sources: Source ceiling per company.
        max_pages: Page ceiling per company.
        max_crawl_links: Same-domain follow-ups read from each verified
            first-party page. Zero disables the crawl.
        resume: Reuse completed companies from a previous run under the
            same settings. Companies that errored are retried, since a
            transient failure should not be permanent.
        progress: Optional callback ``(index, total, raw, record)`` for
            per-company reporting.

    Note:
        The inter-company delay is *not* sufficient rate limiting once
        extraction is on, because a single company issues several model
        calls in quick succession. Those are paced by the LLM limiter,
        which is why it matters that the batch reuses the same process.
    """
    # Extraction reads pages, so it cannot work without discovery.
    if include_extraction:
        include_web = True

    signature = mode_signature(
        include_web=include_web,
        include_extraction=include_extraction,
        include_narrative=include_narrative,
        use_llm=use_llm,
        max_sources=max_sources,
        max_pages=max_pages,
        max_crawl_links=max_crawl_links,
    )

    results: list[dict[str, Any]] = []
    completed: set[str] = set()

    if resume:
        results, completed = load_state(output_file, signature)

    # History is loaded whether or not --resume was asked for. --resume
    # decides whether to *skip* companies that are already done; it has
    # no bearing on whether a fresh run knows what it looked like last
    # time. Loading only under --resume meant an ordinary re-run
    # overwrote the very history that makes change detection possible.
    previous = _load_history(output_file, signature)

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
    reserve = min(
        REGISTRY_REQUESTS_PER_COMPANY * total,
        max(0, ceiling - REGISTRY_REQUESTS_PER_COMPANY),
    )
    run_budget = RequestBudget.from_tracker(
        request_tracker,
        protected=reserve,
    )
    pending = [n.strip() for n in company_numbers if n.strip()]
    pending = [n for n in pending if n not in completed]
    # Walks down as each company is finished, so the divisor is the real
    # number of companies still owed a profile.
    companies_left = len(pending)

    for index, raw in enumerate(company_numbers, start=1):
        raw = raw.strip()
        if not raw:
            continue

        if raw in completed:
            skipped += 1
            continue

        time.sleep(delay)

        # The tracker is the authority on what has been spent; the budget
        # re-reads it rather than counting separately, so the two cannot
        # drift. Reservations carry over, or every company would be
        # promised the same remaining slice.
        run_budget.sync(request_tracker)

        # Depth for this company is whatever the budget can fairly
        # afford, and enrichment stops entirely once only the registry
        # reserve remains.
        enrichment_allowed = not run_budget.registry_only()
        affordable = run_budget.pages_allowed(
            max_pages, PER_COMPANY_ENRICHMENT_COST, max(1, companies_left)
        )
        # Promised is spent. Debiting here is what stops the tail of the
        # run from being granted the same remainder as the head.
        run_budget.commit(affordable)

        try:
            profile = build_profile(
                raw,
                include_web=include_web and enrichment_allowed,
                include_extraction=include_extraction and enrichment_allowed,
                include_narrative=include_narrative and enrichment_allowed,
                use_llm=use_llm,
                max_sources=max_sources,
                max_pages=affordable,
                max_crawl_links=max_crawl_links,
            )
            record = profile.model_dump(mode="json")
        except InvalidCompanyNumber as exc:
            record = {"company_number": raw, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one bad entity must not
            # abort a batch of good ones.
            record = {
                "company_number": raw,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # Compared against whatever the last run stored for this company,
        # before the new record replaces it. Done for failures too, so a
        # company that stops erroring is reported as recovered.
        record["changes"] = diff_profiles(previous.get(raw), record)
        record["changes_summary"] = summarise_changes(record["changes"])

        # A retried company replaces its earlier record rather than appending
        # beside it, otherwise a company that failed once and then
        # succeeded appears twice -- once as an error, once as a profile
        # -- and the summary counts both.
        if "error" not in record:
            completed.add(raw)

        if (
            not enrichment_allowed
            and include_extraction
            and "error" not in record
        ):
            # Enrichment was asked for and not delivered. That is a
            # materially different profile from one that was simply
            # requested and completed, and a caller reading the output
            # needs to be able to tell the two apart -- a run can
            # otherwise produce hundreds of registry-only profiles
            # while looking entirely healthy.
            record.setdefault("retrieval_errors", []).append(
                "enrichment skipped: the outbound request budget was "
                "insufficient for this company"
            )

        results = [
            existing
            for existing in results
            if existing.get("company_number") != raw
        ]
        results.append(record)

        companies_left = max(0, companies_left - 1)

        _persist(output_file, results, completed, signature)

        if progress is not None:
            progress(index, total, raw, record)

    if not results:
        # Still create the file, so an empty batch does not leave stale
        # output from a previous run looking current.
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
