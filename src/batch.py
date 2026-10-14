"""
Batch profile generation.

Runs the agent over a list of organization numbers and writes the
profiles as JSON. Each company is independent: one failure records an
error for that entry and the batch continues.

Enrichment is opt-in. The default run is registry-only, which is fast,
free, and safe to repeat. Web discovery and extraction cost money per
call and are slow enough that a large batch should be a deliberate
choice rather than the thing that happens by default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.agent import DEFAULT_MAX_PAGES, DEFAULT_MAX_SOURCES, build_profile
from src.orgnumber import InvalidCompanyNumber

DEFAULT_DELAY_SECONDS = 0.4


def load_company_numbers(input_file: str = "data/company_numbers.txt") -> list[str]:
    path = Path(input_file)
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        progress: Optional callback ``(index, total, raw, profile)`` for
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

    results: list[dict[str, Any]] = []
    total = len(company_numbers)

    for index, raw in enumerate(company_numbers, start=1):
        raw = raw.strip()
        if not raw:
            continue

        time.sleep(delay)

        try:
            profile = build_profile(
                raw,
                include_web=include_web,
                include_extraction=include_extraction,
                include_narrative=include_narrative,
                use_llm=use_llm,
                max_sources=max_sources,
                max_pages=max_pages,
            )
            record = profile.model_dump(mode="json")
            results.append(record)
        except InvalidCompanyNumber as exc:
            record = {"company_number": raw, "error": str(exc)}
            results.append(record)
        except Exception as exc:  # noqa: BLE001 - one bad entity must not
            # abort a batch of good ones.
            record = {
                "company_number": raw,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(record)

        if progress is not None:
            progress(index, total, raw, record)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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