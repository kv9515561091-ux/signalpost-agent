"""
Batch profile generation.

Runs the agent over a list of organization numbers and writes the
profiles as JSON. Each company is independent: one failure records an
error for that entry and the batch continues.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.agent import build_profile
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
) -> list[dict[str, Any]]:
    """Build a profile for each org number and persist the results."""
    results: list[dict[str, Any]] = []

    for raw in company_numbers:
        raw = raw.strip()
        if not raw:
            continue

        time.sleep(delay)

        try:
            profile = build_profile(raw)
            results.append(profile.model_dump(mode="json"))
        except InvalidCompanyNumber as exc:
            results.append({"company_number": raw, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - one bad entity must not
            # abort a batch of good ones.
            results.append(
                {
                    "company_number": raw,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

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
    }
