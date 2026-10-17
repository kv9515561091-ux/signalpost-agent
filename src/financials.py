"""
Brønnøysundregistrene — Regnskapsregisteret (filed annual accounts).

No authentication is required:

    https://data.brreg.no/regnskapsregisteret/regnskap/{orgnr}

Three properties of this endpoint are easy to get wrong and were each
verified against Equinor ASA's published FY2025 annual report:

1. ``regnskapstype`` must be ``KONSERN`` when available. ``SELSKAP`` is
   the parent company's standalone accounts and reports dramatically
   smaller figures (Equinor: 67.9bn vs 106.5bn revenue).
2. ``valuta`` is the reporting currency and is **not** always NOK. It was
   ``USD`` for Equinor.
3. Optional lines such as ``loennskostnad`` are simply absent under the
   large-company presentation plan, so every lookup must tolerate ``None``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from src.cache import cached_json
from src.orgnumber import validate

REGNSKAPSREGISTERET_BASE = "https://data.brreg.no/regnskapsregisteret/regnskap"

REQUEST_TIMEOUT = 25

# Preferred statement scope, best first.
STATEMENT_PREFERENCE = ("KONSERN", "SELSKAP")


class _Safe:
    """Walk a nested mapping returning ``None`` instead of raising.

    Financial statements vary in depth by company size: small companies
    file a condensed balance sheet, large ones file the full schema.
    """

    def __init__(self, data: Any):
        self._data = data

    def at(self, *path: str) -> Any:
        node = self._data
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    def number(self, *path: str) -> float | None:
        value = self.at(*path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None


def _accounts_url(org_number: str) -> str:
    return f"{REGNSKAPSREGISTERET_BASE}/{org_number}"


def _fetch_accounts(org_number: str) -> list[dict[str, Any]]:
    # Entities with no filed accounts (government bodies, many small sole
    # proprietorships) return 404 rather than an empty array. Absence of
    # filings is a legitimate, reportable state — not an error.
    body, _meta = cached_json(
        _accounts_url(org_number),
        accept_missing=True,
    )
    data = body

    if data is None:
        return []
    if isinstance(data, dict):
        return data.get("regnskap") or data.get("regnskaper") or []
    return data


def _pick_statement(accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the most authoritative statement for a given reporting year."""
    by_type = {a.get("regnskapstype"): a for a in accounts}
    for scope in STATEMENT_PREFERENCE:
        if scope in by_type:
            return by_type[scope]
    return accounts[0] if accounts else None


def _normalise_year(statement: dict[str, Any]) -> dict[str, Any]:
    """Flatten one annual account into the metrics we publish."""
    safe = _Safe(statement)
    period = safe.at("regnskapsperiode", "tilDato") or ""

    revenue = safe.number(
        "resultatregnskapResultat", "driftsresultat", "driftsinntekter",
        "sumDriftsinntekter",
    )
    operating_costs = safe.number(
        "resultatregnskapResultat", "driftsresultat", "driftskostnad",
        "sumDriftskostnad",
    )
    operating_result = safe.number(
        "resultatregnskapResultat", "driftsresultat", "driftsresultat"
    )
    net_result = safe.number(
        "resultatregnskapResultat", "aarsresultat"
    )
    profit_before_tax = safe.number(
        "resultatregnskapResultat", "ordinaertResultatFoerSkattekostnad"
    )
    total_assets = safe.number("eiendeler", "sumEiendeler")
    total_equity = safe.number(
        "egenkapitalGjeld", "egenkapital", "sumEgenkapital"
    )
    total_liabilities = safe.number(
        "egenkapitalGjeld", "gjeldOversikt", "sumGjeld"
    )

    return {
        "fiscal_year": period[:4] or None,
        "period_end": period or None,
        "currency": statement.get("valuta"),
        "statement_scope": statement.get("regnskapstype"),
        "presentation_plan": statement.get("oppstillingsplan"),
        "audited": not safe.at("revisjon", "ikkeRevidertAarsregnskap"),
        "is_liquidation": statement.get("avviklingsregnskap"),

        "revenue": revenue,
        "operating_costs": operating_costs,
        "operating_result": operating_result,
        "profit_before_tax": profit_before_tax,
        "net_result": net_result,

        "total_assets": total_assets,
        "total_equity": total_equity,
        "total_liabilities": total_liabilities,

        "account_id": statement.get("id"),
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Percentage of ``denominator``, or None when that is meaningless.

    Filed accounts are not sanitised: a company may report negative total
    operating income after write-downs or reversals, and an investment
    holding may earn almost all its result below the operating line. A
    ratio computed against a zero or negative base is arithmetic, not
    information, so it is suppressed rather than published.
    """
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100, 1)


def _year_over_year(years: list[dict[str, Any]]) -> None:
    """Annotate each year with growth against the year before it."""
    ordered = sorted(
        (y for y in years if y.get("fiscal_year")),
        key=lambda y: y["fiscal_year"],
    )

# Not strict: consecutive-year pairs come from a list sliced against
    # itself, so the lengths differ by one by construction.
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_revenue = previous.get("revenue")
        current_revenue = current.get("revenue")

        # Both years must have a figure. The guard previously checked
        # only the current year, so a company whose earlier filing
        # omitted revenue raised TypeError subtracting None and the whole
        # profile was lost -- nine of them across a thousand-company run.
        # A filed figure that is absent is not a zero, and treating it
        # as one would be worse than publishing nothing.
        growth = (
            current_revenue - previous_revenue
            if current_revenue is not None and previous_revenue is not None
            else None
        )

        current["revenue_growth_pct"] = _ratio(
            growth,
            abs(previous_revenue) if previous_revenue else None,
        )


def get_financials(company_number: str, max_years: int = 5) -> dict[str, Any]:
    """Fetch filed annual accounts, newest first.

    Returns:
        A dict with ``years`` (most recent first, capped at ``max_years``),
        a ``latest`` convenience view of the newest year, and derived
        ratios. Companies with no filed accounts, or whose accounts could
        not be retrieved, return empty structures rather than raising —
        absence is itself a reportable fact, and a transient network blip
        must not discard the identity data the caller already holds.
    """
    org_number = validate(company_number)
    url = _accounts_url(org_number)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # Annotated because the two branches assign incompatible types, and
    # mypy is right to say so: this is a str-or-absent field, not a str.
    fetch_error: str | None
    try:
        accounts = _fetch_accounts(org_number)
    except (requests.RequestException, ValueError) as exc:
        accounts = []
        fetch_error = f"{type(exc).__name__}: {exc}"
    else:
        fetch_error = None

    if not accounts:
        return {
            "years": [],
            "latest": None,
            "currency": None,
            "source": url,
            "retrieved_at": retrieved_at,
            "error": fetch_error,
        }

    # Group by reporting year *first*, then choose the most authoritative
    # statement within that year. Selecting before grouping silently lets
    # whichever account happens to come last win.
    by_period: dict[str, list[dict[str, Any]]] = {}
    for account in accounts:
        period_end = _Safe(account).at("regnskapsperiode", "tilDato") or ""
        by_period.setdefault(period_end, []).append(account)

    years = []
    for group in by_period.values():
        statement = _pick_statement(group)
        if not statement:
            continue
        year = _normalise_year(statement)
        year["statements_available"] = sorted(
            a.get("regnskapstype") for a in group if a.get("regnskapstype")
        )
        year["operating_margin_pct"] = _ratio(
            year.get("operating_result"), year.get("revenue")
        )
        years.append(year)

    # Newest first, so years[0] is always the most recent filing.
    years.sort(key=lambda y: y.get("period_end") or "", reverse=True)
    years = years[:max_years]

    _year_over_year(years)

    latest = years[0] if years else None

    return {
        "years": years,
        "latest": latest,
        "currency": latest.get("currency") if latest else None,
        "years_available": len(accounts),
        "source": url,
        "retrieved_at": retrieved_at,
        "error": fetch_error,
    }
