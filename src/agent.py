"""
Signalpost Agent — registry-first company intelligence.

The agent resolves a Norwegian organization number to a verified company
profile. Its spine is Brønnøysundregistrene, which supplies identity,
management and filed financials as Tier-1 evidence. Web discovery exists
only to enrich fields the registries do not carry, and anything it
contributes is held to the same identity and reliability checks.

Pipeline::

    org.nr
      -> MOD-11 validation (rejects typos before any network call)
      -> Brreg identity           (canonical)
      -> Brreg roles              (management, board, auditor)
      -> Brreg annual accounts    (multi-year financials)
      -> web discovery            (optional enrichment, identity-gated)
      -> profile + evidence
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field

from src.activity import dated_items
from src.brreg import get_company
from src.conflicts import FieldValue, conflict_report, reconcile_all
from src.contract import utc_now as _utc_now
from src.crawl import DEFAULT_MAX_CRAWL_LINKS as _DEFAULT_MAX_CRAWL_LINKS
from src.discovery import discover_sources
from src.evidence import Evidence, FactRecord
from src.explain import explain
from src.extract import ExtractedFact
from src.financials import get_financials
from src.identity import CompanyIdentity
from src.jobs import careers_index_links, is_careers_url, read_careers_page
from src.nav_jobs import NavJobsIndex, job_facts, match_postings
from src.news import NewsItem, match_news, news_facts, search_news
from src.orgnumber import InvalidCompanyNumber, normalize, validate
from src.roles import get_roles
from src.social import is_social_url
from src.sources import get_domain
from src.website import collect_social_links, verify_registered_page
from src.wikidata import WikidataEntity
from src.wikidata import identity_facts as wikidata_facts

__all__ = [
    "CompanyProfile",
    "FinancialYear",
    "RoleRecord",
    "WebSource",
    "build_profile",
    "normalize_company_number",
    "reconcile_evidence",
    "validate_company_number",
]

# Retained for callers of the previous public surface.
normalize_company_number = normalize


def validate_company_number(company_number: str) -> bool:
    """True when ``company_number`` is a valid Norwegian org number."""
    try:
        validate(company_number)
        return True
    except InvalidCompanyNumber:
        return False


class FinancialYear(BaseModel):
    """One filed fiscal year, with currency kept alongside the figures."""

    fiscal_year: str | None = None
    period_end: str | None = None
    currency: str | None = None
    statement_scope: str | None = None
    audited: bool | None = None

    revenue: float | None = None
    operating_costs: float | None = None
    operating_result: float | None = None
    profit_before_tax: float | None = None
    net_result: float | None = None

    total_assets: float | None = None
    total_equity: float | None = None
    total_liabilities: float | None = None

    operating_margin_pct: float | None = None
    revenue_growth_pct: float | None = None


class RoleRecord(BaseModel):
    """One management, board or audit role."""

    role: str | None = None
    role_code: str | None = None
    group: str | None = None
    group_code: str | None = None
    person_name: str | None = None
    entity_name: str | None = None
    entity_number: str | None = None
    birth_year: str | None = None
    changed_at: str | None = None


class WebSource(BaseModel):
    """A discovered public page whose identity was verified.

    A source is recorded here only once its identity has been
    established. The ``identity_basis`` and ``confidence`` fields record
    *how* that was decided, so a low-confidence source is visibly
    distinguishable from a registry record rather than blended in with it.
    """

    url: str
    title: str | None = None
    domain: str | None = None
    tier: int
    source_type: str
    # Ceiling on any fact this source could support.
    confidence: float
    identity_basis: str
    identity_signals: dict[str, bool] = Field(default_factory=dict)
    query: str | None = None
    retrieved_at: str


class CompanyProfile(BaseModel):
    """The verified output profile."""

    company_number: str
    company_name: str | None = None
    previous_names: list[str] = Field(default_factory=list)

    # Classification
    legal_form: str | None = None
    legal_form_code: str | None = None
    status: str | None = None
    industry: str | None = None
    industry_code: str | None = None
    sector: str | None = None
    business_description: str | None = None
    founding_date: str | None = None
    registration_date: str | None = None
    last_filed_annual_accounts: str | None = None

    # Location
    address: str | None = None
    business_address: str | None = None
    postal_address: str | None = None
    municipality: str | None = None

    # Scale
    employee_count: int | None = None
    employee_count_as_of: str | None = None

    # Contact
    website: str | None = None
    registered_website: str | None = None
    website_verified: bool | None = None
    website_verification: dict[str, Any] | None = None
    social_links: list[dict[str, Any]] = Field(default_factory=list)
    phone: str | None = None
    email: str | None = None

    # Health
    bankruptcy: bool | None = None
    under_liquidation: bool | None = None

    # Group
    is_group_parent: bool | None = None
    parent_company_number: str | None = None
    locations: list[dict[str, Any]] = Field(default_factory=list)

    # Management
    ceo_name: str | None = None
    board_chair_name: str | None = None
    board_size: int | None = None
    auditor_name: str | None = None
    # Sole proprietor. An ENK has no managing director because the
    # owner is the company.
    owner_name: str | None = None
    roles: list[RoleRecord] = Field(default_factory=list)

    # Financials
    financial_currency: str | None = None
    financials: list[FinancialYear] = Field(default_factory=list)

    # Provenance
    sources: list[str] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)
    # Fields where sources disagreed, preserved rather than silently
    # reconciled. Empty when every source agrees.
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    # Candidate facts the extractor refused, with the reason. Kept so a
    # reviewer can see what was offered and why it was not published.
    extraction_rejected: list[dict[str, Any]] = Field(default_factory=list)
    explanation: dict[str, Any] | None = None
    # True when the identity record was served from the local cache
    # rather than read from Brreg during this run.
    retrieved_from_cache: bool = False
    # True when a cached record past its TTL was served because
    # revalidation failed. The profile is usable but knowingly stale.
    served_stale: bool = False
    # Stages that could not be retrieved. An empty profile section that
    # the caller might otherwise mistake for "the company has none of
    # these" is recorded here explicitly.
    retrieval_errors: list[str] = Field(default_factory=list)
    # Families that were checked and produced nothing, in the contract's
    # availability vocabulary. Absent means the family was not checked.
    availability: dict[str, str] = Field(default_factory=dict)
    retrieved_at: str | None = None

    @property
    def latest_financials(self) -> FinancialYear | None:
        return self.financials[0] if self.financials else None


def _record(
    field: str,
    value: Any,
    *,
    source_url: str,
    retrieved_at: str,
    evidence_text: str,
    as_of: str | None = None,
    reporting_period: str | None = None,
    tier: int = 1,
    source_type: str = "official_registry",
) -> Evidence:
    return Evidence(
        field=field,
        value=value,
        source_url=source_url,
        retrieved_at=retrieved_at,
        evidence_text=evidence_text,
        confidence=1.0,
        source_tier=tier,
        source_type=source_type,
        as_of=as_of,
        reporting_period=reporting_period,
    )


def _add(evidence: list[Evidence], entry: Evidence | None) -> None:
    if entry is not None and entry.value is not None and entry.value != "":
        evidence.append(entry)


def _registry_evidence(company: dict[str, Any]) -> list[Evidence]:
    """Attach provenance to every identity field Brreg supplied."""
    url = company["source"]
    at = company["retrieved_at"]
    name = company.get("company_name") or "the entity"
    out: list[Evidence] = []

    _add(out, _record(
        "company_number", company.get("company_number"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Enhetsregisteret org.nr {company.get('company_number')}",

    ))
    _add(out, _record(
        "company_name", company.get("company_name"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Registered legal name: {name}",

    ))
    _add(out, _record(
        "legal_form", company.get("organization_form"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Legal form: {company.get('organization_form')}",

    ))

    industry = company.get("industry") or {}
    _add(out, _record(
        "industry", industry.get("description"),
        source_url=url, retrieved_at=at,
        evidence_text=(
            f"NACE {industry.get('code')}: {industry.get('description')}"
        ),

    ))

    _add(out, _record(
        "business_description", company.get("business_description"),
        source_url=url, retrieved_at=at,
        evidence_text=(
            "Purpose as registered: " + str(company.get("business_description"))
        ),

    ))

    _add(out, _record(
        "address", company.get("business_address") or company.get("address"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Registered business address: {company.get('address')}",

    ))
    _add(out, _record(
        "postal_address", company.get("postal_address"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Registered postal address: {company.get('postal_address')}",

    ))
    _add(out, _record(
        "municipality", company.get("municipality"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Municipality: {company.get('municipality')}",

    ))

    _add(out, _record(
        "registration_date", company.get("registration_date"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Registered in Enhetsregisteret {company.get('registration_date')}",

    ))
    _add(out, _record(
        "founding_date", company.get("founding_date"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Established {company.get('founding_date')}",
    ))
    _add(out, _record(
        "last_filed_annual_accounts", company.get("last_filed_annual_accounts"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Last filed annual accounts: {company.get('last_filed_annual_accounts')}",
        as_of=company.get("last_filed_annual_accounts"),
    ))

    _add(out, _record(
        "employee_count", company.get("employee_count"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Registered employees: {company.get('employee_count')}",
        as_of=company.get("employee_count_as_of"),
    ))
    _add(out, _record(
        "municipality_code", company.get("municipality_code"),
        source_url=url, retrieved_at=at,
        evidence_text=f"Municipality code {company.get('municipality_code')}",

    ))

    for field, label in (
        ("website", "Registered website"),
        ("phone", "Registered phone"),
        ("email", "Registered email"),
    ):
        _add(out, _record(
            field, company.get(field),
            source_url=url, retrieved_at=at,
            evidence_text=f"{label}: {company.get(field)}",

        ))

    for field, label in (
        ("bankruptcy", "Bankruptcy status"),
        ("under_liquidation", "Under liquidation"),
        ("is_group_parent", "Group parent"),
    ):
        value = company.get(field)
        if value is not None:
            _add(out, _record(
                field, value,
                source_url=url, retrieved_at=at,
                evidence_text=f"{label}: {value}",

            ))

    return out


def _roles_evidence(
    roles: dict[str, Any],
) -> tuple[list[Evidence], list[RoleRecord]]:
    """Turn the roles response into records plus provenance."""
    url = roles["source"]
    at = roles["retrieved_at"]
    evidence: list[Evidence] = []
    records: list[RoleRecord] = []

    for raw in roles.get("roles") or []:
        record = RoleRecord(
            role=raw.get("role"),
            role_code=raw.get("role_code"),
            group=raw.get("group"),
            group_code=raw.get("group_code"),
            person_name=raw.get("person_name"),
            entity_name=raw.get("entity_name"),
            entity_number=raw.get("entity_number"),
            birth_year=raw.get("birth_year"),
            changed_at=raw.get("group_changed_at"),
        )
        records.append(record)

        label = record.person_name or record.entity_name
        if not label:
            continue

        _add(evidence, _record(
            f"role:{record.role_code or record.group_code}",
            label,
            source_url=url,
            retrieved_at=at,
            evidence_text=(
                f"{record.role or record.role_code} ({record.group}): {label}"
            ),
            as_of=record.changed_at,
            source_type="official_registry_roles",
        ))

    return evidence, records


def _financial_evidence(
    financials: dict[str, Any],
) -> tuple[list[Evidence], list[FinancialYear]]:
    """Turn filed accounts into year records plus provenance."""
    url = financials["source"]
    at = financials["retrieved_at"]
    evidence: list[Evidence] = []
    years: list[FinancialYear] = []

    metrics = (
        ("revenue", "Revenue"),
        ("operating_result", "Operating result"),
        ("net_result", "Net result"),
        ("total_assets", "Total assets"),
        ("total_equity", "Total equity"),
    )

    for year in financials.get("years") or []:
        record = FinancialYear(**{
            k: v for k, v in year.items()
            if k in FinancialYear.model_fields
        })
        years.append(record)

        currency = record.currency or financials.get("currency") or "NOK"
        scope = record.statement_scope or "SELSKAP"

        for key, label in metrics:
            value = getattr(record, key)
            if value is None:
                continue

            _add(evidence, _record(
                key, value,
                source_url=url,
                retrieved_at=at,
                evidence_text=(
                    f"{label} FY{record.fiscal_year} ({scope}, {currency}): "
                    f"{value:,.0f} {currency}"
                ),
                as_of=record.period_end,
                reporting_period=record.fiscal_year,
                source_type="official_annual_accounts",
            ))

    return evidence, years


def _conflict_key(item: Evidence) -> str | None:
    """Reconciliation key for one evidence record.

    Annual figures are keyed by reporting period, because revenue in
    FY2025 and FY2023 are two different facts rather than a disagreement
    about one. Role records return ``None``: a field like ``role:DAGL``
    legitimately holds several people, and two co-chief executives is a
    fact about the company, not a conflict between sources.
    """
    if item.field == "company_number":
        return None
    if item.field.startswith("role:"):
        return None
    if item.reporting_period:
        return f"{item.field}@{item.reporting_period}"
    return item.field


def reconcile_evidence(evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Detect disagreements across every sourced claim for a company.

    Exposed for testing and for callers that assemble evidence from
    several stages.
    """
    claims: dict[str, list[FieldValue]] = {}

    for item in evidence:
        key = _conflict_key(item)
        if key is None:
            continue

        claims.setdefault(key, []).append(
            FieldValue(
                field=key,
                value=item.value,
                source_url=item.source_url,
                retrieved_at=item.retrieved_at,
                tier=item.source_tier,
                confidence=item.confidence,
                source_type=item.source_type,
                reporting_period=item.reporting_period,
                as_of=item.as_of,
            )
        )

    return conflict_report(reconcile_all(claims))


# Domains whose content the registry already supplied through its API.
# Fetching their HTML spends a request to recover less structured data.
REGISTRY_DOMAINS_FOR_FETCH = frozenset({
    "brreg.no",
    "virksomhet.brreg.no",
    "w2.brreg.no",
    "data.brreg.no",
    "regjeringen.no",
    "forvaltningsdatabasen.sikt.no",
})


def _describe_fetch_failure(exc: Exception) -> str:
    """Turn a transport exception into something actionable.

    A bare ``HTTPError`` gives a reader nothing to act on. The status
    decides what the useful advice is: 403 means the site is refusing an
    automated request and retrying will not help, while 429 means it is
    worth slowing down for.
    """
    import requests

    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status in (401, 403):
            return (
                f"HTTP {status} - the site is refusing automated "
                "requests; nothing to fix here"
            )
        if status == 429:
            return f"HTTP {status} - rate limited; retry later"
        if status == 404:
            return "HTTP 404 - page no longer exists"
        if 500 <= status < 600:
            return f"HTTP {status} - the site had a server error"
        return f"HTTP {status}"

    if isinstance(exc, requests.Timeout):
        return "timed out"
    # SSLError subclasses ConnectionError, so it must be checked first.
    # A TLS failure means the site's certificate or handshake is broken,
    # which is a different problem from a host refusing connections.
    if isinstance(exc, requests.exceptions.SSLError):
        return "TLS/SSL handshake failed (the site may block automated clients)"
    if isinstance(exc, requests.ConnectionError):
        return "connection failed"
    if isinstance(exc, requests.RequestException):
        return f"network error ({type(exc).__name__})"

    return f"{type(exc).__name__}: {exc}"


# Extraction depth.
#
# These bound cost and rate-limit pressure, not evidence quality: the
# identity gate runs before either is consulted, so widening them widens
# how much is *considered*, never what is *trusted*.
#
# Sources beyond roughly eight are mostly directories repeating what
# Enhetsregisteret already states -- measured, tier 2 supplied 81% of
# published facts and tier 6 only 12% -- so past that point extra sources
# buy redundancy rather than information. Pages exceed sources because a
# source costs a search while a page costs a model call.
#
# The measured cost of the previous 6/10 was 17 requests per company.
# With refusing hosts no longer re-requested, 8/14 lands near 20, which
# still fits a hundred companies inside the daily ceiling -- and the
# request budget degrades gracefully rather than failing if it does not.
DEFAULT_MAX_SOURCES = 8
DEFAULT_MAX_PAGES = 14

# Follow-ups read from a verified first-party page. Small on purpose: the
# homepage's best links are already a strong set, and each one costs a
# model call.
#
# Imported rather than redefined. This constant used to exist in both
# here and crawl.py, so the cap the orchestrator applied and the cap the
# planner documented could silently disagree.
DEFAULT_MAX_CRAWL_LINKS = _DEFAULT_MAX_CRAWL_LINKS


def _merge_social_links(profile: CompanyProfile, links: list[dict]) -> None:
    seen = {
        (str(link.get("platform") or ""), str(link.get("handle") or "").lower())
        for link in profile.social_links
    }
    for link in links:
        key = (
            str(link.get("platform") or ""),
            str(link.get("handle") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        profile.social_links.append(link)


def _activity_facts(profile: CompanyProfile, page: Any) -> None:
    """Publish dated headlines listed on a verified first-party page.

    Only the homepage is read in a keyless run, so this is what "recent
    activity" can honestly mean without a search provider: the company
    itself lists this item under this date.
    """
    try:
        items = dated_items(page.text)
    except Exception:  # noqa: BLE001 - a heuristic must never fail a profile
        return

    for item in items:
        profile.facts.append(
            FactRecord(
                field="recent_activity",
                value=item["title"],
                source_url=page.url,
                retrieved_at=page.retrieved_at,
                evidence_text=item["evidence_text"],
                confidence=0.6,
                source_tier=2,
                source_type="official_company_website",
                as_of=item["date"],
            )
        )


_PLATFORM_LABELS = {
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "youtube": "YouTube",
    "twitter": "Twitter",
    "x": "X",
    "tiktok": "TikTok",
}


def _social_quote(page: Any, link: dict[str, Any]) -> str:
    """Find the page text that proves this link exists.

    The link's own label is best. Failing that, a line naming the handle,
    then a line naming the platform: an icon link still sits in a block
    that says "Følg oss på LinkedIn". Nothing found means nothing is
    published -- a claim whose span is not on the page would break the
    quote invariant that every other fact obeys.
    """
    haystack = " ".join(str(page.text or "").split())
    anchor = " ".join(str(link.get("anchor_text") or "").split())
    if anchor and len(anchor) <= 200 and anchor in haystack:
        return anchor

    platform = str(link.get("platform") or "").strip().lower()
    handle = str(link.get("handle") or "").strip()
    label = _PLATFORM_LABELS.get(platform, platform)

    lines = [
        " ".join(line.split())
        for line in str(page.text or "").splitlines()
        if line.strip()
    ]
    for needle in (handle, label):
        if not needle:
            continue
        for line in lines:
            if needle.casefold() in line.casefold() and len(line) <= 200:
                return line
    return ""


def _social_facts(profile: CompanyProfile, page: Any) -> None:
    """Publish validated handles with the page that proves them.

    ``social_links`` was populated but never published, because the
    contract adapter only reads ``evidence`` and ``facts``. The claim is
    the account URL; the evidence is the label or line on the verified
    page that carries it.
    """
    published = {
        str(fact.field)
        for fact in profile.facts
        if str(fact.field).startswith("social_")
    }
    for link in profile.social_links:
        platform = str(link.get("platform") or "").strip()
        source_url = str(link.get("source_url") or "").strip()
        if not platform or not source_url or source_url != page.url:
            continue
        field = f"social_{platform}"
        if field in published:
            continue
        quote = _social_quote(page, link)
        if not quote:
            continue
        published.add(field)
        profile.facts.append(
            FactRecord(
                field=field,
                value=str(link.get("url") or ""),
                source_url=source_url,
                retrieved_at=str(
                    link.get("retrieved_at") or page.retrieved_at
                ),
                evidence_text=quote,
                confidence=0.85,
                source_tier=2,
                source_type="official_company_website",
            )
        )


def _news_fetch(url: str) -> bytes:
    """Fetch a search feed, counted and retried by the shared client."""
    from src.http import fetch_bytes

    # The URL is built here from a fixed endpoint and an encoded name,
    # so the private-address guard has nothing to inspect.
    return fetch_bytes(url, attempts=2, validate=False).body


def _apply_news(
    profile: CompanyProfile,
    *,
    fetch: Callable[[str], bytes] | None = None,
) -> None:
    """Publish recent press mentions that name this company.

    A Google News query returns near-misses as readily as hits, so an
    item is published only when the folded company name appears as a
    whole token in its title or summary. Nothing found is recorded as
    ``not_available`` rather than left silent.
    """
    if not profile.company_name:
        return

    try:
        items: list[NewsItem] = search_news(
            profile.company_name, fetch=fetch or _news_fetch
        )
    except Exception:  # noqa: BLE001 - news is an enrichment
        profile.availability.setdefault("news", "failed")
        return

    matches = match_news(
        items,
        company_name=profile.company_name,
        previous_names=profile.previous_names,
    )
    if not matches:
        profile.availability.setdefault("news", "not_available")
        return

    retrieved_at = profile.retrieved_at or _utc_now()
    for item in news_facts(matches, retrieved_at=retrieved_at):
        profile.facts.append(FactRecord(**item))

    profile.availability.pop("news", None)
    profile.sources = list(
        dict.fromkeys([*profile.sources, *[item.link for item in matches]])
    )


def _wikidata_fetch(query: str) -> dict:
    """POST one SPARQL query to WDQS, counted against the ceiling.

    The endpoint is fixed and the query is built from validated digits,
    so there is no caller-controlled URL for the private-address guard to
    inspect; the request is counted here rather than through the shared
    GET client because SPARQL needs a form-encoded POST.
    """
    import requests

    from src.request_tracker import request_tracker
    from src.wikidata import DEFAULT_ENDPOINT, USER_AGENT

    request_tracker.record()
    response = requests.post(
        DEFAULT_ENDPOINT,
        data={"query": query, "format": "json"},
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _apply_wikidata(
    profile: CompanyProfile,
    entity: WikidataEntity | None,
) -> None:
    """Publish the external identifiers Wikidata resolved for this company.

    The lookup is a single batch request for the whole run, so a company
    that is absent simply has no external identifier; that is not a
    checked-and-empty connector result, so no availability state is
    recorded for it.
    """
    if entity is None:
        return

    retrieved_at = profile.retrieved_at or _utc_now()
    for item in wikidata_facts(entity, retrieved_at=retrieved_at):
        profile.facts.append(FactRecord(**item))

    profile.sources = list(dict.fromkeys([*profile.sources, entity.url]))


def _apply_nav_jobs(
    profile: CompanyProfile,
    index: NavJobsIndex | None,
) -> None:
    """Attribute national vacancy ads to this company by exact name.

    Matching is exact-name only, with a municipality tie-break when two
    different employer spellings fold to the same name. A job ad is
    strong evidence of activity and the fastest way to be disqualified
    is to attach one to the wrong company, so an ambiguous name yields
    nothing rather than a guess.
    """
    if index is None or not index.postings:
        return

    municipalities = [
        value
        for value in [
            profile.municipality,
            *[
                location.get("municipality")
                for location in profile.locations
                if isinstance(location, dict)
            ],
        ]
        if value
    ]
    subunit_names = [
        str(location.get("name"))
        for location in profile.locations
        if isinstance(location, dict) and location.get("name")
    ]

    matches = match_postings(
        index,
        company_name=profile.company_name,
        previous_names=profile.previous_names,
        subunit_names=subunit_names,
        municipalities=municipalities,
    )
    if not matches:
        profile.availability.setdefault("jobs", "not_available")
        return

    for item in job_facts(matches, retrieved_at=index.retrieved_at):
        profile.facts.append(FactRecord(**item))

    # An explicit empty state from the careers probe is superseded by
    # real vacancies; keeping both would publish a contradiction.
    profile.availability.pop("jobs", None)
    profile.sources = list(
        dict.fromkeys([*profile.sources, matches[0].url])
    )


def _drop_registered_website_source(
    profile: CompanyProfile,
    website: str,
) -> None:
    """Remove a registry-seeded site that failed first-party verification."""
    urls = {website.rstrip("/")}
    if not website.startswith(("http://", "https://")):
        urls.add(f"https://{website}".rstrip("/"))
    profile.web_sources = [
        source
        for source in profile.web_sources
        if source.identity_basis != "registered_website"
    ]
    profile.sources = [
        url for url in profile.sources if url.rstrip("/") not in urls
    ]


def _apply_registered_website(
    profile: CompanyProfile,
    identity: CompanyIdentity,
) -> None:
    """Check the registry-listed website without using a model.

    Keyless runs still need a trustworthy ``website`` value. Brreg's URL
    is a candidate, not proof: it may be a parent group or brand site, so
    it is fetched and accepted only when the page carries the company's
    organization number or exact registered name.
    """
    website = (identity.registered_website or "").strip()
    if not website:
        return

    url = website if website.startswith(("http://", "https://")) else f"https://{website}"
    from src.http import HostRefused
    from src.page import UnsafeUrl, fetch_page

    try:
        page = fetch_page(url)
    except UnsafeUrl as exc:
        profile.website_verified = False
        profile.website_verification = {
            "url": website,
            "verified": False,
            "basis": "not_checked",
            "error": f"blocked ({exc})",
        }
        profile.retrieval_errors.append(
            f"registered website {website}: blocked ({exc})"
        )
        _drop_registered_website_source(profile, website)
        return
    except HostRefused as exc:
        profile.website_verified = False
        profile.website_verification = {
            "url": website,
            "verified": False,
            "basis": "not_checked",
            "error": f"not requested ({exc})",
        }
        profile.retrieval_errors.append(
            f"registered website {website}: not requested ({exc})"
        )
        _drop_registered_website_source(profile, website)
        return
    except Exception as exc:  # noqa: BLE001
        detail = _describe_fetch_failure(exc)
        profile.website_verified = False
        profile.website_verification = {
            "url": website,
            "verified": False,
            "basis": "not_checked",
            "error": detail,
        }
        profile.retrieval_errors.append(
            f"registered website {website}: {detail}"
        )
        _drop_registered_website_source(profile, website)
        return

    check = verify_registered_page(identity, page)
    profile.website_verified = check.verified
    profile.website_verification = check.as_dict()
    if not check.verified:
        profile.retrieval_errors.append(
            f"registered website {website}: page does not carry this "
            "company's organization number or exact name; not used as a "
            "first-party source"
        )
        _drop_registered_website_source(profile, website)
        return

    profile.website = identity.registered_website
    profile.sources = list(dict.fromkeys([*profile.sources, page.url]))
    _merge_social_links(profile, collect_social_links(identity, page))
    _social_facts(profile, page)
    _activity_facts(profile, page)


def _careers_probe(website: str | None) -> str | None:
    """One guessed careers URL on the verified domain.

    Careers links are frequently scripted into a footer that the bounded
    crawl never reaches, so the two common paths are tried directly. A
    404 costs one request and is recorded; a hit is the only way many
    small companies ever expose an opening.
    """
    if not website:
        return None
    url = (
        website
        if website.startswith(("http://", "https://"))
        else f"https://{website}"
    )
    parsed = urlparse(url)
    if not parsed.netloc:
        return None
    host = (parsed.hostname or "").lower()
    path = "/karriere" if host.endswith(".no") else "/careers"
    return urlunparse((parsed.scheme or "https", parsed.netloc, path, "", "", ""))


def _apply_extraction(
    profile: CompanyProfile,
    identity: CompanyIdentity,
    *,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_crawl_links: int = DEFAULT_MAX_CRAWL_LINKS,
) -> None:
    """Read verified pages and extract publishable facts from them.

    Only sources that already passed identity verification are fetched, and
    every extracted fact must quote text that exists verbatim in the page
    it claims to come from. Anything else is refused and recorded.

    A verified first-party page may also enqueue a bounded set of
    same-domain follow-ups. Each follow-up is fetched and must itself
    carry the target organization number or exact name before its text
    is extracted, so a footer link cannot borrow the homepage's trust.
    """
    from src.crawl import plan_crawl
    from src.extract import (
        MAX_BATCH_PAGES,
        PageInput,
        extract_from_pages,
    )
    from src.http import HostRefused
    from src.page import UnsafeUrl, fetch_page

    candidates = [
        source
        for source in profile.web_sources
        # Tier 7 pages cannot support a published fact; fetching them
        # would cost a request and a model call for nothing.
        if source.tier <= 6
        # Brreg is already read through the API, with better structure
        # and cleaner provenance than its HTML. Re-fetching a registry
        # search page spends a request to recover data already held.
        and source.domain not in REGISTRY_DOMAINS_FOR_FETCH
        and not is_social_url(source.url)
    ][:max_sources]

    if not candidates:
        return

    # Work queue of (source, is_seed). A seed is a discovered source;
    # only seeds may enqueue follow-ups, which caps the walk at depth 1.
    queue: list[tuple[WebSource, bool]] = [(s, True) for s in candidates]

    # A careers page is often reachable only by guessing its path: the
    # footer link may be JavaScript, or the walk may spend its five
    # follow-ups on About/Services first. One probe is cheap relative to
    # losing the whole jobs family.
    if identity.registered_domain and not any(
        is_careers_url(source.url) for source in candidates
    ):
        probe = _careers_probe(identity.registered_website or profile.website)
        if probe:
            queue.append(
                (
                    WebSource(
                        url=probe,
                        domain=identity.registered_domain,
                        tier=2,
                        source_type="official_company_website",
                        confidence=0.8,
                        identity_basis="registered_website+careers_probe",
                        identity_signals={},
                        retrieved_at="",
                    ),
                    False,
                )
            )

    fetched: set[str] = set()
    pages_read = 0
    # Counted separately from ``pages_read`` because the ceiling has to
    # bound attempts, not successes. A page that times out costs as much
    # wall clock as one that loads, so capping only successes means a
    # company whose sources mostly fail keeps pulling new URLs and can
    # spend several times the intended budget. Measured on a degraded
    # run: twenty companies at a fourteen-page ceiling made 939 requests.
    attempts = 0

    # Carried across pages so the same statement from two directories is
    # published once.
    accepted_facts: list[ExtractedFact] = []

    def enqueue(
        source: WebSource,
        url: str,
        identity_basis: str,
        page: Any,
    ) -> None:
        """Queue one same-domain follow-up for identity-checked reading."""
        if url in fetched:
            return
        if any(queued.url == url for queued, _seed in queue):
            return
        follow_up = WebSource(
            url=url,
            domain=source.domain,
            tier=source.tier,
            source_type=source.source_type,
            confidence=source.confidence,
            identity_basis=identity_basis,
            identity_signals=dict(source.identity_signals),
            retrieved_at=page.retrieved_at,
        )
        queue.append((follow_up, False))
        profile.web_sources.append(follow_up)

    def enqueue_careers_index(source: WebSource, page: Any) -> None:
        """Follow a careers page's own link to its vacancy list.

        Many careers landing pages list nothing themselves and link a
        separate vacancies index; that page is where the openings are. A
        landing page is allowed to hand over the link even when its own
        identity could not be established: the link is only a candidate,
        and the target is verified before it can publish.
        """
        for url in careers_index_links(
            page.url,
            page.links,
            has_postings=bool(
                read_careers_page(page.text, page.url, page.links).posting_links
            ),
        ):
            if identity.registered_domain and (
                get_domain(url) != identity.registered_domain
            ):
                continue
            enqueue(source, url, f"{source.identity_basis}+careers_index", page)

    def fill_batch() -> list[tuple[WebSource, Any]]:
        """Fetch and plan the next batch of pages.

        Runs on the prefetch thread, so it is the only writer of the
        queue and the attempt counter. The model call reads a finished
        batch and touches none of that state.
        """
        nonlocal attempts, pages_read
        batch: list[tuple[WebSource, Any]] = []

        # Fill one batch before calling the model. The page ceiling is
        # checked between fetches, so a batch never pushes a company past
        # it; a failed fetch just leaves a smaller batch.
        while queue and len(batch) < MAX_BATCH_PAGES and attempts < max_pages:
            source, is_seed = queue.pop(0)

            if source.url in fetched:
                continue

            fetched.add(source.url)
            attempts += 1

            try:
                page = fetch_page(source.url)
            except UnsafeUrl as exc:
                if source.identity_basis.endswith("careers_probe"):
                    profile.availability.setdefault("jobs", "blocked")
                profile.retrieval_errors.append(
                    f"page {source.domain}: blocked ({exc})"
                )
                continue
            except HostRefused as exc:
                # Recorded, not hidden: a skipped source is a gap in
                # coverage and the profile should say where the gap came
                # from.
                if source.identity_basis.endswith("careers_probe"):
                    profile.availability.setdefault("jobs", "blocked")
                profile.retrieval_errors.append(
                    f"page {source.domain}: not requested ({exc})"
                )
                continue
            except Exception as exc:  # noqa: BLE001
                # Name the actual cause. "HTTPError" alone tells a reader
                # nothing about whether to retry, switch model, or move
                # on.
                detail = _describe_fetch_failure(exc)
                if source.identity_basis.endswith("careers_probe"):
                    if "404" in detail:
                        profile.availability.setdefault("jobs", "not_available")
                    elif any(
                        token in detail
                        for token in ("403", "429", "TLS", "SSL", "block")
                    ):
                        profile.availability.setdefault("jobs", "blocked")
                    else:
                        profile.availability.setdefault("jobs", "failed")
                profile.retrieval_errors.append(
                    f"page {source.domain}: {detail}"
                )
                continue

            pages_read += 1

            registered_domain = identity.registered_domain
            on_registered_domain = bool(
                registered_domain and source.domain == registered_domain
            )

            if source.identity_basis == "registered_website":
                check = verify_registered_page(identity, page)
                profile.website_verified = check.verified
                profile.website_verification = check.as_dict()
                if not check.verified:
                    profile.retrieval_errors.append(
                        f"registered website {source.domain}: page does "
                        "not carry this company's organization number or "
                        "exact name; not used as a first-party source"
                    )
                    profile.website = None
                    profile.web_sources = [
                        item
                        for item in profile.web_sources
                        if item.url != source.url
                    ]
                    profile.sources = [
                        url for url in profile.sources if url != source.url
                    ]
                    continue
                profile.website = identity.registered_website
                _merge_social_links(
                    profile,
                    collect_social_links(identity, page),
                )
                _social_facts(profile, page)
                _activity_facts(profile, page)
            elif (
                not is_seed
                and on_registered_domain
                and not (
                    profile.website_verified and is_careers_url(page.url)
                )
            ):
                # A careers page on the verified domain inherits the
                # homepage's identity proof. Posting pages rarely repeat
                # the organization number or the registered name, and
                # requiring them was why the jobs family was empty.
                check = verify_registered_page(identity, page)
                if not check.verified:
                    profile.retrieval_errors.append(
                        f"page {source.domain}: follow-up omitted because "
                        "it does not carry this company's organization "
                        "number or exact name"
                    )
                    if is_careers_url(page.url):
                        enqueue_careers_index(source, page)
                    continue

            if on_registered_domain and is_careers_url(page.url):
                enqueue_careers_index(source, page)

            # Follow-ups, planned only from a seed on the registered
            # domain. Comparing against the *identity's* domain rather
            # than trusting the source's is the point: a source that
            # somehow claimed to be on the registered domain must not be
            # able to grant that to the pages it links to.
            #
            # ``attempts`` is not yet the cap on queue size, so the walk
            # is bounded by the remaining page budget below.
            if is_seed and on_registered_domain and source.domain:
                for url in plan_crawl(
                    source.url,
                    page.links,
                    source.domain,
                    max_links=max_crawl_links,
                    already_seen=fetched,
                ):
                    enqueue(
                        source,
                        url,
                        f"{source.identity_basis}+same_domain",
                        page,
                    )

            batch.append((source, page))

        return batch

    def read_batch(batch: list[tuple[WebSource, Any]]) -> None:
        """Send one batch to the model and publish what it proves."""
        # One call for up to two pages. Reports come back in batch
        # order, each carrying its own source URL and refusals.
        reports = extract_from_pages(
            [
                PageInput(
                    page_text=page.text,
                    source_url=page.url,
                    retrieved_at=page.retrieved_at,
                    tier=source.tier,
                    source_type=source.source_type,
                    title=page.title,
                )
                for source, page in batch
            ],
            company_number=identity.company_number,
            company_name=identity.name,
            # Values Brreg already published, so a directory reprinting
            # them is not mistaken for a new discovery.
            registry_values=(
                identity.name,
                identity.address,
                identity.postal_code,
                identity.registered_website,
            ),
            already_accepted=accepted_facts,
        )

        for (source, page), report in zip(batch, reports, strict=True):
            for fact in report.facts:
                accepted_facts.append(fact)
                profile.facts.append(
                    FactRecord(
                        field=fact.field,
                        value=fact.value,
                        source_url=fact.source_url,
                        retrieved_at=fact.retrieved_at,
                        evidence_text=fact.quote,
                        confidence=fact.confidence,
                        source_tier=fact.tier,
                        source_type=fact.source_type,
                        as_of=fact.published_date,
                    )
                )

            if is_careers_url(page.url):
                findings = read_careers_page(page.text, page.url, page.links)
                profile.facts.append(
                    FactRecord(
                        field="careers_url",
                        value=findings.careers_url,
                        source_url=page.url,
                        retrieved_at=page.retrieved_at,
                        evidence_text=page.title or findings.careers_url,
                        confidence=0.8,
                        source_tier=source.tier,
                        source_type=source.source_type,
                    )
                )
                if findings.titles:
                    for title in findings.titles:
                        profile.facts.append(
                            FactRecord(
                                field="job_openings",
                                value=title,
                                source_url=page.url,
                                retrieved_at=page.retrieved_at,
                                evidence_text=findings.snippet or title,
                                confidence=0.6,
                                source_tier=source.tier,
                                source_type=source.source_type,
                            )
                        )
                else:
                    profile.availability.setdefault("jobs", "not_available")
                profile.sources = list(
                    dict.fromkeys([*profile.sources, page.url])
                )

            if report.rejected:
                profile.extraction_rejected.append(
                    {
                        "source_url": page.url,
                        "rejected": report.rejected,
                    }
                )
            if report.error and not report.llm_used:
                profile.retrieval_errors.append(
                    f"extraction: {report.error}"
                )

    # Fetch the next batch while the model reads the current one. A page
    # walk waits on two independent resources -- the site's host and the
    # model endpoint -- and overlapping them takes a fetch off the
    # critical path instead of adding it to every batch. One prefetch
    # thread per company: the fetch order is unchanged, and only the
    # thread that owns the queue touches it.
    #
    # ``AGENT_PREFETCH=0`` keeps the serial walk; it exists so the
    # overlap can be measured against itself on the same provider.
    if os.environ.get("AGENT_PREFETCH", "1") == "0":
        while True:
            batch = fill_batch()
            if not batch:
                break
            read_batch(batch)
    else:
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            upcoming = prefetch.submit(fill_batch)

            while True:
                batch = upcoming.result()
                if not batch:
                    break

                # Start the next fetch before this call blocks on the model.
                upcoming = prefetch.submit(fill_batch)
                read_batch(batch)

    # A stage that is unavailable reports the same reason for every page;
    # repeating it once per attempt obscures the gaps that differ.
    deduplicated = list(dict.fromkeys(profile.retrieval_errors))
    profile.retrieval_errors = deduplicated


def _apply_web_enrichment(
    profile: CompanyProfile,
    company: dict[str, Any],
) -> None:
    """Discover, verify and reconcile web sources against the registry.

    Web sources are attached to the profile with their reliability tier and
    identity basis. They contribute no new facts on their own: without a
    trustworthy extractor, a value read off an arbitrary page is weaker
    evidence than the registry record already in hand, and replacing one
    with the other would be a downgrade dressed as an enrichment.
    """
    identity = CompanyIdentity.from_brreg(company)

    def provider(query: str, max_results: int) -> list[dict]:
        from src.tinyfish_search import search_tinyfish

        return search_tinyfish(query=query, max_results=max_results)

    discovered = discover_sources(identity, provider, max_results=10)

    profile.web_sources = [
        WebSource(
            url=source["url"],
            title=source.get("title"),
            domain=source.get("domain"),
            tier=source["tier"],
            source_type=source["source_type"],
            confidence=source["confidence"],
            identity_basis=source["identity_basis"],
            identity_signals=source.get("identity_signals") or {},
            query=source.get("query"),
            retrieved_at=source["retrieved_at"],
        )
        for source in discovered
    ]

    profile.sources = list(
        dict.fromkeys(
            [*profile.sources, *(s.url for s in profile.web_sources)]
        )
    )

    # Registry records are Tier 1, so any source asserting a different
    # value surfaces as a conflict rather than overwriting them.
    profile.conflicts = reconcile_evidence(profile.evidence)


def _locations_evidence(
    subunits: list[dict[str, Any]] | None,
) -> tuple[list[Evidence], list[dict[str, Any]]]:
    """Turn registered subunits into workplace records plus provenance."""
    evidence: list[Evidence] = []
    records: list[dict[str, Any]] = []

    for raw in (subunits or [])[:25]:
        name = raw.get("name")
        if not name:
            continue
        source_url = str(raw.get("source") or "")
        retrieved_at = str(raw.get("retrieved_at") or "")
        record = {
            "organisation_number": raw.get("organisation_number"),
            "name": name,
            "form": raw.get("organisation_form"),
            "address": raw.get("address"),
            "municipality": raw.get("municipality"),
            "industry": raw.get("industry"),
            "employee_count": raw.get("employee_count"),
            "website": raw.get("website"),
            "email": raw.get("email"),
            "phone": raw.get("phone"),
            "source": source_url,
            "retrieved_at": retrieved_at,
        }
        records.append(record)

        _add(evidence, _record(
            "workplace_name", name,
            source_url=source_url, retrieved_at=retrieved_at,
            evidence_text=f"Registered workplace: {name}",
            source_type="official_subunits",
        ))
        if record["address"]:
            _add(evidence, _record(
                "workplace_address", record["address"],
                source_url=source_url, retrieved_at=retrieved_at,
                evidence_text=(
                    f"Registered workplace address for {name}: "
                    f"{record['address']}"
                ),
                source_type="official_subunits",
            ))
        if record["employee_count"] is not None:
            _add(evidence, _record(
                "workplace_employee_count", record["employee_count"],
                source_url=source_url, retrieved_at=retrieved_at,
                evidence_text=(
                    f"Registered employees at {name}: "
                    f"{record['employee_count']}"
                ),
                source_type="official_subunits",
            ))

    return evidence, records


def build_profile(
    company_number: str,
    *,
    include_web: bool = False,
    include_extraction: bool = False,
    include_website_check: bool = False,
    include_narrative: bool = False,
    use_llm: bool = True,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_crawl_links: int = DEFAULT_MAX_CRAWL_LINKS,
    company_record: dict[str, Any] | None = None,
    subunits: list[dict[str, Any]] | None = None,
    include_roles: bool = True,
    include_financials: bool = True,
    nav_index: NavJobsIndex | None = None,
    include_news: bool = False,
    news_fetch: Callable[[str], bytes] | None = None,
    wikidata_entity: WikidataEntity | None = None,
) -> CompanyProfile:
    """Build a verified company profile from a Norwegian org number.

    Args:
        company_number: The 9-digit organization number, in any
            separator style.
        include_web: Discover and verify public sources. Requires a
            search provider key.
        include_extraction: Read those sources and extract facts from
            them. Each fact must quote the page verbatim or it is
            refused. Requires ``include_web``.
        include_website_check: Verify the registry-listed website by
            fetching it and checking for the organization number or exact
            name, without a model. Ignored when extraction already does
            the same check.
        include_narrative: Add a written explanation. Falls back to a
            deterministic summary when no LLM is configured.
        nav_index: Pre-harvested NAV vacancy index. Costs no request per
            company and publishes vacancy facts for exact-name matches.
        include_news: Look up recent press mentions in the keyless
            Google News RSS search feed. Costs one request per company.
        news_fetch: Injected feed fetcher, so tests need no network.
        wikidata_entity: The entity the batch-wide Wikidata lookup
            resolved for this company. Costs no per-company request and
            publishes the QID and LEI as external identifiers.
        use_llm: Whether to permit LLM calls at all. ``False`` forces
            fully deterministic behaviour.
        max_sources: Source discovery ceiling. Costs scale linearly
            with it, and past roughly six the marginal sources are
            mostly directories that add nothing the registry does not
            already say.
        max_pages: Page-fetch ceiling. Counts attempts rather than
            successes, because a page that times out costs the same wall
            clock as one that loads. Bounds both cost and rate-limit
            pressure, since each page read costs a model call.
        max_crawl_links: Follow-ups read from a verified first-party
            page. Zero disables the crawl entirely, leaving the
            homepage-only behaviour.
        company_record: Pre-fetched Enhetsregisteret record, normally from a
            frozen bulk export. Skips the live identity request.
        subunits: Pre-fetched underenhet records for this parent, normally
            from the frozen underenhet export. Adds registered workplaces
            without a live request.
        include_roles: Fetch management, board and auditor roles. Disable
            only when the clock has run out and identity alone must be
            written for the remaining companies.
        include_financials: Fetch filed annual accounts. Disable when the
            identity record shows no filing, or when the request budget
            cannot afford a call that is known to return nothing.
        nav_index: Pre-harvested NAV Arbeidsplassen vacancy index. When
            given, postings are attributed by exact registered name and
            published as dated job claims. It costs no request per
            company, which is why the index is harvested once per batch.

    Raises:
        InvalidCompanyNumber: Malformed number or failed MOD-11 checksum.
        CompanyNotFound: No such entity in Enhetsregisteret.
    """
    org_number = validate(company_number)

    company = (
        company_record
        if company_record is not None
        else get_company(org_number)
    )
    if include_roles:
        roles = get_roles(org_number)
    else:
        roles = {
            "source": (
                "https://data.brreg.no/enhetsregisteret/api/enheter/"
                f"{org_number}/roller"
            ),
            "retrieved_at": company.get("retrieved_at"),
            "roles": [],
            "error": None,
        }
    if include_financials:
        financials = get_financials(org_number)
    else:
        financials = {
            "years": [],
            "latest": None,
            "currency": None,
            "source": (
                "https://data.brreg.no/regnskapsregisteret/regnskap/"
                f"{org_number}"
            ),
            "retrieved_at": company.get("retrieved_at"),
            "error": None,
        }

    identity = CompanyIdentity.from_brreg(company)

    registry_evidence = _registry_evidence(company)
    role_evidence, role_records = _roles_evidence(roles)
    fin_evidence, financial_years = _financial_evidence(financials)
    location_evidence, location_records = _locations_evidence(subunits)

    evidence = [
        *registry_evidence,
        *role_evidence,
        *fin_evidence,
        *location_evidence,
    ]

    # Identity is mandatory and raises on failure; management and accounts
    # are enrichments that degrade to empty. Record which, so a missing
    # CEO is never mistaken for a company that has none.
    #
    # Built by appending rather than by filtering a list that may hold
    # None, so the result is list[str] from the start and the filtering
    # step cannot be forgotten when this is edited.
    retrieval_errors: list[str] = []

    if roles.get("error"):
        retrieval_errors.append(f"roles: {roles['error']}")

    if financials.get("error"):
        retrieval_errors.append(f"financials: {financials['error']}")

    if company.get("stale"):
        retrieval_errors.append(
            "identity record served from cache past its time-to-live "
            "because Brreg could not be reached"
        )

    industry = company.get("industry") or {}

    source_urls = [company["source"], roles["source"], financials["source"]]
    if location_records:
        source_urls.append(str(location_records[0].get("source") or ""))
    sources: list[str] = list(
        dict.fromkeys(source for source in source_urls if source)
    )

    profile = CompanyProfile(
        company_number=company["company_number"],
        company_name=company.get("company_name"),
        previous_names=[
            n["name"] for n in company.get("previous_names") or []
            if n.get("name")
        ],

        legal_form=company.get("organization_form"),
        legal_form_code=company.get("organization_form_code"),
        status=company.get("status"),
        industry=industry.get("description"),
        industry_code=industry.get("code"),
        sector=(company.get("sector") or {}).get("description"),
        business_description=company.get("business_description"),
        founding_date=company.get("founding_date"),
        registration_date=company.get("registration_date"),
        last_filed_annual_accounts=company.get("last_filed_annual_accounts"),

        address=company.get("address"),
        business_address=company.get("business_address"),
        postal_address=company.get("postal_address"),
        municipality=company.get("municipality"),

        employee_count=company.get("employee_count"),
        employee_count_as_of=company.get("employee_count_as_of"),

        website=None,
        registered_website=company.get("website"),
        phone=company.get("phone"),
        email=company.get("email"),

        bankruptcy=company.get("bankruptcy"),
        under_liquidation=company.get("under_liquidation"),

        is_group_parent=company.get("is_group_parent"),
        parent_company_number=company.get("parent_company_number"),
        locations=location_records,

        ceo_name=roles.get("ceo_name"),
        board_chair_name=roles.get("board_chair_name"),
        board_size=roles.get("board_size"),
        auditor_name=roles.get("auditor_name"),
        owner_name=roles.get("owner_name"),
        roles=role_records,

        financial_currency=financials.get("currency"),
        financials=financial_years,

        sources=sources,
        evidence=evidence,
        retrieval_errors=retrieval_errors,
        retrieved_from_cache=bool(company.get("from_cache")),
        served_stale=bool(company.get("stale")),
        retrieved_at=company["retrieved_at"],
    )

    # -------------------------------------------------------------
    # Optional web enrichment
    #
    # Discovery never overrides a registry record. It can corroborate
    # one, and it can contradict one — in which case the disagreement is
    # recorded rather than resolved by convenience.
    # -------------------------------------------------------------
    if include_web:
        _apply_web_enrichment(profile, company)

        if include_extraction:
            _apply_extraction(
                profile,
                identity,
                max_sources=max_sources,
                max_pages=max_pages,
                max_crawl_links=max_crawl_links,
            )

    if include_website_check and not include_extraction:
        _apply_registered_website(profile, identity)

    _apply_nav_jobs(profile, nav_index)
    _apply_wikidata(profile, wikidata_entity)

    if include_news:
        _apply_news(profile, fetch=news_fetch)

    if include_narrative:
        profile.explanation = explain(profile, use_llm=use_llm).to_dict()

    # `facts` is deliberately left empty here. Every registry field already
    # carries full provenance in `evidence`; mirroring it would double the
    # payload with identical records. This list is reserved for claims the
    # enrichment stage produces, which do not map 1:1 onto a registry field.

    # -----------------------------------------------------------
    # Zero-request derived facts from existing financials
    # -----------------------------------------------------------
    # profile_metrics: computed ratios from filed accounts. The registry
    # returns years newest-first, but the order is not part of the contract,
    # so sort explicitly rather than trusting list position -- picking the
    # wrong year would publish a materially wrong figure.
    ordered_years = sorted(
        financial_years,
        key=lambda year: (year.fiscal_year or "", year.period_end or ""),
    )
    if ordered_years:
        latest = ordered_years[-1]
        prev = ordered_years[-2] if len(ordered_years) >= 2 else None
        currency = latest.currency or financials.get("currency") or "NOK"
        emp = profile.employee_count

        # Revenue growth YoY
        if prev and prev.revenue and latest.revenue and prev.revenue != 0:
            growth = (latest.revenue - prev.revenue) / prev.revenue
            profile.facts.append(FactRecord(
                field="profile_metric_revenue_growth_yoy",
                value=round(growth, 4),
                source_url=financials["source"],
                retrieved_at=financials["retrieved_at"],
                evidence_text=(
                    f"Revenue FY{latest.fiscal_year}: {latest.revenue:,.0f}; "
                    f"FY{prev.fiscal_year}: {prev.revenue:,.0f}; "
                    f"growth: {growth:.2%}"
                ),
                confidence=0.85,
                source_tier=3,
                source_type="official_annual_accounts",
                as_of=latest.period_end,
                reporting_period=latest.fiscal_year,
            ))

        # Profit margin
        if latest.revenue and latest.net_result is not None:
            margin = latest.net_result / latest.revenue
            profile.facts.append(FactRecord(
                field="profile_metric_profit_margin",
                value=round(margin, 4),
                source_url=financials["source"],
                retrieved_at=financials["retrieved_at"],
                evidence_text=(
                    f"Net result FY{latest.fiscal_year}: {latest.net_result:,.0f}; "
                    f"Revenue: {latest.revenue:,.0f}; margin: {margin:.2%}"
                ),
                confidence=0.85,
                source_tier=3,
                source_type="official_annual_accounts",
                as_of=latest.period_end,
                reporting_period=latest.fiscal_year,
            ))

        # Equity ratio
        if latest.total_assets and latest.total_equity is not None and latest.total_assets != 0:
            equity_ratio = latest.total_equity / latest.total_assets
            profile.facts.append(FactRecord(
                field="profile_metric_equity_ratio",
                value=round(equity_ratio, 4),
                source_url=financials["source"],
                retrieved_at=financials["retrieved_at"],
                evidence_text=(
                    f"Equity FY{latest.fiscal_year}: {latest.total_equity:,.0f}; "
                    f"Assets: {latest.total_assets:,.0f}; ratio: {equity_ratio:.2%}"
                ),
                confidence=0.85,
                source_tier=3,
                source_type="official_annual_accounts",
                as_of=latest.period_end,
                reporting_period=latest.fiscal_year,
            ))

        # Revenue per employee
        if emp and latest.revenue:
            rev_per_emp = latest.revenue / emp
            profile.facts.append(FactRecord(
                field="profile_metric_revenue_per_employee",
                value=round(rev_per_emp, 2),
                source_url=financials["source"],
                retrieved_at=financials["retrieved_at"],
                evidence_text=(
                    f"Revenue FY{latest.fiscal_year}: {latest.revenue:,.0f}; "
                    f"Employees: {emp}; per employee: {rev_per_emp:,.0f} {currency}"
                ),
                confidence=0.8,
                source_tier=3,
                source_type="official_annual_accounts",
                as_of=latest.period_end,
                reporting_period=latest.fiscal_year,
            ))

    # freshness_evidence: last activity date across all sources
    as_of_dates = [
        f.as_of for f in profile.facts
        if f.as_of and isinstance(f.as_of, str) and len(f.as_of) >= 10
    ]
    if as_of_dates:
        last_activity = max(as_of_dates)
        retrieved_on = (profile.retrieved_at or "")[:10] or "unknown"
        profile.facts.append(FactRecord(
            field="last_activity_date",
            value=last_activity,
            source_url=financials["source"],
            retrieved_at=profile.retrieved_at or "",
            evidence_text=(
                f"Latest fact date across all sources: {last_activity}; "
                f"Profile retrieved: {retrieved_on}"
            ),
            confidence=0.9,
            source_tier=1,
            source_type="official_registry",
            as_of=last_activity,
        ))


    return profile


# ``python -m src.agent <org.nr>`` runs this module as __main__, so the
# entry point is exposed here as well as in src/__main__.py. The import
# is deferred to keep module import free of side effects.
if __name__ == "__main__":  # pragma: no cover - process entry point
    from src.__main__ import main

    raise SystemExit(main())
