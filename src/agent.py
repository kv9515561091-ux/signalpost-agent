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

from typing import Any

from pydantic import BaseModel, Field

from src.brreg import get_company
from src.conflicts import FieldValue, conflict_report, reconcile_all
from src.crawl import DEFAULT_MAX_CRAWL_LINKS as _DEFAULT_MAX_CRAWL_LINKS
from src.discovery import discover_sources
from src.evidence import Evidence, FactRecord
from src.explain import explain
from src.extract import ExtractedFact
from src.financials import get_financials
from src.identity import CompanyIdentity
from src.orgnumber import InvalidCompanyNumber, normalize, validate
from src.roles import get_roles

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
    phone: str | None = None
    email: str | None = None

    # Health
    bankruptcy: bool | None = None
    under_liquidation: bool | None = None

    # Group
    is_group_parent: bool | None = None
    parent_company_number: str | None = None

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
# Sources beyond roughly six are mostly directories repeating what
# Enhetsregisteret already states, so past that point extra sources buy
# redundancy rather than information. Pages can exceed sources because
# one page costs a model call while a source costs a search.
DEFAULT_MAX_SOURCES = 6
DEFAULT_MAX_PAGES = 10

# Follow-ups read from a verified first-party page. Small on purpose: the
# homepage's best links are already a strong set, and each one costs a
# model call.
#
# Imported rather than redefined. This constant used to exist in both
# here and crawl.py, so the cap the orchestrator applied and the cap the
# planner documented could silently disagree.
DEFAULT_MAX_CRAWL_LINKS = _DEFAULT_MAX_CRAWL_LINKS


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
    same-domain follow-ups. Those inherit the homepage's verification
    rather than earning their own, which is why they are only ever planned
    from a page whose domain the registry already vouched for.
    """
    from src.crawl import plan_crawl
    from src.extract import extract_from_page
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
    ][:max_sources]

    if not candidates:
        return

    # Work queue of (source, is_seed). A seed is a discovered source;
    # only seeds may enqueue follow-ups, which caps the walk at depth 1.
    queue: list[tuple[WebSource, bool]] = [(s, True) for s in candidates]

    fetched: set[str] = set()
    pages_read = 0

    # Carried across pages so the same statement from two directories is
    # published once.
    accepted_facts: list[ExtractedFact] = []

    while queue and pages_read < max_pages:
        source, is_seed = queue.pop(0)

        if source.url in fetched:
            continue

        fetched.add(source.url)

        try:
            page = fetch_page(source.url)
        except UnsafeUrl as exc:
            profile.retrieval_errors.append(
                f"page {source.domain}: blocked ({exc})"
            )
            continue
        except Exception as exc:  # noqa: BLE001
            # Name the actual cause. "HTTPError" alone tells a reader
            # nothing about whether to retry, switch model, or move on.
            detail = _describe_fetch_failure(exc)
            profile.retrieval_errors.append(
                f"page {source.domain}: {detail}"
            )
            continue

        pages_read += 1

        # Follow-ups, planned only from a seed on the registered domain.
        # Comparing against the *identity's* domain rather than trusting
        # the source's is the point: a source that somehow claimed to be
        # on the registered domain must not be able to grant that to the
        # pages it links to.
        #
        # ``pages_read`` is not yet the cap on queue size, so the walk is
        # bounded by the remaining page budget below.
        registered_domain = identity.registered_domain

        if is_seed and registered_domain and source.domain == registered_domain:
            for url in plan_crawl(
                source.url,
                page.links,
                source.domain,
                max_links=max_crawl_links,
                already_seen=fetched,
            ):
                queue.append(
                    (
                        WebSource(
                            url=url,
                            domain=source.domain,
                            tier=source.tier,
                            source_type=source.source_type,
                            # Inherited, not re-earned: the domain was
                            # already verified, so the ceiling is the
                            # homepage's.
                            confidence=source.confidence,
                            identity_basis=f"{source.identity_basis}+same_domain",
                            identity_signals=dict(source.identity_signals),
                            retrieved_at=page.retrieved_at,
                        ),
                        False,
                    )
                )
                profile.web_sources.append(
                    WebSource(
                        url=url,
                        domain=source.domain,
                        tier=source.tier,
                        source_type=source.source_type,
                        confidence=source.confidence,
                        identity_basis=f"{source.identity_basis}+same_domain",
                        identity_signals=dict(source.identity_signals),
                        retrieved_at=page.retrieved_at,
                    )
                )

        report = extract_from_page(
            page_text=page.text,
            source_url=page.url,
            retrieved_at=page.retrieved_at,
            company_number=identity.company_number,
            company_name=identity.name,
            tier=source.tier,
            source_type=source.source_type,
            title=page.title,
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

        if report.rejected:
            profile.extraction_rejected.append(
                {
                    "source_url": source.url,
                    "rejected": report.rejected,
                }
            )
        if report.error and not report.llm_used:
            profile.retrieval_errors.append(f"extraction: {report.error}")

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


def build_profile(
    company_number: str,
    *,
    include_web: bool = False,
    include_extraction: bool = False,
    include_narrative: bool = False,
    use_llm: bool = True,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_crawl_links: int = DEFAULT_MAX_CRAWL_LINKS,
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
        include_narrative: Add a written explanation. Falls back to a
            deterministic summary when no LLM is configured.
        use_llm: Whether to permit LLM calls at all. ``False`` forces
            fully deterministic behaviour.
        max_sources: Source discovery ceiling. Costs scale linearly
            with it, and past roughly six the marginal sources are
            mostly directories that add nothing the registry does not
            already say.
        max_pages: Page-reading ceiling. Bounds both cost and rate-limit
            pressure, since each page costs a model call.
        max_crawl_links: Follow-ups read from a verified first-party
            page. Zero disables the crawl entirely, leaving the
            homepage-only behaviour.

    Raises:
        InvalidCompanyNumber: Malformed number or failed MOD-11 checksum.
        CompanyNotFound: No such entity in Enhetsregisteret.
    """
    org_number = validate(company_number)

    company = get_company(org_number)
    roles = get_roles(org_number)
    financials = get_financials(org_number)

    identity = CompanyIdentity.from_brreg(company)

    registry_evidence = _registry_evidence(company)
    role_evidence, role_records = _roles_evidence(roles)
    fin_evidence, financial_years = _financial_evidence(financials)

    evidence = [*registry_evidence, *role_evidence, *fin_evidence]

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

    sources: list[str] = list(
        dict.fromkeys(
            [company["source"], roles["source"], financials["source"]]
        )
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

        website=company.get("website"),
        phone=company.get("phone"),
        email=company.get("email"),

        bankruptcy=company.get("bankruptcy"),
        under_liquidation=company.get("under_liquidation"),

        is_group_parent=company.get("is_group_parent"),
        parent_company_number=company.get("parent_company_number"),

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

    if include_narrative:
        profile.explanation = explain(profile, use_llm=use_llm).to_dict()

    # `facts` is deliberately left empty here. Every registry field already
    # carries full provenance in `evidence`; mirroring it would double the
    # payload with identical records. This list is reserved for claims the
    # enrichment stage produces, which do not map 1:1 onto a registry field.
    return profile


# ``python -m src.agent <org.nr>`` runs this module as __main__, so the
# entry point is exposed here as well as in src/__main__.py. The import
# is deferred to keep module import free of side effects.
if __name__ == "__main__":  # pragma: no cover - process entry point
    from src.__main__ import main

    raise SystemExit(main())
