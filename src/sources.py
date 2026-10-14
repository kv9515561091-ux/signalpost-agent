"""
Source reliability model.

Sources are not equally trustworthy and the agent must be able to say
*why* it trusts one more than another. This module assigns every
discovered URL a reliability tier, a source type, and the structural
properties that identity verification depends on.

Two structural traps drive most of the design here:

**Search result pages.** A URL that carries the organization number only
in its query string — ``proff.no/bransjesok?q=989061593`` — is a listing
page listing many companies, not a page about one. Text scraped from it
belongs to an unknown subset of companies, which is how a phone number
for an unrelated entity ends up attributed to the target.

**Person pages.** ``firmadatabasen.no/person/geir-inge-gjorva-1967``
describes a human, not the company. Mixing the two corrupts the profile
with facts that are individually true and collectively wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

# Reliability tiers. Lower number is more authoritative.
TIER_OFFICIAL_REGISTRY = 1
TIER_FIRST_PARTY = 2
TIER_OFFICIAL_FILING = 3
TIER_BUSINESS_DATABASE = 4
TIER_NEWS = 5
TIER_DIRECTORY = 6
TIER_UNVERIFIED = 7

TIER_NAMES = {
    TIER_OFFICIAL_REGISTRY: "official_registry",
    TIER_FIRST_PARTY: "official_company_website",
    TIER_OFFICIAL_FILING: "official_annual_report",
    TIER_BUSINESS_DATABASE: "reputable_business_database",
    TIER_NEWS: "news_source",
    TIER_DIRECTORY: "general_directory",
    TIER_UNVERIFIED: "unverified_web_content",
}

# Confidence ceiling by tier. A Tier-4 source can corroborate a Tier-1
# fact but can never outrank it, so the ceiling encodes that ordering.
TIER_MAX_CONFIDENCE = {
    TIER_OFFICIAL_REGISTRY: 1.00,
    TIER_FIRST_PARTY: 0.90,
    TIER_OFFICIAL_FILING: 0.90,
    TIER_BUSINESS_DATABASE: 0.70,
    TIER_NEWS: 0.60,
    TIER_DIRECTORY: 0.45,
    TIER_UNVERIFIED: 0.25,
}

# Norwegian government and registry infrastructure.
REGISTRY_DOMAINS = {
    "brreg.no",
    "data.brreg.no",
    "virksomhet.brreg.no",
    "w2.brreg.no",
    "regjeringen.no",
    "sikt.no",
    "statsregnskapet.dfo.no",
    "skatteetaten.no",
    "kartverket.no",
}

GOVERNMENT_SUFFIXES = (".gov.no", ".kommune.no", ".fylke.no", ".stat.no")

# Professional and curated company databases.
BUSINESS_DATABASE_DOMAINS = {
    "proff.no",
    "proff.com",
    "byndle.no",
    "opplysning.byndle.no",
    "virksomhet.brreg.no",
    "helsenorge.no",
    "1850.no",
}

# Sites that host many companies per page, mix sources freely, and are
# scraped from other databases. These can never support a published fact
# on a name match alone.
AGGREGATOR_DOMAINS = {
    "purehelp.no",
    "bedri.no",
    "kredittsjekk.no",
    "kredittverdi.no",
    "norgessjekk.no",
    "gulesider.no",
    "firmalisten.no",
    "listings.no",
    "merinfo.no",
    "rettsplassen.no",
    "bestebedrift.no",
    "fagfolkguiden.no",
    "altlokalt.no",
    "sceneweb.no",
    "topograph.co",
    "nor47business.com",
    "firmadatabasen.no",
    "firmabasen.no",
    "vexter.no",
    "forvalt.no",
    "finansavisen.no",
    "okonomi.no",
    "scribd.com",
}

# Path fragments that indicate a person record rather than a company.
PERSON_PATH_HINTS = ("/person", "/personer", "/person-", "/people", "/leder")

# Path fragments that indicate a listing/search page covering many rows.
LISTING_PATH_HINTS = (
    "/sok", "/soek", "/search", "/bransjesok", "/bransjesøk",
    "/produkter", "/kategori", "/liste", "/resultater", "/find",
    "/sted", "/by", "/region",
    # Observed in the live corpus: firmadatabasen.no/kommune/bergen/
    # selskaper lists every company in a municipality. Without this the
    # page reads as being about one company, because its URL path does
    # contain a real name.
    "/kommune", "/kommuner", "/fylke", "/land", "/bransje",
)


@dataclass
class SourceClassification:
    """Structural facts about a discovered URL."""

    url: str
    domain: str
    tier: int
    source_type: str
    is_aggregator: bool = False
    is_listing_page: bool = False
    is_person_page: bool = False
    # True when the *target* organization number appears in the path.
    org_number_in_path: str | None = None
    org_number_in_query: str | None = None
    # Other organization numbers that appear but are not the target.
    foreign_numbers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def max_confidence(self) -> float:
        return TIER_MAX_CONFIDENCE.get(self.tier, TIER_MAX_CONFIDENCE[TIER_UNVERIFIED])


def get_domain(url: str) -> str:
    """The comparable domain for a URL *or* a bare hostname.

    Brreg stores websites as bare hostnames ("www.example.no"), not as
    URLs. ``urlparse`` reads a scheme-less string as a path, so its
    ``netloc`` is empty and every bare hostname silently produced "".

    That is not cosmetic: ``CompanyIdentity.registered_domain`` is
    derived from this, so the whole field was ``None`` for every real
    company -- which disabled ``registered_domain_match`` and with it
    the strongest identity rule, registered website plus name.
    """
    candidate = (url or "").strip()

    if not candidate:
        return ""

    # "example.no" is a path until it is made protocol-relative.
    if "://" not in candidate:
        candidate = f"//{candidate}"

    host = urlparse(candidate).netloc.lower()

    if "@" in host:
        host = host.rsplit("@", 1)[1]

    for prefix in ("www.", "m.", "stage.", "test."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def _domain_matches(domain: str, known: set[str]) -> bool:
    """True when ``domain`` is, or is a subdomain of, any entry."""
    return any(domain == k or domain.endswith("." + k) for k in known)


def find_org_number(text: str) -> str | None:
    """Extract the first standalone 9-digit organization number."""
    return next(iter(org_numbers_in(text)), None)


def normalize_org_number(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def org_numbers_in(text: str) -> set[str]:
    """All standalone 9-digit organization numbers appearing in ``text``."""
    return {
        "".join(m.groups())
        for m in re.finditer(r"(?<!\d)(\d{3})[\s-]?(\d{3})[\s-]?(\d{3})(?!\d)", text)
    }


def classify(url: str, company_number: str | None = None) -> SourceClassification:
    """Assign a reliability tier and structural flags to ``url``."""
    parsed = urlparse(url)
    domain = get_domain(url)
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    notes: list[str] = []

    # --- Reliability tier -------------------------------------------
    if _domain_matches(domain, REGISTRY_DOMAINS) or domain.endswith(
        GOVERNMENT_SUFFIXES
    ):
        tier = TIER_OFFICIAL_REGISTRY
    elif _domain_matches(domain, BUSINESS_DATABASE_DOMAINS):
        tier = TIER_BUSINESS_DATABASE
    elif _domain_matches(domain, AGGREGATOR_DOMAINS):
        tier = TIER_DIRECTORY
    else:
        tier = TIER_UNVERIFIED

    source_type = TIER_NAMES[tier]

    # --- Structural flags ------------------------------------------
    is_aggregator = _domain_matches(domain, AGGREGATOR_DOMAINS)

    is_person_page = any(hint in path for hint in PERSON_PATH_HINTS)
    if is_person_page:
        notes.append("path indicates a person record, not a company")

    is_listing_page = any(hint in path for hint in LISTING_PATH_HINTS)
    if is_listing_page:
        notes.append("path indicates a search or listing page")
        tier = TIER_UNVERIFIED
        source_type = TIER_NAMES[TIER_UNVERIFIED]

    # The organization number must appear as its own path segment. A
    # number buried in a query string identifies the *query*, not a page
    # about one company.
    #
    # Only the target number counts. Many directory URLs carry a 9-digit
    # identifier of their own — ``gulesider.no/.../125157773/bedrift``,
    # ``scribd.com/document/991205763/...`` — and treating that as the
    # org number would confirm identity against a different entity.
    target = normalize_org_number(company_number) if company_number else None
    path_numbers = org_numbers_in(path)
    query_numbers: set[str] = set()
    for values in query.values():
        query_numbers |= org_numbers_in(" ".join(values))

    in_path = target if target and target in path_numbers else None
    in_query = target if target and target in query_numbers else None
    foreign = sorted((path_numbers | query_numbers) - {target}) if target else sorted(
        path_numbers | query_numbers
    )

    if in_query and not in_path:
        # Exception: official registry lookup endpoints legitimately take
        # the org number as a parameter.
        if _domain_matches(domain, REGISTRY_DOMAINS):
            notes.append("org number supplied as a registry lookup parameter")
        else:
            notes.append(
                "org number appears only in the query string; this is a "
                "listing page covering many companies"
            )
            is_listing_page = True
            tier = TIER_UNVERIFIED
            source_type = TIER_NAMES[TIER_UNVERIFIED]

    return SourceClassification(
        url=url,
        domain=domain,
        tier=tier,
        source_type=source_type,
        is_aggregator=is_aggregator,
        is_listing_page=is_listing_page,
        is_person_page=is_person_page,
        org_number_in_path=in_path,
        org_number_in_query=in_query,
        foreign_numbers=foreign,
        notes=notes,
    )


def confidence_for(tier: int, *, corroborated: bool = False) -> float:
    """Confidence ceiling for a source of ``tier``.

    Corroboration by an independent, more authoritative source lifts a
    claim one tier. It never lifts it *to* Tier 1: a directory page that
    happens to agree with something cannot thereby become a registry.
    """
    ceiling = TIER_MAX_CONFIDENCE.get(tier, TIER_MAX_CONFIDENCE[TIER_UNVERIFIED])
    if not corroborated or tier <= TIER_OFFICIAL_REGISTRY:
        return ceiling
    return TIER_MAX_CONFIDENCE.get(tier - 1, ceiling)


def rank(sources: list[dict]) -> list[dict]:
    """Order sources most-authoritative first, stably.

    Tier numbers increase as authority decreases, so sorting ascending on
    the tier puts the registry first and the unverified web last.
    Verified sources precede unverified ones within a tier.
    """
    return sorted(
        sources,
        key=lambda s: (
            int(s.get("tier", TIER_UNVERIFIED)),
            not bool(s.get("identity_verified")),
        ),
    )
