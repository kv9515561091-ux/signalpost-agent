"""
Wrong-company protection.

A source is only usable if it demonstrably describes the target company.
The governing rule is that an organization number — not a name — settles
identity. Company names are reused, truncated, translated and re-branded
in Norway often enough that "Acme Solutions" can plausibly be a dozen
unrelated entities, and several directory sites trade on exactly that
ambiguity.

This module is pure: it takes a search result and a canonical identity
and returns a verdict, with no network access and no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.sources import (
    TIER_BUSINESS_DATABASE,
    TIER_FIRST_PARTY,
    TIER_OFFICIAL_REGISTRY,
    SourceClassification,
    classify,
    get_domain,
    org_numbers_in,
)
from src.text import fold

# Legal-form suffixes that carry no identifying signal.
LEGAL_FORMS = (
    "as", "asa", "aksjeselskap", "allmennaksjeselskap", "da", "ba",
    "ansvarlig selskap", "enkeltpersonforetak", "norsk ASA",
    "ltd", "limited", "inc", "gmbh", "ab", "oy", "aps", "hf", "spv",
    "samvirkeforetak", "stiftelse", "forening", "lag",
)

_NEGATIVE_PHRASES = (
    "finnes ikke", "føretaket finst ikkje", "foretaket finst ikkje",
    "finn ikkje føretak", "ikke funnet", "not found",
    "does not exist", "no company found", "no results",
    "ingen treff", "no such company",
)


@dataclass
class CompanyIdentity:
    """The canonical entity, as established by Brreg."""

    company_number: str
    name: str
    previous_names: list[str] = field(default_factory=list)
    address: str | None = None
    postal_code: str | None = None
    municipality: str | None = None
    registered_website: str | None = None
    legal_form: str | None = None
    registered_domain: str | None = None

    @classmethod
    def from_brreg(cls, company: dict) -> CompanyIdentity:
        website = company.get("website")
        return cls(
            company_number=str(company.get("company_number") or ""),
            name=company.get("company_name") or "",
            previous_names=[
                n.get("name")
                for n in company.get("previous_names") or []
                if n.get("name")
            ],
            address=company.get("address"),
            postal_code=_postal_code(company.get("address")),
            municipality=company.get("municipality"),
            registered_website=website,
            legal_form=company.get("organization_form"),
            registered_domain=get_domain(
                website if website else ""
            ) or None,
        )


@dataclass
class IdentityVerdict:
    """Whether a discovered page describes the target company."""

    verified: bool
    basis: str
    signals: dict[str, bool]
    reasons: list[str] = field(default_factory=list)
    classification: SourceClassification | None = None
    # Ranking aid for borderline cases. The verdict itself is decided by
    # the explicit rules above, never by this number.
    score: float = 0.0

    @property
    def tier(self) -> int:
        return self.classification.tier if self.classification else 7

    @property
    def source_type(self) -> str:
        return (
            self.classification.source_type
            if self.classification
            else "unverified_web_content"
        )


def _postal_code(address: str | None) -> str | None:
    if not address:
        return None
    match = re.search(r"\b(\d{4})\b", address)
    return match.group(1) if match else None


def _fold(text: str) -> str:
    """Lowercase and strip diacritics. See :func:`src.text.fold`."""
    return fold(text)


def strip_legal_form(name: str) -> str:
    """Remove a trailing legal form from a company name."""
    folded = _fold(name)
    for form in sorted(LEGAL_FORMS, key=len, reverse=True):
        suffix = f" {form}"
        if folded.endswith(suffix):
            return folded[: -len(suffix)].strip()
    return folded


def name_tokens(name: str) -> set[str]:
    """Distinctive tokens in a company name.

    Single characters and generic words carry no identifying weight, so
    a one-letter name fragment cannot on its own confirm a match.
    """
    generic = {"og", "i", "the", "og", "a", "as", "no", "noe", "den", "det"}
    return {
        token
        for token in strip_legal_form(name).split()
        if len(token) >= 3 and token not in generic
    }


def _is_negative(result: dict) -> bool:
    blob = " ".join(
        [result.get("title") or "", result.get("snippet") or ""]
    ).lower()
    return any(phrase in blob for phrase in _NEGATIVE_PHRASES)


def _website_path_tokens(website: str | None) -> set[str]:
    if not website:
        return set()
    path = urlparse(website if "//" in website else f"//{website}").path
    return {
        token
        for token in re.split(r"[^a-z0-9]+", path.lower())
        if len(token) >= 3
    }


def verify(
    result: dict,
    identity: CompanyIdentity,
) -> IdentityVerdict:
    """Decide whether ``result`` describes ``identity``.

    The decision is made by explicit rules rather than a weighted sum, so
    every verdict can be explained. A score is still reported, but only to
    rank borderline cases.
    """
    url = result.get("url") or ""
    title = result.get("title") or ""
    snippet = result.get("snippet") or ""

    classification = classify(url, identity.company_number)
    signals: dict[str, bool] = {}
    reasons: list[str] = []

    # ---- Hard rejections -------------------------------------------
    if not url:
        return IdentityVerdict(
            False, "no_url", signals, ["result carries no URL"],
            classification,
        )

    if _is_negative(result):
        return IdentityVerdict(
            False, "explicit_not_found", signals,
            ["result states the company was not found"], classification,
        )

    if classification.is_person_page:
        reasons.append("page describes a person, not the company")
        return IdentityVerdict(
            False, "person_page", signals, reasons, classification
        )

    if classification.is_listing_page:
        reasons.append(
            "page lists many companies; its text cannot be attributed "
            "to one of them"
        )
        return IdentityVerdict(
            False, "listing_page", signals, reasons, classification
        )

    # ---- Identity signals ------------------------------------------
    page_text = " ".join([title, snippet])
    target = identity.company_number

    signals["org_number_in_url_path"] = classification.org_number_in_path == target
    # Official registries legitimately take the entity key as a lookup
    # parameter, e.g. w2.brreg.no/kunngjoring/hent_nr.jsp?orgnr=... . The
    # classifier has already confirmed this is a registry domain and not a
    # multi-company listing.
    signals["org_number_in_registry_query"] = (
        classification.org_number_in_query == target
    )

    page_numbers = org_numbers_in(page_text)
    signals["org_number_on_page"] = target in page_numbers
    signals["only_target_number_present"] = page_numbers <= {target}

    # A page naming a *different* org number is describing another entity.
    foreign = sorted(page_numbers - {target})
    signals["foreign_org_number_present"] = bool(foreign)
    if foreign:
        reasons.append(
            f"page cites other organizations: {', '.join(foreign[:3])}"
        )

    folded_title = _fold(title)
    folded_snippet = _fold(snippet)
    folded_page = f"{folded_title} {folded_snippet}"

    core = strip_legal_form(identity.name)
    signals["exact_name_in_title"] = bool(core) and core in folded_title
    signals["exact_name_on_page"] = bool(core) and core in folded_page

    target_tokens = name_tokens(identity.name)
    page_tokens = set(folded_page.split())
    overlap = target_tokens & page_tokens
    ratio = len(overlap) / len(target_tokens) if target_tokens else 0.0
    signals["name_token_overlap"] = ratio >= 0.6

    previous = {strip_legal_form(n) for n in identity.previous_names}
    signals["previous_name_on_page"] = bool(previous) and any(
        p and p in folded_page for p in previous
    )

    if identity.registered_domain:
        signals["registered_domain_match"] = (
            classification.domain == identity.registered_domain
        )
        path_tokens = _website_path_tokens(identity.registered_website)
        signals["registered_path_token_match"] = bool(
            path_tokens & set(re.findall(r"[a-z0-9]+", urlparse(url).path.lower()))
        )
    else:
        signals["registered_domain_match"] = False
        signals["registered_path_token_match"] = False

    if identity.postal_code:
        signals["postal_code_match"] = identity.postal_code in page_text

    # ---- Decision --------------------------------------------------
    verified = False
    basis = "insufficient"

    # Rule 1: the organization number identifies the entity outright,
    # whether it sits in the URL path, in an official registry lookup
    # parameter, or stated on the page.
    if (
        signals["org_number_in_url_path"]
        or signals["org_number_in_registry_query"]
        or signals["org_number_on_page"]
    ):
        verified = True
        basis = "organization_number"

    # Rule 2: the company's own registered website, corroborated by name.
    elif (
        signals["registered_domain_match"]
        and (signals["exact_name_on_page"] or signals["name_token_overlap"])
    ):
        verified = True
        basis = "registered_domain_and_name"

    # Rule 3: a first-party or registry source asserting the exact legal
    # name. Aggregators never qualify on name alone.
    elif (
        classification.tier <= TIER_BUSINESS_DATABASE
        and not classification.is_aggregator
        and signals["exact_name_in_title"]
    ):
        verified = True
        basis = "exact_name_on_authoritative_source"

    if not verified:
        if classification.is_aggregator and not (
            signals["org_number_in_url_path"] or signals["org_number_on_page"]
        ):
            reasons.append(
                "aggregator page with no organization number: a name match "
                "alone cannot establish identity here"
            )
        elif classification.tier not in (
            TIER_OFFICIAL_REGISTRY,
            TIER_FIRST_PARTY,
        ):
            reasons.append("unverified domain with no organization number")

    score = sum(
        (
            1.0 if signals["org_number_in_url_path"] else 0.0,
            1.0 if signals["org_number_in_registry_query"] else 0.0,
            0.7 if signals["org_number_on_page"] else 0.0,
            0.6 if signals["registered_domain_match"] else 0.0,
            0.4 if signals["exact_name_in_title"] else 0.0,
            0.3 if signals["exact_name_on_page"] else 0.0,
            0.3 if signals["name_token_overlap"] else 0.0,
            0.2 if signals["registered_path_token_match"] else 0.0,
            0.2 if signals["previous_name_on_page"] else 0.0,
            0.2 if signals["postal_code_match"] else 0.0,
        )
    )

    return IdentityVerdict(
        verified=verified,
        basis=basis,
        signals=signals,
        reasons=reasons,
        classification=classification,
    )
