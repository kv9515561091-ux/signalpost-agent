"""
Web discovery.

Search is a *discovery* mechanism, never evidence. This module finds
candidate pages, hands each to the identity verifier, and returns only
those whose identity could be established. Ranking by reliability tier
happens here so downstream consumers see the most authoritative source
first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.identity import CompanyIdentity, verify
from src.social import is_social_url, social_handle_matches
from src.sources import TIER_FIRST_PARTY, confidence_for, get_domain, rank

DEFAULT_MAX_RESULTS = 10


def build_queries(identity: CompanyIdentity) -> list[str]:
    """Targeted queries, most precise first.

    Quoting the organization number is the strongest search operator
    available, so it leads.
    """
    number = identity.company_number
    name = identity.name
    return [
        f'"{number}"',
        f'"{number}" "{name}"',
        f'"{name}" Norge',
    ]


def _search(provider, query: str, max_results: int) -> list[dict]:
    return provider(query, max_results)


def _absolute(url: str) -> str:
    if not url:
        return ""
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


def _official_website_source(
    identity: CompanyIdentity,
    retrieved_at: str,
) -> dict[str, Any] | None:
    """The company's own registered site, as a first-party source.

    Brreg already asserted this URL belongs to the entity, so identity is
    established by definition. It is not fetched here: the content is
    read later, only if the extraction stage actually needs it.
    """
    website = _absolute(identity.registered_website or "")
    if not website:
        return None

    # A government directory such as "regjeringen.no/aid" is the
    # company's site, but it hosts hundreds of unrelated pages and is
    # already covered by the registry.
    domain = get_domain(website)
    if not domain:
        return None

    return {
        "url": website,
        "title": identity.name,
        "domain": domain,
        "tier": TIER_FIRST_PARTY,
        "source_type": "official_company_website",
        "confidence": confidence_for(TIER_FIRST_PARTY),
        "identity_verified": True,
        "identity_basis": "registered_website",
        "identity_signals": {"registered_domain_match": True},
        "identity_reasons": [],
        "query": None,
        "retrieved_at": retrieved_at,
    }


def discover_sources(
    identity: CompanyIdentity,
    provider,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_queries: int = 2,
) -> list[dict[str, Any]]:
    """Find, verify and rank public sources for a company.

    The company's own registered website is included first and ranked as
    first-party. It is the only source that speaks about the business
    itself — what it sells, who its customers are — rather than
    restating registry fields, and search often fails to surface it.

    Args:
        identity: The canonical entity from Brreg.
        provider: ``(query, max_results) -> list[dict]`` search callable.
        max_results: Results to request per query.
        max_queries: How many queries to issue before giving up.

    Returns:
        Verified source records ordered most-authoritative first. Each
        record carries the identity verdict so a rejected candidate can be
        explained rather than silently dropped.
    """
    accepted: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # ---- the official site, ahead of anything search turns up --------
    official = _official_website_source(identity, retrieved_at)
    if official:
        accepted.append(official)
        seen_urls.add(official["url"])

    for query in build_queries(identity)[:max_queries]:
        try:
            results = _search(provider, query, max_results)
        except Exception:  # noqa: BLE001 - a failed provider must not
            # abort discovery; the registry spine stands on its own.
            continue

        for result in results:
            url = result.get("url")
            if not url or url in seen_urls:
                continue

            if is_social_url(url) and not social_handle_matches(
                url,
                company_name=identity.name,
                previous_names=identity.previous_names,
            ):
                seen_urls.add(url)
                continue

            verdict = verify(result, identity)
            seen_urls.add(url)

            if not verdict.verified:
                continue

            accepted.append(
                {
                    "url": url,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "domain": verdict.classification.domain
                    if verdict.classification
                    else "",
                    "tier": verdict.tier,
                    "source_type": verdict.source_type,
                    "confidence": confidence_for(verdict.tier),
                    "identity_verified": True,
                    "identity_basis": verdict.basis,
                    "identity_signals": verdict.signals,
                    "identity_reasons": verdict.reasons,
                    "query": query,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        if accepted:
            break

    return rank(accepted)


def explain_rejections(
    identity: CompanyIdentity,
    provider,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Report why candidates were refused.

    Useful for auditing: a profile with few sources is far easier to
    improve when the agent can say which candidates it saw and why it did
    not trust them.
    """
    rejected: list[dict[str, Any]] = []

    try:
        results = _search(provider, build_queries(identity)[0], max_results)
    except Exception:  # noqa: BLE001
        return rejected

    for result in results:
        verdict = verify(result, identity)
        if verdict.verified:
            continue
        rejected.append(
            {
                "url": result.get("url"),
                "basis": verdict.basis,
                "tier": verdict.tier,
                "reasons": verdict.reasons or ["no decisive identity signal"],
            }
        )

    return rejected


__all__ = [
    "CompanyIdentity",
    "build_queries",
    "discover_sources",
    "explain_rejections",
]
