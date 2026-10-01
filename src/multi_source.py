from urllib.parse import urlparse

from src.public_search import build_search_queries
from src.search_provider import search_web
from src.source_hierarchy import rank_sources
from src.tinyfish_search import search_tinyfish


def get_domain(url: str) -> str:
    """Extract and normalize the domain from a URL."""
    return urlparse(url).netloc.lower().replace("www.", "")


def classify_source_type(url: str) -> str:
    """
    Classify a discovered source using its domain.
    """

    domain = get_domain(url)

    if domain == "brreg.no" or domain.endswith(".brreg.no"):
        return "official_registry"

    if domain.endswith(".gov.no"):
        return "official_government_source"

    if domain == "regjeringen.no" or domain.endswith(".regjeringen.no"):
        return "official_government_source"

    if domain.endswith(".kommune.no"):
        return "official_government_source"

    if domain in {
        "sikt.no",
        "utdanning.no",
    }:
        return "official_government_source"

    if domain in {
        "foretaksinfo.no",
        "byndle.no",
    }:
        return "reputable_business_database"

    return "search_result"


def verify_search_result_identity(
    result: dict,
    company_name: str,
    company_number: str,
    official_domain: str | None = None,
) -> bool:
    """
    Verify company identity using search-result metadata.

    Organization numbers are normalized so both:

        983887457
        983 887 457

    are treated as the same identifier.
    """

    title = result.get("title", "")
    url = result.get("url", "")
    snippet = result.get("snippet", "")

    text = " ".join(
        [
            title,
            url,
            snippet,
        ]
    ).lower()

    # Normalize organization numbers.
    company_number_normalized = (
        company_number.lower()
        .replace(" ", "")
        .replace("-", "")
    )

    text_number_normalized = (
        text
        .replace(" ", "")
        .replace("-", "")
    )

    company_name_lower = company_name.lower()

    # Strongest identity signal:
    # exact organization number, allowing spaces/hyphens.
    if company_number_normalized in text_number_normalized:
        return True

    # Strong identity signal:
    # exact registered company name.
    if company_name_lower in text:
        return True

    # Official registered domain.
    if official_domain:
        result_domain = get_domain(url)

        normalized_official = (
            official_domain
            .lower()
            .replace("www.", "")
        )

        if result_domain == normalized_official:
            return True

    return False


def discover_company_sources(
    company_name: str,
    company_number: str,
    website: str | None = None,
) -> list[dict]:
    """
    Discover public company sources.

    Request-efficient path:

        Brreg
          ↓
        TinyFish search
          ↓
        Identity verification
          ↓
        Source classification
          ↓
        Source ranking
          ↓
        Verified snippets

    The registered website is NOT fetched here.
    """

    sources = []
    existing_urls = set()

    queries = build_search_queries(
        company_name=company_name,
        company_number=company_number,
    )

    official_domain = (
        get_domain(website)
        if website
        else None
    )

    # ---------------------------------------------------------
    # TinyFish primary search
    # ---------------------------------------------------------
    try:
        results = search_tinyfish(
            query=queries[0],
            max_results=5,
        )

        for result in results:

            url = result.get("url")

            if not url or url in existing_urls:
                continue

            identity_verified = verify_search_result_identity(
                result=result,
                company_name=company_name,
                company_number=company_number,
                official_domain=official_domain,
            )

            if not identity_verified:
                continue

            sources.append(
                {
                    "url": url,
                    "title": result.get("title", ""),
                    "domain": get_domain(url),
                    "source_type": classify_source_type(url),
                    "identity_verified": True,
                    "snippet": result.get("snippet", ""),
                }
            )

            existing_urls.add(url)

    except Exception:
        pass

    # ---------------------------------------------------------
    # Return TinyFish results if available
    # ---------------------------------------------------------
    if sources:
        return rank_sources(sources)

    # ---------------------------------------------------------
    # LangSearch fallback
    # ---------------------------------------------------------
    try:
        results = search_web(
            query=queries[0],
            max_results=5,
            company_name=company_name,
            company_number=company_number,
        )

        for result in results:

            url = result.get("url")

            if not url or url in existing_urls:
                continue

            identity_verified = verify_search_result_identity(
                result=result,
                company_name=company_name,
                company_number=company_number,
                official_domain=official_domain,
            )

            if not identity_verified:
                continue

            sources.append(
                {
                    "url": url,
                    "title": result.get("title", ""),
                    "domain": get_domain(url),
                    "source_type": classify_source_type(url),
                    "identity_verified": True,
                    "snippet": result.get("snippet", ""),
                }
            )

            existing_urls.add(url)

    except Exception:
        pass

    return rank_sources(sources)