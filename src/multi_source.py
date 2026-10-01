from urllib.parse import urlparse

from src.public_search import build_search_queries
from src.search_provider import search_web
from src.source_discovery import discover_sources
from src.source_hierarchy import rank_sources
from src.web_search import fetch_page


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def verify_source_identity(
    url: str,
    company_name: str,
    company_number: str,
    official_domain: str | None = None,
) -> bool:
    try:
        page = fetch_page(url)

        content = " ".join(
            page["content"].lower().split()
        )

        domain = get_domain(url)

        company_number_lower = company_number.lower()
        company_name_lower = company_name.lower()

        # Company number is valid only when the
        # company name appears in the same context.
        if company_number_lower in content:
            number_position = content.find(
                company_number_lower
            )

            context_start = max(
                0,
                number_position - 500,
            )

            context_end = min(
                len(content),
                number_position
                + len(company_number_lower)
                + 500,
            )

            context = content[
                context_start:context_end
            ]

            if company_name_lower in context:
                return True

        # Official company domain is trusted.
        if official_domain:
            normalized_official = (
                official_domain
                .lower()
                .replace("www.", "")
            )

            if domain == normalized_official:
                return True

        return False

    except Exception:
        return False


def discover_company_sources(
    company_name: str,
    company_number: str,
    website: str | None = None,
) -> list[dict]:

    sources = discover_sources(
        company_name=company_name,
        company_number=company_number,
        website=website,
    )
    existing_urls = {
        source.get("url")
        for source in sources
        if source.get("url")
    }

    if sources:
        return rank_sources(
            [
                source
                for source in sources
                if source.get("identity_verified")
            ]
        )

    queries = build_search_queries(
        company_name=company_name,
        company_number=company_number,
    )

    for query in queries:
        try:
            results = search_web(
                query=query,
                max_results=5,
                company_name=company_name,
                company_number=company_number,
            )

            for result in results:
                url = result["url"]

                if not url or url in existing_urls:
                    continue

                identity_verified = verify_source_identity(
                    url=url,
                    company_name=company_name,
                    company_number=company_number,
                    official_domain=(
                        get_domain(website)
                        if website
                        else None
                    ),
                )

                sources.append(
                    {
                        "url": url,
                        "title": result["title"],
                        "domain": get_domain(url),
                        "source_type": "search_result",
                        "identity_verified": identity_verified,
                    }
                )

                existing_urls.add(url)

        except Exception:
            continue

    return rank_sources(
        [
            source
            for source in sources
            if source.get("identity_verified")
        ]
    )