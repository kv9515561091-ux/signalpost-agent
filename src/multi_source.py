from src.source_discovery import discover_sources
from src.source_hierarchy import rank_sources


def discover_company_sources(
    company_name: str,
    company_number: str,
    website: str | None = None,
) -> list[dict]:
    """
    Discover, verify, and rank public company sources.
    """

    sources = discover_sources(
        company_name=company_name,
        company_number=company_number,
        website=website,
    )

    return rank_sources(sources)