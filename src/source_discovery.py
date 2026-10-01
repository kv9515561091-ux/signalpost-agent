from src.identity import verify_website_content
from src.web_search import fetch_page


def discover_sources(
    company_name: str,
    company_number: str,
    website: str | None = None,
) -> list[dict]:
    """
    Discover and validate public sources for a company.

    Brreg remains the primary identity anchor.
    Only identity-verified sources are returned.
    """

    sources = []

    if website:
        try:
            if not website.startswith(
                ("http://", "https://")
            ):
                website = "https://" + website

            page = fetch_page(website)

            verification = verify_website_content(
                company_name=company_name,
                company_number=company_number,
                website=website,
                page=page,
            )

            if verification["verified"]:
                sources.append(
                    {
                        "url": verification["source_url"],
                        "title": verification["page_title"],
                        "domain": verification["domain"],
                        "source_type": "official_company_website",
                        "identity_verified": True,
                        "page_content": verification[
                            "page_content"
                        ],
                    }
                )

        except Exception:
            pass

    return sources