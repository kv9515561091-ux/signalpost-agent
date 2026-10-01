from src.identity import verify_website


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
            verification = verify_website(
                company_name=company_name,
                company_number=company_number,
                website=website,
            )

            if verification["verified"]:
                sources.append(
                    {
                        "url": verification["source_url"],
                        "title": verification["page_title"],
                        "domain": verification["domain"],
                        "source_type": "official_company_website",
                        "identity_verified": True,
                    }
                )

        except Exception:
            pass

    return sources