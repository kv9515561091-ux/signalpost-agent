from urllib.parse import urlparse

from src.web_search import fetch_page


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_company_name(name: str) -> str:
    name = normalize_text(name)

    for suffix in [
        " as",
        " asa",
        " aksjeselskap",
        " limited",
        " ltd",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    return name


def company_name_tokens(name: str) -> set[str]:
    normalized = normalize_company_name(name)

    tokens = {
        token
        for token in normalized.replace("-", " ").split()
        if len(token) >= 3
    }

    return tokens


def verify_website_content(
    company_name: str,
    company_number: str,
    website: str,
    page: dict,
) -> dict:
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    parsed = urlparse(website)
    domain = parsed.netloc.lower().replace("www.", "")

    content = normalize_text(page["content"])
    title = normalize_text(page["title"])

    normalized_name = normalize_company_name(company_name)

    name_match = (
        normalized_name in content
        or normalized_name in title
    )

    number_match = (
        company_number in content
        or company_number in title
    )

    tokens = company_name_tokens(company_name)

    matched_tokens = {
        token
        for token in tokens
        if token in content or token in title
    }

    token_match = len(matched_tokens) >= 1

    verified = (
        number_match
        or name_match
        or token_match
    )

    return {
        "verified": verified,
        "website": website,
        "domain": domain,
        "name_match": name_match,
        "company_number_match": number_match,
        "token_match": token_match,
        "matched_name_tokens": sorted(matched_tokens),
        "source_url": page["url"],
        "page_title": page["title"],
        "page_content": page["content"],
    }


def verify_website(
    company_name: str,
    company_number: str,
    website: str,
) -> dict:
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    page = fetch_page(website)

    return verify_website_content(
        company_name=company_name,
        company_number=company_number,
        website=website,
        page=page,
    )