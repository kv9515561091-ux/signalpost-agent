from datetime import datetime, timezone

from src.web_search import fetch_page


def extract_page_evidence(
    url: str,
    company_name: str,
) -> dict:
    """
    Fetch a verified company page and extract text surrounding
    the company-name occurrence as evidence.
    """

    page = fetch_page(url)

    content = page["content"]
    content_lower = content.lower()
    company_lower = company_name.lower()

    match_position = content_lower.find(company_lower)

    if match_position >= 0:
        start = max(0, match_position - 500)
        end = min(len(content), match_position + len(company_name) + 1500)
        evidence_text = content[start:end]
        company_match = True
    else:
        evidence_text = content[:2000]
        company_match = False

    retrieved_at = datetime.now(timezone.utc).isoformat()

    return {
        "source_url": page["url"],
        "page_title": page["title"],
        "retrieved_at": retrieved_at,
        "company_match": company_match,
        "evidence_text": evidence_text,
    }