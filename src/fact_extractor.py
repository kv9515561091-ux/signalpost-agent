import re


def extract_facts(
    evidence_text: str,
    company_name: str,
    website: str | None = None,
) -> dict:
    """
    Extract conservative, directly observable facts from page evidence.
    """

    facts = {
        "company_name": company_name,
        "website": website,
        "description": None,
        "contact_information": None,
    }

    text = " ".join(evidence_text.split())

    # Email
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text,
    )

    if email_match:
        facts["contact_information"] = email_match.group(0)

    # Conservative description extraction.
    description_patterns = [
        r"(?i)(?:om oss|about us|om nav)[\s:]+(.{40,500})",
        r"(?i)(?:vi tilbyr|we provide|our services)[\s:]+(.{40,500})",
    ]

    for pattern in description_patterns:
        match = re.search(pattern, text)

        if match:
            facts["description"] = match.group(1).strip()
            break

    return facts