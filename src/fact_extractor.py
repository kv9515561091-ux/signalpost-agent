import re


def extract_facts(
    evidence_text: str,
    company_name: str,
    website: str | None = None,
) -> dict:
    """
    Extract conservative, directly observable company facts.
    """

    facts = {}

    text = " ".join(evidence_text.split())

    # Email addresses
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text,
    )

    if email_match:
        facts["contact_email"] = email_match.group(0)

    # Telephone numbers
    phone_match = re.search(
        r"(?:\+47\s?)?\d{2}(?:[\s-]?\d{2}){3}",
        text,
    )

    if phone_match:
        facts["phone"] = phone_match.group(0)

    # Website
    if website:
        facts["website"] = website

    return facts