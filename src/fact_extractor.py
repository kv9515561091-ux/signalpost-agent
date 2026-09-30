import re


def extract_facts(
    evidence_text: str,
    company_name: str,
    website: str | None = None,
) -> dict:
    """
    Extract only conservative, directly observable facts.
    """

    facts = {}

    text = " ".join(evidence_text.split())

    # Email
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text,
    )

    if email_match:
        facts["contact_information"] = email_match.group(0)

    return facts