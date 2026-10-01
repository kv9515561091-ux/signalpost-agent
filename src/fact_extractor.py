import re


def extract_facts(
    evidence_text: str,
    company_name: str,
    website: str | None = None,
) -> dict:
    facts = {}

    # Normalize whitespace
    text = " ".join(evidence_text.split())

    # ---------------------------------------------------------
    # Email
    # ---------------------------------------------------------
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text,
    )

    if email_match:
        facts["contact_email"] = email_match.group(0)

    # ---------------------------------------------------------
    # Norwegian telephone number
    # ---------------------------------------------------------
    phone_patterns = [
        r"\+47\s?\d{3}\s?\d{2}\s?\d{3}",
        r"\+47\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}",
        r"\b\d{3}\s?\d{2}\s?\d{3}\b",
        r"\b\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\b",
    ]

    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)

        if phone_match:
            facts["phone"] = phone_match.group(0)
            break

    # ---------------------------------------------------------
    # Norwegian organization number
    # ---------------------------------------------------------
    org_match = re.search(
        r"(?:organisasjonsnummer|org\.?\s*nr\.?)"
        r"\s*:?\s*(\d{3}\s?\d{3}\s?\d{3})",
        text,
        flags=re.IGNORECASE,
    )

    if org_match:
        facts["organization_number"] = (
            org_match.group(1).replace(" ", "")
        )

    # ---------------------------------------------------------
    # Postal address postal code
    #
    # Example:
    # Postadresse: Postboks 8019 Dep, 0030 Oslo
    # ---------------------------------------------------------
    postal_address_match = re.search(
        r"(?:postadresse|postal\s+address)"
        r".*?postboks\s+\d{4}"
        r".*?\b(\d{4})\s+[A-Za-zÆØÅæøå]",
        text,
        flags=re.IGNORECASE,
    )

    if postal_address_match:
        facts["postal_code"] = postal_address_match.group(1)

    # ---------------------------------------------------------
    # Business address postal code
    #
    # Example:
    # Forretningsadresse: Akersgata 64 0180 Oslo
    # ---------------------------------------------------------
    business_address_match = re.search(
        r"(?:forretningsadresse|business\s+address)"
        r".*?\b(\d{4})\s+[A-Za-zÆØÅæøå]",
        text,
        flags=re.IGNORECASE,
    )

    if business_address_match:
        facts["business_postal_code"] = (
            business_address_match.group(1)
        )

    # ---------------------------------------------------------
    # Generic Norwegian address
    #
    # Example:
    # Adresse: Akersgata 64, 0180 Oslo
    #
    # Only capture when explicitly introduced by "Adresse".
    # ---------------------------------------------------------
    generic_address_match = re.search(
        r"(?:adresse|address)"
        r"\s*:\s*"
        r"([^.;]+?)"
        r"(?:,\s*|\s+)"
        r"(\d{4})\s+"
        r"([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå\-]*)",
        text,
        flags=re.IGNORECASE,
    )

    if generic_address_match:
        street = generic_address_match.group(1).strip(" ,")
        postal_code = generic_address_match.group(2)
        city = generic_address_match.group(3)

        facts["business_address"] = (
            f"{street}, {postal_code} {city}"
        )

        facts["business_postal_code"] = postal_code

    # ---------------------------------------------------------
    # Website
    # ---------------------------------------------------------
    if website:
        facts["website"] = website

    return facts