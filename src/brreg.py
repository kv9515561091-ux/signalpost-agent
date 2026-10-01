import requests
from datetime import datetime, timezone
from src.request_tracker import request_tracker

def format_address(address: dict) -> str:
    street = ", ".join(address.get("adresse", []))
    postal = address.get("postnummer", "")
    city = address.get("poststed", "")
    country = address.get("land", "")

    return ", ".join(
        part for part in [street, postal, city, country] if part
    )

def get_company(company_number: str) -> dict:
    url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{company_number}"
    request_tracker.record()
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return {
        "company_number": data.get("organisasjonsnummer"),
        "company_name": data.get("navn"),
        "address": format_address(data.get("forretningsadresse", {})),
        "registration_date": data.get("registreringsdatoEnhetsregisteret"),
        "industry": data.get("naeringskode1"),
        "phone": data.get("telefon"),
        "website": data.get("hjemmeside"),

        "under_liquidation": data.get("underAvvikling"),
        "under_forced_liquidation": data.get(
            "underTvangsavviklingEllerTvangsopplosning"
        ),

        "registered_business_register": data.get(
            "registrertIForetaksregisteret"
        ),
        "registered_vat_register": data.get(
            "registrertIMvaregisteret"
        ),
        "registered_voluntary_register": data.get(
            "registrertIFrivillighetsregisteret"
        ),

        "parent_company_number": data.get("overordnetEnhet"),

        "source": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }