import requests
from datetime import datetime, timezone

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

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return {
        "company_number": data.get("organisasjonsnummer"),
        "company_name": data.get("navn"),
        "address": format_address(data.get("forretningsadresse", {})),
        "registration_date": data.get("registreringsdatoEnhetsregisteret"),
        "industry": data.get("naeringskode1"),
        "website": data.get("hjemmeside"),
        "source": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }