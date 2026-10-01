from typing import Optional

from pydantic import BaseModel, Field

from src.brreg import get_company
from src.evidence import Evidence
from src.fact_extractor import extract_facts
from src.fact_record import FactRecord
from src.multi_source import discover_company_sources


class CompanyProfile(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    industry_code: Optional[str] = None
    industry: Optional[str] = None
    registration_date: Optional[str] = None

    phone: Optional[str] = None
    under_liquidation: Optional[bool] = None
    under_forced_liquidation: Optional[bool] = None
    registered_business_register: Optional[bool] = None
    registered_vat_register: Optional[bool] = None
    registered_voluntary_register: Optional[bool] = None
    parent_company_number: Optional[str] = None

    sources: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)


def validate_company_number(company_number: str) -> bool:
    return company_number.isdigit() and len(company_number) == 9


def build_profile(company_number: str) -> CompanyProfile:

    if not validate_company_number(company_number):
        raise ValueError(
            "Company number must contain exactly 9 digits"
        )

    # ---------------------------------------------------------
    # Primary source: Brreg
    # ---------------------------------------------------------
    data = get_company(company_number)

    source_url = data["source"]
    retrieved_at = data["retrieved_at"]

    evidence = [
        Evidence(
            field="company_number",
            value=str(data["company_number"]),
            source_url=source_url,
            retrieved_at=retrieved_at,
            evidence_text=(
                f"Brreg registered organization number: "
                f"{data['company_number']}"
            ),
            confidence=1.0,
        ),
        Evidence(
            field="company_name",
            value=data["company_name"],
            source_url=source_url,
            retrieved_at=retrieved_at,
            evidence_text=(
                f"Brreg registered company name: "
                f"{data['company_name']}"
            ),
            confidence=1.0,
        ),
    ]

    # ---------------------------------------------------------
    # Address
    # ---------------------------------------------------------
    if data.get("address"):
        evidence.append(
            Evidence(
                field="address",
                value=data["address"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=(
                    f"Brreg registered business address: "
                    f"{data['address']}"
                ),
                confidence=1.0,
            )
        )

    # ---------------------------------------------------------
    # Registration date
    # ---------------------------------------------------------
    if data.get("registration_date"):
        evidence.append(
            Evidence(
                field="registration_date",
                value=data["registration_date"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=(
                    f"Brreg registration date: "
                    f"{data['registration_date']}"
                ),
                confidence=1.0,
            )
        )

    # ---------------------------------------------------------
    # Industry
    # ---------------------------------------------------------
    industry = data.get("industry") or {}

    industry_code = industry.get("kode")
    industry_description = industry.get("beskrivelse")

    if industry_description:
        evidence.append(
            Evidence(
                field="industry",
                value=industry_description,
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=(
                    f"Brreg industry classification: "
                    f"{industry_description}"
                ),
                confidence=1.0,
            )
        )

    # ---------------------------------------------------------
    # Website
    # ---------------------------------------------------------
    if data.get("website"):
        evidence.append(
            Evidence(
                field="website",
                value=data["website"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=(
                    f"Brreg registered website: "
                    f"{data['website']}"
                ),
                confidence=1.0,
            )
        )

    # ---------------------------------------------------------
    # Other Brreg fields
    # ---------------------------------------------------------
    brreg_fields = [
        (
            "phone",
            data.get("phone"),
            "Brreg registered phone number",
        ),
        (
            "under_liquidation",
            data.get("under_liquidation"),
            "Brreg liquidation status",
        ),
        (
            "under_forced_liquidation",
            data.get("under_forced_liquidation"),
            "Brreg forced liquidation status",
        ),
        (
            "registered_business_register",
            data.get("registered_business_register"),
            "Brreg business register registration",
        ),
        (
            "registered_vat_register",
            data.get("registered_vat_register"),
            "Brreg VAT register registration",
        ),
        (
            "registered_voluntary_register",
            data.get("registered_voluntary_register"),
            "Brreg voluntary register registration",
        ),
        (
            "parent_company_number",
            data.get("parent_company_number"),
            "Brreg parent company organization number",
        ),
    ]

    for field, value, description in brreg_fields:

        if value is not None:
            evidence.append(
                Evidence(
                    field=field,
                    value=str(value),
                    source_url=source_url,
                    retrieved_at=retrieved_at,
                    evidence_text=f"{description}: {value}",
                    confidence=1.0,
                )
            )

    # ---------------------------------------------------------
    # Secondary source discovery
    # ---------------------------------------------------------
    discovered_sources = discover_company_sources(
        company_name=data["company_name"],
        company_number=str(data["company_number"]),
        website=data.get("website"),
    )

    verified_source_urls = [
        source["url"]
        for source in discovered_sources
        if source.get("identity_verified")
    ]

    all_sources = list(
        dict.fromkeys(
            [source_url] + verified_source_urls
        )
    )

    # ---------------------------------------------------------
    # Secondary fact extraction
    #
    # Collect facts from all verified snippets.
    # Only one value is retained for each field.
    # Higher-ranked sources are processed first.
    # ---------------------------------------------------------
    facts = []
    seen_fields = set()

    target_company_number = str(
        data["company_number"]
    ).lower()

    target_company_name = data["company_name"].lower()

    for source in discovered_sources:

        if not source.get("identity_verified"):
            continue

        snippet = source.get("snippet", "").strip()

        if not snippet:
            continue

        snippet_lower = snippet.lower()

        # Require identity evidence in the SAME snippet.
        snippet_identifies_company = (
            target_company_number in snippet_lower
            or target_company_name in snippet_lower
        )

        if not snippet_identifies_company:
            continue

        extracted_facts = extract_facts(
            snippet,
            data["company_name"],
        )

        for field, value in extracted_facts.items():

            if value is None:
                continue

            # Keep the first value encountered.
            # Sources are already ranked by priority.
            if field in seen_fields:
                continue

            seen_fields.add(field)

            facts.append(
                FactRecord(
                    field=field,
                    value=str(value),
                    source_url=source["url"],
                    retrieved_at=retrieved_at,
                    evidence_text=snippet,
                    confidence=0.90,
                )
            )

    # ---------------------------------------------------------
    # Final company profile
    # ---------------------------------------------------------
    return CompanyProfile(
        company_number=str(data["company_number"]),
        company_name=data["company_name"],
        website=data["website"],
        address=data["address"],
        industry_code=industry_code,
        industry=industry_description,
        registration_date=data["registration_date"],
        phone=data.get("phone"),
        under_liquidation=data.get("under_liquidation"),
        under_forced_liquidation=data.get(
            "under_forced_liquidation"
        ),
        registered_business_register=data.get(
            "registered_business_register"
        ),
        registered_vat_register=data.get(
            "registered_vat_register"
        ),
        registered_voluntary_register=data.get(
            "registered_voluntary_register"
        ),
        parent_company_number=data.get(
            "parent_company_number"
        ),
        sources=all_sources,
        evidence=evidence,
        facts=facts,
    )