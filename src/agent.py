from typing import Optional

from pydantic import BaseModel

from src.brreg import get_company
from src.evidence import Evidence
from src.fact_extractor import extract_facts
from src.fact_record import FactRecord


class CompanyProfile(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    industry_code: Optional[str] = None
    industry: Optional[str] = None
    registration_date: Optional[str] = None
    sources: list[str] = []
    evidence: list[Evidence] = []
    facts: list[FactRecord] = []


def validate_company_number(company_number: str) -> bool:
    return company_number.isdigit() and len(company_number) == 9


def build_profile(company_number: str) -> CompanyProfile:
    data = get_company(company_number)

    source_url = data["source"]
    retrieved_at = data["retrieved_at"]

    evidence = [
        Evidence(
            field="company_number",
            value=str(data["company_number"]),
            source_url=source_url,
            retrieved_at=retrieved_at,
            evidence_text=f"Brreg registered organization number: {data['company_number']}",
            confidence=1.0,
        ),
        Evidence(
            field="company_name",
            value=data["company_name"],
            source_url=source_url,
            retrieved_at=retrieved_at,
            evidence_text=f"Brreg registered company name: {data['company_name']}",
            confidence=1.0,
        ),
    ]

    if data.get("address"):
        evidence.append(
            Evidence(
                field="address",
                value=data["address"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=f"Brreg registered business address: {data['address']}",
                confidence=1.0,
            )
        )

    if data.get("registration_date"):
        evidence.append(
            Evidence(
                field="registration_date",
                value=data["registration_date"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=f"Brreg registration date: {data['registration_date']}",
                confidence=1.0,
            )
        )

    industry = data.get("industry") or {}
    industry_description = industry.get("beskrivelse")
    industry_code = industry.get("kode")

    if industry_description:
        evidence.append(
            Evidence(
                field="industry",
                value=industry_description,
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=f"Brreg industry classification: {industry_description}",
                confidence=1.0,
            )
        )

    if data.get("website"):
        evidence.append(
            Evidence(
                field="website",
                value=data["website"],
                source_url=source_url,
                retrieved_at=retrieved_at,
                evidence_text=f"Brreg registered website: {data['website']}",
                confidence=1.0,
            )
        )

    facts = []

    if data.get("website"):
        from src.evidence_extractor import extract_page_evidence

        website = data["website"]

        if not website.startswith(("http://", "https://")):
            website = "https://" + website

        page_evidence = extract_page_evidence(
            website,
            data["company_name"],
        )

        if page_evidence["company_match"]:
            extracted_facts = extract_facts(
                page_evidence["evidence_text"],
                data["company_name"],
                website,
            )

            for field, value in extracted_facts.items():
                if value is not None:
                    facts.append(
                        FactRecord(
                            field=field,
                            value=str(value),
                            source_url=page_evidence["source_url"],
                            retrieved_at=page_evidence["retrieved_at"],
                            evidence_text=page_evidence["evidence_text"],
                            confidence=1.0,
                        )
                    )

    return CompanyProfile(
        company_number=data["company_number"],
        company_name=data["company_name"],
        website=data["website"],
        address=data["address"],
        industry_code=industry_code,
        industry=industry_description,
        registration_date=data["registration_date"],
        sources=[source_url],
        evidence=evidence,
        facts=facts,
    )