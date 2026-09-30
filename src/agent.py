from pydantic import BaseModel
from typing import Optional
class Evidence(BaseModel):
    source_url: str
    retrieved_at: str

class CompanyProfile(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    industry: Optional[str] = None
    registration_date: Optional[str] = None
    sources: list[str] = []
    evidence: list[Evidence] = []
def validate_company_number(company_number: str) -> bool:
    return company_number.isdigit() and len(company_number) == 9
from src.brreg import format_address, get_company


def build_profile(company_number: str) -> CompanyProfile:
    data = get_company(company_number)

    return CompanyProfile(
        company_number=data["company_number"],
        company_name=data["company_name"],
        address=data["address"],
        industry=data["industry"]["beskrivelse"],
        registration_date=data["registration_date"],
        sources=[data["source"]],
        evidence=[
            Evidence(
                source_url=data["source"],
                retrieved_at=data["retrieved_at"],
            )
        ],
    )