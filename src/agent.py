from pydantic import BaseModel
from typing import Optional


class CompanyProfile(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    industry: Optional[str] = None
    registration_date: Optional[str] = None
    sources: list[str] = []
def validate_company_number(company_number: str) -> bool:
    return company_number.isdigit() and len(company_number) == 9
from src.brreg import format_address, get_company


def build_profile(company_number: str) -> CompanyProfile:
    data = get_company(company_number)

    return CompanyProfile(
        company_number=data["company_number"],
        company_name=data["company_name"],
        address=format_address(data.get("forretningsadresse", {})),
        industry=data["industry"]["beskrivelse"],
        registration_date=data["registration_date"],
        sources=[data["source"]],
    )