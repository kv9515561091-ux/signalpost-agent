from pydantic import BaseModel
from typing import Optional


class CompanyProfile(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    industry: Optional[str] = None
    sources: list[str] = []
def validate_company_number(company_number: str) -> bool:
    return company_number.isdigit() and len(company_number) == 9