"""Evidence and fact models.

Every publishable fact carries the source it came from, when that source
was retrieved, when the fact's subject matter was current, and the
reliability tier of the source.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.freshness import assess

# Which freshness clock applies to each kind of source.
#   live_registry — continuously maintained record, read live; retrieval
#                   time is the guarantee, there is no staleness.
#   registry      — a dated event such as an appointment, where the date
#                   it last changed genuinely matters.
#   financial     — figures that describe a closed reporting period.
FRESHNESS_KIND = {
    "official_registry": "live_registry",
    "official_registry_roles": "registry",
    "official_annual_accounts": "financial",
    "official_company_website": "web",
    "news_source": "news",
    "reputable_business_database": "web",
    "search_result": "web",
}


class _Provenance(BaseModel):
    """Shared provenance fields and freshness derivation."""

    model_config = ConfigDict(validate_assignment=True)

    field: str
    value: Any
    source_url: str
    retrieved_at: str
    evidence_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)

    # 1 = government registry, 2 = first-party website,
    # 3 = official filing, 4 = reputable database, 5 = news, 6 = directory
    source_tier: int = 1
    source_type: str = "official_registry"

    # Date the *fact* describes, when the source states one.
    as_of: str | None = None
    # Fiscal year / reporting period, for financial facts.
    reporting_period: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def freshness(self) -> dict:
        kind = FRESHNESS_KIND.get(self.source_type, "registry")
        return assess(self.as_of, kind)

    def to_dict(self) -> dict:
        return self.model_dump()


class Evidence(_Provenance):
    """A single sourced claim backing one profile field."""


class FactRecord(_Provenance):
    """A structured fact published alongside its provenance."""
