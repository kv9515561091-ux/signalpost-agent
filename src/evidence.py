from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.freshness import freshness_status


class Evidence(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    field: str
    value: str
    source_url: str
    retrieved_at: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)

    @computed_field
    @property
    def freshness(self) -> dict:
        return freshness_status(self.retrieved_at)

    def to_dict(self) -> dict:
        return self.model_dump()