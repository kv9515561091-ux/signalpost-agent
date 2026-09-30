from dataclasses import dataclass

from src.freshness import freshness_status


@dataclass
class FactRecord:
    field: str
    value: str
    source_url: str
    retrieved_at: str
    evidence_text: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "freshness": freshness_status(self.retrieved_at),
        }