from dataclasses import dataclass


@dataclass
class Evidence:
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
        }