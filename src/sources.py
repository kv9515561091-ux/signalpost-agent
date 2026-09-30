from dataclasses import dataclass


@dataclass
class SourceResult:
    url: str
    title: str
    retrieved_at: str
    content: str