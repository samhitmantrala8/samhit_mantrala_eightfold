from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedFact:
    field: str
    value: Any
    source: str
    method: str
    confidence: float
    evidence: str | None = None


@dataclass
class ExtractionBundle:
    facts: list[ExtractedFact]
    errors: list[str]

