"""Shared data types for the classification pipeline.

Centralising these lets every classifier, the organiser, and the CLI agree on
the same vocabulary, rather than passing dicts and tuples around the codebase.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

UNKNOWN_CATEGORY = "Unknown_Unsorted"
DEFAULT_CONFIDENCE_THRESHOLD = 80


@dataclass(frozen=True)
class FileItem:
    """A file under consideration for classification."""
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def ext(self) -> str:
        return self.path.suffix.lower()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.path)


@dataclass(frozen=True)
class Classification:
    """The verdict of one classifier for one file.

    `confidence` is a 0-100 integer. `method` identifies which classifier
    produced the result so users (and logs) can audit decisions.
    """
    category: str
    confidence: int
    method: str
    reason: str

    @classmethod
    def unknown(cls, reason: str = "No classifier matched", method: str = "None") -> "Classification":
        return cls(category=UNKNOWN_CATEGORY, confidence=0, method=method, reason=reason)

    @property
    def is_known(self) -> bool:
        return self.category != UNKNOWN_CATEGORY

    def is_confident(self, threshold: int = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
        return self.is_known and self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
