"""Decide which queue a file should land on.

The router is a pure function over file traits. Keeping it data-driven
(``RouteRule`` list) means we can later load routes from config without
touching code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from classifier.types import FileItem

ROUTE_RULES = "rules"
ROUTE_AI_SMALL = "ai-small"
ROUTE_AI_LARGE = "ai-large"
ROUTE_OCR = "ocr"

# Files larger than this go to the large-context model.
LARGE_FILE_BYTES = 2 * 1024 * 1024
EXTRACTABLE_EXTS = {".pdf", ".docx", ".txt", ".md", ".csv", ".rtf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class RouteRule:
    """A single ``predicate -> route`` mapping.

    The first matching rule wins. Predicates are pure functions of FileItem
    plus the file's size in bytes (passed in so we don't stat the file twice).
    """
    route: str
    predicate: Callable[[FileItem, int], bool]
    note: str = ""


@dataclass
class Router:
    """First-match router over a configurable rule list."""
    rules: list[RouteRule] = field(default_factory=list)
    default_route: str = ROUTE_RULES

    def route(self, file: FileItem) -> str:
        size = _safe_size(file.path)
        for rule in self.rules:
            if rule.predicate(file, size):
                return rule.route
        return self.default_route

    @classmethod
    def default(cls) -> "Router":
        return cls(
            rules=[
                RouteRule(
                    route=ROUTE_OCR,
                    predicate=lambda f, _s: f.ext in IMAGE_EXTS,
                    note="Images need OCR before any LLM can read them.",
                ),
                RouteRule(
                    route=ROUTE_AI_LARGE,
                    predicate=lambda f, s: f.ext in EXTRACTABLE_EXTS and s >= LARGE_FILE_BYTES,
                    note="Large extractable docs benefit from larger context.",
                ),
                RouteRule(
                    route=ROUTE_AI_SMALL,
                    predicate=lambda f, s: f.ext in EXTRACTABLE_EXTS and s < LARGE_FILE_BYTES,
                    note="Small docs go to the fast model.",
                ),
            ],
            default_route=ROUTE_RULES,
        )


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def describe_routes(router: Optional[Router] = None) -> list[tuple[str, str]]:
    """Human-readable summary of the configured routes (for `--help`-style output)."""
    r = router or Router.default()
    out = [(rule.route, rule.note or "") for rule in r.rules]
    out.append((r.default_route, "Fallback route when no rule matches."))
    return out
