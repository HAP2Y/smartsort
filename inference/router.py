"""Decide which worker queue a file should land on.

Routing only handles cases the **prefilter could not classify locally**.
The cheap filename-rules path runs on the dispatcher before this code ever
sees a file, so the router's job is narrowly: of the remaining "needs a
worker" files, pick the right AI / OCR queue based on file traits.

The router is a pure function over file traits. Keeping it data-driven
(``RouteRule`` list) means routes can later be loaded from config without
touching code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from classifier.types import FileItem

ROUTE_AI_SMALL = "ai-small"
ROUTE_AI_LARGE = "ai-large"
ROUTE_OCR = "ocr"
# Sentinel: file landed here means the dispatcher should classify it as
# Unknown_Unsorted locally rather than enqueueing — e.g. an image when no
# OCR worker is implemented yet. Removes the need for an OCR placeholder
# worker that just times out.
ROUTE_UNROUTABLE = "unroutable"
# Legacy: kept so external tests / callers can construct stub workers on a
# `rules` queue. Never produced by the default router any more — cheap
# rules classification happens on the dispatcher via Prefilter.
ROUTE_RULES = "rules"

# Files larger than this go to the large-context model.
LARGE_FILE_BYTES = 2 * 1024 * 1024
EXTRACTABLE_EXTS = {".pdf", ".docx", ".txt", ".md", ".csv", ".rtf", ".html", ".eml", ".json", ".xml", ".log"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class RouteRule:
    """A single ``predicate -> route`` mapping.

    First matching rule wins. Predicates are pure functions of FileItem
    plus the file's size in bytes (passed in so we don't stat the file twice).
    """
    route: str
    predicate: Callable[[FileItem, int], bool]
    note: str = ""


@dataclass
class Router:
    """First-match router over a configurable rule list."""
    rules: list[RouteRule] = field(default_factory=list)
    default_route: str = ROUTE_UNROUTABLE

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
                    note="Small extractable docs go to the fast model.",
                ),
            ],
            default_route=ROUTE_UNROUTABLE,
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
    out.append((r.default_route, "Files the dispatcher can't route to any worker; classified Unknown locally."))
    return out
