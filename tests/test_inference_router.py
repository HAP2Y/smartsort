"""Router decision tests using temp files."""
from pathlib import Path

from classifier.types import FileItem
from inference.router import (
    LARGE_FILE_BYTES,
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_RULES,
    Router,
)


def _make(path: Path, size: int = 100) -> FileItem:
    path.write_bytes(b"x" * size)
    return FileItem(path=path)


def test_default_router_sends_small_docs_to_ai_small(tmp_path):
    f = _make(tmp_path / "resume.pdf", size=1024)
    assert Router.default().route(f) == ROUTE_AI_SMALL


def test_default_router_sends_large_docs_to_ai_large(tmp_path):
    f = _make(tmp_path / "huge.pdf", size=LARGE_FILE_BYTES + 10)
    assert Router.default().route(f) == ROUTE_AI_LARGE


def test_default_router_sends_images_to_ocr(tmp_path):
    f = _make(tmp_path / "scan.png", size=2048)
    assert Router.default().route(f) == ROUTE_OCR


def test_default_router_falls_back_to_rules(tmp_path):
    f = _make(tmp_path / "binary.bin", size=2048)
    assert Router.default().route(f) == ROUTE_RULES
