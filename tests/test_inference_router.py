"""Router decision tests using temp files."""
from pathlib import Path

from classifier.types import FileItem
from inference.router import (
    LARGE_FILE_BYTES,
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_UNROUTABLE,
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


def test_default_router_falls_back_to_unroutable(tmp_path):
    """Files with no matching rule must fall through to UNROUTABLE so the
    dispatcher can classify them Unknown locally rather than enqueuing
    work nobody will drain."""
    f = _make(tmp_path / "binary.bin", size=2048)
    assert Router.default().route(f) == ROUTE_UNROUTABLE


def test_router_handles_missing_file_via_safe_size(tmp_path):
    """File doesn't exist (or stat fails) → treat as size=0.

    The router should still route a non-existent file by extension rather
    than blowing up — this lets the dispatcher submit jobs even if a file
    is racing against deletion.
    """
    from classifier.types import FileItem

    missing = tmp_path / "ghost.pdf"  # never created
    assert Router.default().route(FileItem(path=missing)) == ROUTE_AI_SMALL


def test_describe_routes_reports_every_rule_plus_default():
    from inference.router import describe_routes
    summary = describe_routes()
    routes_listed = [route for route, _ in summary]

    assert ROUTE_OCR in routes_listed
    assert ROUTE_AI_LARGE in routes_listed
    assert ROUTE_AI_SMALL in routes_listed
    # Default (last) entry is UNROUTABLE in the new architecture.
    assert routes_listed[-1] == ROUTE_UNROUTABLE
    assert all(isinstance(note, str) for _, note in summary)
