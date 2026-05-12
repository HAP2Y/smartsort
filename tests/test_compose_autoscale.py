"""Workload-aware autoscaler math.

`smartsort run --up` pre-routes the file set locally and asks
`_scale_targets` how many AI worker replicas to bring up. The math is
pure and worth pinning so future tuning to `COMPOSE_SCALE` doesn't
accidentally break the contract.

In the AI-first architecture only `ai-small-worker` and
`ai-large-worker` are Compose services — rules run inline as a fallback
inside the AI worker pipeline, and OCR has no dedicated worker yet.
"""
from __future__ import annotations

from pathlib import Path

from inference.router import (
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_UNROUTABLE,
    Router,
)
from main import COMPOSE_SCALE, _route_counts, _scale_targets


def _make_file(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_route_counts_tallies_each_queue(tmp_path):
    small = _make_file(tmp_path / "small.pdf", 1024)
    big = _make_file(tmp_path / "big.pdf", 3 * 1024 * 1024)
    img = _make_file(tmp_path / "scan.png", 1024)
    misc = _make_file(tmp_path / "weird.bin", 1024)

    counts = _route_counts([small, big, img, misc], Router.default())

    # Unmatched files now route to UNROUTABLE (Unknown locally on dispatcher).
    assert counts == {
        ROUTE_AI_SMALL: 1,
        ROUTE_AI_LARGE: 1,
        ROUTE_OCR: 1,
        ROUTE_UNROUTABLE: 1,
    }


def test_scale_targets_keeps_one_warm_worker_per_service_when_idle():
    """No jobs anywhere → still bring up one of each AI service so the
    next dispatch doesn't pay cold-start latency."""
    targets = _scale_targets({})
    assert targets == {
        "ai-small-worker": 1,
        "ai-large-worker": 1,
    }


def test_scale_targets_rounds_up_at_saturation_threshold():
    """At the files-per-worker threshold we still want one worker (ceil(N/N) = 1).
    Just above it we want two."""
    fpw = COMPOSE_SCALE["ai-small-worker"]["files_per_worker"]
    assert _scale_targets({ROUTE_AI_SMALL: fpw})["ai-small-worker"] == 1
    assert _scale_targets({ROUTE_AI_SMALL: fpw + 1})["ai-small-worker"] == 2


def test_scale_targets_caps_at_max_replicas():
    """A huge workload mustn't try to bring up unbounded workers."""
    huge = 10_000
    targets = _scale_targets({
        ROUTE_AI_SMALL: huge,
        ROUTE_AI_LARGE: huge,
    })
    assert targets["ai-small-worker"] == COMPOSE_SCALE["ai-small-worker"]["max_workers"]
    assert targets["ai-large-worker"] == COMPOSE_SCALE["ai-large-worker"]["max_workers"]


def test_scale_targets_ignores_ocr_and_unroutable_routes():
    """OCR + UNROUTABLE files are handled on the dispatcher (Unknown
    locally) — they must not influence the AI worker scale."""
    targets = _scale_targets({
        ROUTE_OCR: 50,
        ROUTE_UNROUTABLE: 100,
    })
    # Falls back to the warm-worker baseline — neither AI route has work.
    assert targets == {"ai-small-worker": 1, "ai-large-worker": 1}


def test_scale_targets_uses_workload_specific_replica_counts(tmp_path):
    """The 134-file split from the debugging run, after the rules+ocr
    queues were dropped: only AI counts drive scale now."""
    counts = {
        ROUTE_AI_SMALL: 92,
        ROUTE_AI_LARGE: 6,
        ROUTE_OCR: 10,           # → unrouted, no influence
        ROUTE_UNROUTABLE: 26,    # → no influence
    }
    targets = _scale_targets(counts)

    # ai-small: ceil(92 / 25) = 4 but capped at max_workers=2.
    assert targets["ai-small-worker"] == COMPOSE_SCALE["ai-small-worker"]["max_workers"]
    # ai-large: ceil(6 / 10) = 1.
    assert targets["ai-large-worker"] == 1
