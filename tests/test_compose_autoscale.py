"""Workload-aware autoscaler math.

`smartsort run --up` pre-routes the file set locally and asks
`_scale_targets` how many replicas to bring up per service. The math is
pure and worth pinning so future tuning to `COMPOSE_SCALE` doesn't
accidentally break the contract.
"""
from __future__ import annotations

from pathlib import Path

from inference.router import (
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_RULES,
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

    assert counts == {
        ROUTE_AI_SMALL: 1,
        ROUTE_AI_LARGE: 1,
        ROUTE_OCR: 1,
        ROUTE_RULES: 1,
    }


def test_scale_targets_keeps_one_warm_worker_per_service_when_idle():
    """No jobs anywhere → still bring up one of each (cold start latency
    matters more than container count when the user is going to submit
    again in a moment)."""
    targets = _scale_targets({})
    assert targets == {
        "rules-worker": 1,
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
        ROUTE_RULES: huge,
        ROUTE_OCR: huge,
    })
    assert targets["ai-small-worker"] == COMPOSE_SCALE["ai-small-worker"]["max_workers"]
    assert targets["ai-large-worker"] == COMPOSE_SCALE["ai-large-worker"]["max_workers"]
    assert targets["rules-worker"]    == COMPOSE_SCALE["rules-worker"]["max_workers"]


def test_rules_worker_scale_combines_rules_and_ocr_counts():
    """The rules-worker container subscribes to BOTH the rules and ocr
    queues, so its replica count must reflect the sum of those queues."""
    fpw = COMPOSE_SCALE["rules-worker"]["files_per_worker"]
    # Split the load so neither queue alone would trigger scale-up but the
    # combined load does.
    counts = {
        ROUTE_RULES: fpw // 2 + 1,
        ROUTE_OCR:   fpw // 2 + 1,
    }
    targets = _scale_targets(counts)
    assert targets["rules-worker"] == 2  # combined > fpw → ceil(N/fpw) = 2


def test_scale_targets_uses_workload_specific_replica_counts(tmp_path):
    """A real example: 92 ai-small jobs, 6 ai-large, 26 rules, 10 ocr —
    the exact split from the user's debugging run."""
    counts = {
        ROUTE_AI_SMALL: 92,
        ROUTE_AI_LARGE: 6,
        ROUTE_RULES: 26,
        ROUTE_OCR: 10,
    }
    targets = _scale_targets(counts)

    # ai-small: 92 / 25 = 3.68 → ceil = 4, under cap of 6
    assert targets["ai-small-worker"] == 4
    # ai-large: 6 / 10 = 0.6 → ceil = 1
    assert targets["ai-large-worker"] == 1
    # rules-worker handles rules + ocr = 36 / 100 = 0.36 → ceil = 1
    assert targets["rules-worker"] == 1
