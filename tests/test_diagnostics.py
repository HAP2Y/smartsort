"""Diagnostics helpers — pre-flight, ProgressReporter, post-mortem.

The Redis / Ollama / docker-compose probes need a real environment to
exercise end-to-end, so we test the contract (return shape, ok flag,
message format) without requiring those services.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from inference.diagnostics import (
    PreflightCheck,
    ProgressReporter,
    preflight_ollama,
    preflight_redis,
    preflight_workers,
    queue_depths,
)
from inference.types import JobResult


# ---------------------------------------------------------------- preflight


def test_preflight_check_status_glyph():
    ok = PreflightCheck("foo", True, "fine")
    bad = PreflightCheck("foo", False, "broken")
    assert ok.status == "✓"
    assert bad.status == "✗"


def test_preflight_redis_returns_failure_when_unreachable():
    # Use a port nothing is listening on.
    result = preflight_redis("redis://127.0.0.1:1/0")
    assert result.name == "redis"
    assert result.ok is False
    assert "127.0.0.1:1" in result.detail


def test_preflight_ollama_returns_failure_when_unreachable():
    result = preflight_ollama("http://127.0.0.1:1", timeout=0.5)
    assert result.name == "ollama"
    assert result.ok is False
    assert "127.0.0.1:1" in result.detail


def test_preflight_workers_returns_failure_when_redis_dead():
    """One check per route, all failing if the redis client can't init."""
    checks = preflight_workers("redis://127.0.0.1:1/0", ["rules", "ai-small"])
    assert len(checks) == 2
    assert {c.name for c in checks} == {"workers/rules", "workers/ai-small"}
    assert all(not c.ok for c in checks)


# -------------------------------------------------------------- progress


def _result(route: str, ok: bool = True) -> JobResult:
    return JobResult(
        job_id=f"j-{route}-{ok}",
        file_path=f"/tmp/{route}.bin",
        route=route,
        worker_id="w",
        duration_ms=1.0,
        classification={"category": "X", "confidence": 90, "method": "M", "reason": ""} if ok else None,
        error=None if ok else "boom",
    )


def test_progress_reporter_counts_completions_and_failures():
    msgs = []
    p = ProgressReporter(
        expected_total=4,
        expected_by_route={"rules": 2, "ai-small": 2},
        tick_seconds=10.0,  # avoid auto-tick during the test
        _printer=msgs.append,
    )
    p.on_result(_result("rules"), None)
    p.on_result(_result("rules"), None)
    p.on_result(_result("ai-small"), None)
    p.on_result(_result("ai-small", ok=False), None)

    assert p.completed == 4
    assert p.failed == 1
    assert p.completed_by_route == {"rules": 2, "ai-small": 2}


def test_progress_reporter_tick_emits_summary_string():
    msgs = []
    p = ProgressReporter(
        expected_total=2,
        expected_by_route={"rules": 1, "ai-small": 1},
        tick_seconds=0.0,  # tick on every result
        _printer=msgs.append,
    )
    p.on_result(_result("rules"), None)
    p.on_result(_result("ai-small"), None)
    assert msgs, "expected at least one tick"
    last = msgs[-1]
    assert "rules 1/1" in last
    assert "ai-small 1/1" in last
    assert "2/2" in last


def test_progress_reporter_final_tick_runs_even_with_no_periodic_ticks():
    msgs = []
    p = ProgressReporter(
        expected_total=1,
        expected_by_route={"rules": 1},
        tick_seconds=10_000,
        _printer=msgs.append,
    )
    p.on_result(_result("rules"), None)
    assert msgs == []  # no auto-tick yet
    p.final()
    assert len(msgs) == 1
    assert "rules 1/1" in msgs[0]


def test_progress_reporter_handles_unexpected_route_gracefully():
    """A result for a route not in expected_by_route should still count
    in `completed` (the orchestrator already filters strays elsewhere)."""
    p = ProgressReporter(
        expected_total=1,
        expected_by_route={"rules": 1},
        tick_seconds=10_000,
        _printer=lambda _: None,
    )
    p.on_result(_result("ghost-route"), None)
    assert p.completed == 1
    assert p.completed_by_route == {"ghost-route": 1}


# ------------------------------------------------------------- queue_depths


def test_queue_depths_with_in_memory_backend_returns_empty():
    """The in-memory backend has no `_redis` attribute; helper should
    return an empty dict rather than raise."""
    from inference.queue import InMemoryQueueBackend
    backend = InMemoryQueueBackend()
    assert queue_depths(backend, ["rules", "ai-small"]) == {}


def test_queue_depths_uses_redis_xlen_when_backend_has_redis():
    fake_redis = MagicMock()
    fake_redis.xlen.side_effect = [4, 0, 7]

    backend = MagicMock()
    backend._redis = fake_redis

    out = queue_depths(backend, ["rules", "ai-small", "ai-large"])

    assert out == {"rules": 4, "ai-small": 0, "ai-large": 7}
    # Confirm the stream prefix is correctly applied.
    calls = [c.args[0] for c in fake_redis.xlen.call_args_list]
    assert calls == [
        "smartsort:jobs:rules",
        "smartsort:jobs:ai-small",
        "smartsort:jobs:ai-large",
    ]


def test_queue_depths_returns_minus_one_on_xlen_failure():
    fake_redis = MagicMock()
    fake_redis.xlen.side_effect = RuntimeError("connection lost")
    backend = MagicMock()
    backend._redis = fake_redis

    out = queue_depths(backend, ["rules"])
    assert out == {"rules": -1}
