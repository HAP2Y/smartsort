"""End-to-end orchestrator + worker test using the in-memory backend.

A stub classifier stands in for the real Ollama-backed pipeline so the test
runs offline.
"""
from __future__ import annotations

import time
from pathlib import Path

from classifier.types import Classification, FileItem
from inference import InMemoryQueueBackend, Orchestrator, Router, Worker
from inference.router import (
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_RULES,
    RouteRule,
)


class StubClassifier:
    """Classifier that tags every file with the route name."""
    def __init__(self, name: str, label: str):
        self.name = name
        self._label = label

    def classify(self, file: FileItem):
        return Classification(
            category="Travel_Transit",
            confidence=92,
            method=self.name,
            reason=f"stub:{self._label}",
        )


def _spin_workers(backend, routes_to_label: dict[str, str]) -> list[Worker]:
    workers = []
    for route, label in routes_to_label.items():
        w = Worker(
            name=f"w-{route}",
            routes=[route],
            classifier=StubClassifier(name=label, label=label),
            backend=backend,
            poll_timeout=0.1,
        )
        w.run_in_thread()
        workers.append(w)
    return workers


def test_orchestrator_round_trip(tmp_path):
    # Three files spanning every route.
    small = tmp_path / "small.pdf"
    small.write_bytes(b"x" * 1024)
    large = tmp_path / "huge.pdf"
    large.write_bytes(b"x" * (3 * 1024 * 1024))
    img = tmp_path / "scan.png"
    img.write_bytes(b"x" * 1024)
    misc = tmp_path / "weird.bin"
    misc.write_bytes(b"x" * 1024)

    backend = InMemoryQueueBackend()
    workers = _spin_workers(backend, {
        ROUTE_RULES: "rules-stub",
        ROUTE_AI_SMALL: "ai-small-stub",
        ROUTE_AI_LARGE: "ai-large-stub",
        ROUTE_OCR: "ocr-stub",
    })

    orchestrator = Orchestrator(backend=backend, router=Router.default())
    pending = orchestrator.submit([small, large, img, misc])

    assert orchestrator.stats.submitted == 4
    assert orchestrator.stats.by_route[ROUTE_AI_SMALL] == 1
    assert orchestrator.stats.by_route[ROUTE_AI_LARGE] == 1
    assert orchestrator.stats.by_route[ROUTE_OCR] == 1
    assert orchestrator.stats.by_route[ROUTE_RULES] == 1

    plan = orchestrator.collect(pending, timeout=5.0, poll=0.1)

    assert set(plan.keys()) == {str(small), str(large), str(img), str(misc)}
    assert all(c.category == "Travel_Transit" for c in plan.values())
    assert orchestrator.stats.completed == 4
    assert orchestrator.stats.failed == 0

    methods = {plan[str(small)].reason, plan[str(large)].reason}
    assert "stub:ai-small-stub" in methods
    assert "stub:ai-large-stub" in methods

    for w in workers:
        w.stop()


def test_orchestrator_times_out_without_workers(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x" * 1024)
    backend = InMemoryQueueBackend()
    orchestrator = Orchestrator(backend=backend, router=Router.default())
    pending = orchestrator.submit([f])

    started = time.time()
    plan = orchestrator.collect(pending, timeout=0.5, poll=0.1)
    elapsed = time.time() - started

    assert elapsed < 2.0
    assert plan[str(f)].category == "Unknown_Unsorted"
    assert orchestrator.stats.failed == 1


def test_router_rules_are_first_match(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x" * 100)
    # Two rules both match a .pdf; first wins.
    router = Router(
        rules=[
            RouteRule(route="first", predicate=lambda f, s: f.ext == ".pdf"),
            RouteRule(route="second", predicate=lambda f, s: True),
        ]
    )
    assert router.route(FileItem(path=f)) == "first"
