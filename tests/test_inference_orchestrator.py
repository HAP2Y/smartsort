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
    ROUTE_UNROUTABLE,
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
        ROUTE_UNROUTABLE: "rules-stub",
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
    assert orchestrator.stats.by_route[ROUTE_UNROUTABLE] == 1

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


def test_orchestrator_invokes_on_result_callback(tmp_path):
    f1 = tmp_path / "a.pdf"; f1.write_bytes(b"x")
    f2 = tmp_path / "b.bin"; f2.write_bytes(b"x")

    backend = InMemoryQueueBackend()
    workers = _spin_workers(backend, {
        ROUTE_UNROUTABLE: "rules", ROUTE_AI_SMALL: "small",
        ROUTE_AI_LARGE: "large", ROUTE_OCR: "ocr",
    })

    seen: list = []
    def cb(result, classification):
        seen.append((result.file_path, classification.category))

    orchestrator = Orchestrator(backend=backend, router=Router.default())
    pending = orchestrator.submit([f1, f2])
    orchestrator.collect(pending, timeout=5.0, poll=0.1, on_result=cb)

    assert len(seen) == 2
    assert {p for p, _ in seen} == {str(f1), str(f2)}

    for w in workers: w.stop()


def test_orchestrator_ignores_stray_results(tmp_path):
    """Results for unknown job_ids (e.g. a previous run on a shared Redis
    stream) must be silently skipped, not crash the collector."""
    # .bin lands on the rules route via the default router fallback.
    f = tmp_path / "a.bin"; f.write_bytes(b"x")

    backend = InMemoryQueueBackend()
    # Publish a stray result before any job is submitted.
    from inference.types import JobResult
    backend.publish_result(JobResult(
        job_id="stray-id-from-prior-run",
        file_path="/tmp/old.pdf",
        route="rules",
        worker_id="ghost",
        duration_ms=1.0,
        classification={"category": "Foo", "confidence": 99, "method": "X", "reason": "y"},
    ))

    workers = _spin_workers(backend, {ROUTE_UNROUTABLE: "rules"})
    orchestrator = Orchestrator(backend=backend, router=Router.default())
    pending = orchestrator.submit([f])
    plan = orchestrator.collect(pending, timeout=5.0, poll=0.1)

    assert str(f) in plan
    assert plan[str(f)].is_known
    # The stray result must not have been counted as a completion.
    assert orchestrator.stats.completed == 1

    for w in workers: w.stop()


def test_orchestrator_handles_malformed_classification(tmp_path):
    """If a worker publishes a result with a classification dict missing
    required fields, the orchestrator falls back to Unknown_Unsorted rather
    than crashing the whole run."""
    f = tmp_path / "a.pdf"; f.write_bytes(b"x")

    class BadClassifier:
        name = "bad"
        def classify(self, file: FileItem):
            # Return a Classification with a category our fallback parser
            # can't reconstruct (we'll fake it by post-processing the result).
            return Classification(category="Travel_Transit", confidence=90, method="bad", reason="r")

    backend = InMemoryQueueBackend()
    # Run the worker manually and corrupt the published result.
    from inference.types import Job, JobResult
    job = Job(file_path=str(f), route=ROUTE_UNROUTABLE)
    backend.publish_result(JobResult(
        job_id=job.id, file_path=str(f), route=ROUTE_UNROUTABLE,
        worker_id="w", duration_ms=1.0,
        classification={"missing_fields": True},
    ))

    orchestrator = Orchestrator(backend=backend, router=Router.default())
    # Inject the pending job directly so collect() has the expected id.
    pending = {job.id: job}
    plan = orchestrator.collect(pending, timeout=2.0, poll=0.1)

    assert plan[str(f)].category == "Unknown_Unsorted"
    assert "malformed" in plan[str(f)].reason


def test_orchestrator_handles_worker_error_field(tmp_path):
    f = tmp_path / "a.pdf"; f.write_bytes(b"x")

    from inference.types import Job, JobResult
    backend = InMemoryQueueBackend()
    job = Job(file_path=str(f), route=ROUTE_UNROUTABLE)
    backend.publish_result(JobResult(
        job_id=job.id, file_path=str(f), route=ROUTE_UNROUTABLE,
        worker_id="w", duration_ms=1.0, error="RuntimeError: boom",
    ))

    orchestrator = Orchestrator(backend=backend, router=Router.default())
    plan = orchestrator.collect({job.id: job}, timeout=2.0, poll=0.1)

    assert plan[str(f)].category == "Unknown_Unsorted"
    assert "worker error" in plan[str(f)].reason
    assert "boom" in plan[str(f)].reason
    assert orchestrator.stats.failed == 1


def test_orchestrator_by_route_stats_track_each_queue(tmp_path):
    files = []
    for name, size in [("small.pdf", 1024), ("big.pdf", 3 * 1024 * 1024),
                       ("img.png", 1024), ("misc.bin", 1024)]:
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        files.append(p)

    backend = InMemoryQueueBackend()
    orchestrator = Orchestrator(backend=backend, router=Router.default())
    orchestrator.submit(files)

    assert orchestrator.stats.submitted == 4
    assert orchestrator.stats.by_route == {
        ROUTE_AI_SMALL: 1, ROUTE_AI_LARGE: 1, ROUTE_OCR: 1, ROUTE_UNROUTABLE: 1,
    }


def test_orchestrator_submit_with_empty_files_is_a_noop():
    backend = InMemoryQueueBackend()
    orchestrator = Orchestrator(backend=backend, router=Router.default())
    pending = orchestrator.submit([])

    assert pending == {}
    assert orchestrator.stats.submitted == 0
    assert orchestrator.stats.by_route == {}
