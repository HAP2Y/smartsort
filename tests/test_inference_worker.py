"""Worker isolation tests: error capture, ack behaviour."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from classifier.types import Classification, FileItem
from inference import InMemoryQueueBackend, Worker
from inference.types import Job


class BoomClassifier:
    name = "boom"
    def classify(self, file: FileItem):
        raise RuntimeError("kaboom")


class OkClassifier:
    name = "ok"
    def classify(self, file: FileItem):
        return Classification(category="Travel_Transit", confidence=99, method="ok", reason="r")


def _drive(worker: Worker) -> threading.Thread:
    t = worker.run_in_thread()
    return t


def test_worker_reports_classifier_exception_as_error(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x")
    backend = InMemoryQueueBackend()
    w = Worker(name="w", routes=["rules"], classifier=BoomClassifier(), backend=backend, poll_timeout=0.1)
    t = _drive(w)

    backend.enqueue(Job(file_path=str(f), route="rules"))
    res = backend.consume_result(timeout=2.0)
    w.stop()
    t.join(timeout=2.0)

    assert res is not None
    assert res.error is not None
    assert "kaboom" in res.error
    assert res.classification is None
    assert w.stats.failed == 1
    assert w.stats.processed == 0


def test_worker_publishes_classification(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x")
    backend = InMemoryQueueBackend()
    w = Worker(name="w", routes=["rules"], classifier=OkClassifier(), backend=backend, poll_timeout=0.1)
    t = _drive(w)

    backend.enqueue(Job(file_path=str(f), route="rules"))
    res = backend.consume_result(timeout=2.0)
    w.stop()
    t.join(timeout=2.0)

    assert res is not None and res.ok
    assert res.classification["category"] == "Travel_Transit"
    assert res.duration_ms >= 0
    assert w.stats.processed == 1
