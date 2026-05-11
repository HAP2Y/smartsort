"""Worker loop.

A ``Worker`` subscribes to one or more routes on a ``QueueBackend``, runs each
dequeued job through a ``Classifier`` (the same protocol the inline pipeline
uses), and publishes a ``JobResult``. The classifier is opaque to the worker
— that's what lets one binary host rules workers, small-model workers, and
large-model workers identically.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from classifier.pipeline import Classifier
from classifier.types import FileItem
from inference.queue import QueueBackend
from inference.types import Job, JobResult

log = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    processed: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0

    def record(self, duration_ms: float, ok: bool) -> None:
        self.total_duration_ms += duration_ms
        if ok:
            self.processed += 1
        else:
            self.failed += 1


@dataclass
class Worker:
    name: str
    routes: list[str]
    classifier: Classifier
    backend: QueueBackend
    poll_timeout: float = 1.0
    stats: WorkerStats = field(default_factory=WorkerStats)
    _stop: threading.Event = field(default_factory=threading.Event)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("worker %s starting on routes=%s", self.name, self.routes)
        while not self._stop.is_set():
            item = self.backend.dequeue(self.routes, timeout=self.poll_timeout)
            if item is None:
                continue
            job, ack_token = item
            self._handle(job, ack_token)
        log.info(
            "worker %s stopped (processed=%d failed=%d)",
            self.name, self.stats.processed, self.stats.failed,
        )

    def run_in_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name=f"worker-{self.name}", daemon=True)
        t.start()
        return t

    def _handle(self, job: Job, ack_token: str) -> None:
        started = time.perf_counter()
        error: Optional[str] = None
        classification = None
        try:
            file_item = FileItem(path=Path(job.file_path))
            result = self.classifier.classify(file_item)
            if result is not None:
                classification = result.to_dict()
        except Exception as exc:  # pragma: no cover - defensive; logged below
            log.exception("worker %s crashed on %s", self.name, job.file_path)
            error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        self.stats.record(duration_ms, ok=error is None)

        self.backend.publish_result(
            JobResult(
                job_id=job.id,
                file_path=job.file_path,
                route=job.route,
                worker_id=self.name,
                duration_ms=duration_ms,
                classification=classification,
                error=error,
            )
        )
        self.backend.ack(job.route, ack_token)
