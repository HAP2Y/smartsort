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
    # Emit a heartbeat log every N seconds even when idle. Without this a
    # silent worker is indistinguishable from a hung one — operators only
    # learn it's alive when a job comes through, which can be hours apart.
    heartbeat_seconds: float = 30.0
    stats: WorkerStats = field(default_factory=WorkerStats)
    _stop: threading.Event = field(default_factory=threading.Event)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("worker %s starting on routes=%s", self.name, self.routes)
        last_heartbeat = time.time()
        while not self._stop.is_set():
            try:
                item = self.backend.dequeue(self.routes, timeout=self.poll_timeout)
            except Exception:
                # A dead Redis socket or transient broker error must not
                # kill the worker — log and back off briefly so we don't
                # tight-loop hammering a broken broker.
                log.exception("worker %s: dequeue failed; retrying in 2 s", self.name)
                self._stop.wait(2.0)
                continue

            now = time.time()
            if now - last_heartbeat >= self.heartbeat_seconds:
                log.info(
                    "worker %s heartbeat: routes=%s processed=%d failed=%d",
                    self.name, self.routes, self.stats.processed, self.stats.failed,
                )
                last_heartbeat = now

            if item is None:
                continue
            job, ack_token = item
            self._handle(job, ack_token)
            last_heartbeat = time.time()  # job log already proves we're alive

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
        file_name = Path(job.file_path).name
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

        # One INFO line per job so `docker compose logs -f` is a usable
        # progress feed without forcing -vv. Includes the method that
        # decided the file (Rules / Local AI / etc) so the operator can
        # see whether the LLM is actually being exercised or whether the
        # cheap rules path is matching everything.
        if error:
            log.warning("[%s] %s -> ERROR (%s) in %.0fms",
                        self.name, file_name, error, duration_ms)
        elif classification:
            log.info("[%s] %s -> %s (%s, %d%%) in %.0fms",
                     self.name, file_name,
                     classification.get("category", "?"),
                     classification.get("method", "?"),
                     int(classification.get("confidence", 0)),
                     duration_ms)
        else:
            log.info("[%s] %s -> no result in %.0fms", self.name, file_name, duration_ms)

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
