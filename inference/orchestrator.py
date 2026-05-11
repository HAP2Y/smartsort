"""Submit jobs, collect results.

The orchestrator is the producer side: it walks a set of files, asks the
``Router`` which queue each belongs on, enqueues a ``Job``, and then drains
the result stream until every submitted job has reported back (or it times
out).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from classifier.types import Classification, FileItem
from inference.queue import QueueBackend
from inference.router import Router
from inference.types import Job, JobResult

log = logging.getLogger(__name__)


@dataclass
class OrchestratorStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    by_route: dict[str, int] = field(default_factory=dict)


@dataclass
class Orchestrator:
    backend: QueueBackend
    router: Router
    payload: dict = field(default_factory=dict)
    stats: OrchestratorStats = field(default_factory=OrchestratorStats)

    def submit(self, files: Iterable[Path]) -> dict[str, Job]:
        """Enqueue one Job per file. Returns ``{job_id: Job}`` for tracking."""
        pending: dict[str, Job] = {}
        for path in files:
            item = FileItem(path=path)
            route = self.router.route(item)
            job = Job(file_path=str(path), route=route, payload=dict(self.payload))
            self.backend.enqueue(job)
            pending[job.id] = job
            self.stats.submitted += 1
            self.stats.by_route[route] = self.stats.by_route.get(route, 0) + 1
        return pending

    def collect(
        self,
        pending: dict[str, Job],
        timeout: float = 300.0,
        poll: float = 1.0,
        on_result: Optional[callable] = None,
    ) -> dict[str, Classification]:
        """Block until every pending job has reported back or timeout expires.

        Returns ``{file_path: Classification}``. Files whose workers errored or
        whose jobs never finished get ``Classification.unknown()``.
        """
        out: dict[str, Classification] = {}
        deadline = time.time() + timeout
        remaining = dict(pending)

        while remaining and time.time() < deadline:
            result = self.backend.consume_result(timeout=poll)
            if result is None:
                continue
            if result.job_id not in remaining:
                # Stray result from a previous run on the same Redis stream.
                continue
            job = remaining.pop(result.job_id)
            classification = _result_to_classification(result)
            out[job.file_path] = classification
            if result.ok:
                self.stats.completed += 1
            else:
                self.stats.failed += 1
            if on_result is not None:
                try:
                    on_result(result, classification)
                except Exception:  # pragma: no cover - callback should not kill the loop
                    log.exception("orchestrator on_result callback raised")

        # Anything still pending is a timeout — fill with Unknown so the caller
        # can still produce a complete plan rather than crashing.
        for job in remaining.values():
            self.stats.failed += 1
            out[job.file_path] = Classification.unknown(
                reason="orchestrator timeout waiting for worker",
                method=f"Queue:{job.route}",
            )
        return out


def _result_to_classification(result: JobResult) -> Classification:
    if result.error:
        return Classification.unknown(
            reason=f"worker error: {result.error}",
            method=f"Queue:{result.route}",
        )
    data = result.classification or {}
    try:
        return Classification(
            category=data["category"],
            confidence=int(data["confidence"]),
            method=data.get("method", f"Queue:{result.route}"),
            reason=data.get("reason", ""),
        )
    except (KeyError, TypeError, ValueError):
        return Classification.unknown(
            reason="worker returned malformed classification",
            method=f"Queue:{result.route}",
        )
