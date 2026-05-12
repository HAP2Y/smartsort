"""Wire types for distributed inference.

Jobs and results travel across processes (and eventually hosts), so they are
plain dataclasses with explicit dict serialisation. Keeping the schema small
and stable makes it cheap to swap queue backends.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    """A unit of inference work targeted at one route (queue)."""
    file_path: str
    route: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    enqueued_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "route": self.route,
            "payload": self.payload,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            route=data["route"],
            payload=data.get("payload", {}) or {},
            attempts=int(data.get("attempts", 0)),
            max_attempts=int(data.get("max_attempts", 3)),
            enqueued_at=float(data.get("enqueued_at", time.time())),
        )


@dataclass
class JobResult:
    """The outcome of running one Job through a worker."""
    job_id: str
    file_path: str
    route: str
    worker_id: str
    duration_ms: float
    classification: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.classification is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_path": self.file_path,
            "route": self.route,
            "worker_id": self.worker_id,
            "duration_ms": self.duration_ms,
            "classification": self.classification,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobResult":
        return cls(
            job_id=data["job_id"],
            file_path=data["file_path"],
            route=data["route"],
            worker_id=data["worker_id"],
            duration_ms=float(data["duration_ms"]),
            classification=data.get("classification"),
            error=data.get("error"),
        )
