"""Pluggable queue backends.

A backend exposes two logical streams per topology:

* per-route job streams the workers pull from
* a single result stream the orchestrator collects from

The in-memory implementation is for single-process runs and tests. The Redis
Streams implementation reuses the same shape across processes / hosts via
consumer groups (one consumer group per route, one stream for results).
"""
from __future__ import annotations

import json
import logging
import queue as stdlib_queue
import threading
import time
from typing import Optional, Protocol

from inference.types import Job, JobResult

log = logging.getLogger(__name__)

DEFAULT_RESULT_STREAM = "smartsort:results"
DEFAULT_JOB_STREAM_PREFIX = "smartsort:jobs:"
DEFAULT_GROUP = "smartsort-workers"

__all__ = [
    "DEFAULT_RESULT_STREAM",
    "DEFAULT_JOB_STREAM_PREFIX",
    "DEFAULT_GROUP",
    "QueueBackend",
    "InMemoryQueueBackend",
    "RedisStreamBackend",
    "build_backend",
]


class QueueBackend(Protocol):
    def enqueue(self, job: Job) -> None: ...
    def dequeue(self, routes: list[str], timeout: float = 1.0) -> Optional[tuple[Job, str]]:
        """Return (job, ack_token) or None on timeout."""
    def ack(self, route: str, ack_token: str) -> None: ...
    def publish_result(self, result: JobResult) -> None: ...
    def consume_result(self, timeout: float = 1.0) -> Optional[JobResult]: ...
    def close(self) -> None: ...


# ----------------------------------------------------------------- in-memory


class InMemoryQueueBackend:
    """Thread-safe queues. Workers and orchestrator share one instance."""

    def __init__(self) -> None:
        self._routes: dict[str, stdlib_queue.Queue] = {}
        self._results: stdlib_queue.Queue = stdlib_queue.Queue()
        self._lock = threading.Lock()

    def _route_queue(self, route: str) -> stdlib_queue.Queue:
        with self._lock:
            q = self._routes.get(route)
            if q is None:
                q = stdlib_queue.Queue()
                self._routes[route] = q
            return q

    def enqueue(self, job: Job) -> None:
        self._route_queue(job.route).put(job)

    def dequeue(self, routes: list[str], timeout: float = 1.0) -> Optional[tuple[Job, str]]:
        # Round-robin poll across routes within the timeout window so a single
        # worker can subscribe to multiple queues without starving any one.
        deadline = time.time() + timeout
        per_poll = max(0.01, timeout / max(len(routes), 1))
        while time.time() < deadline:
            for route in routes:
                q = self._route_queue(route)
                try:
                    job = q.get(timeout=per_poll)
                except stdlib_queue.Empty:
                    continue
                # ack token unused in-memory; queue.get() already removed it.
                return job, job.id
        return None

    def ack(self, route: str, ack_token: str) -> None:
        # No-op: in-memory dequeue already removes the job.
        return None

    def publish_result(self, result: JobResult) -> None:
        self._results.put(result)

    def consume_result(self, timeout: float = 1.0) -> Optional[JobResult]:
        try:
            return self._results.get(timeout=timeout)
        except stdlib_queue.Empty:
            return None

    def close(self) -> None:
        return None


# --------------------------------------------------------------------- Redis


class RedisStreamBackend:
    """Redis Streams backend.

    One stream per route (``smartsort:jobs:<route>``) consumed via consumer
    groups, plus a single result stream. ``redis`` is imported lazily so the
    package keeps working without the dependency installed for in-memory runs.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        consumer_name: Optional[str] = None,
        group: str = DEFAULT_GROUP,
        stream_prefix: str = DEFAULT_JOB_STREAM_PREFIX,
        result_stream: str = DEFAULT_RESULT_STREAM,
        block_ms: int = 1000,
    ):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "redis backend requires the 'redis' package: pip install redis"
            ) from exc

        # Long-running workers sit idle on XREADGROUP for hours. Without
        # TCP keepalive a NAT / firewall in between can silently drop the
        # connection, leaving the worker blocked forever on a dead socket
        # while Redis still happily lists it as a consumer. Enable OS-level
        # keepalive and let redis-py's health-check probe surface dead
        # connections so the next call reconnects automatically.
        self._redis = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        self._consumer = consumer_name or f"c-{int(time.time()*1000)}"
        self._group = group
        self._stream_prefix = stream_prefix
        self._result_stream = result_stream
        self._block_ms = block_ms
        self._groups_ensured: set[str] = set()
        self._result_last_id = "$"

    def _stream(self, route: str) -> str:
        return f"{self._stream_prefix}{route}"

    def _ensure_group(self, stream: str) -> None:
        if stream in self._groups_ensured:
            return
        try:
            self._redis.xgroup_create(stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            # BUSYGROUP means the group already exists, which is fine.
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups_ensured.add(stream)

    def enqueue(self, job: Job) -> None:
        stream = self._stream(job.route)
        self._ensure_group(stream)
        self._redis.xadd(stream, {"job": json.dumps(job.to_dict())})

    def dequeue(self, routes: list[str], timeout: float = 1.0) -> Optional[tuple[Job, str]]:
        streams = {self._stream(r): ">" for r in routes}
        for s in streams:
            self._ensure_group(s)
        block_ms = int(timeout * 1000)
        resp = self._redis.xreadgroup(
            self._group, self._consumer, streams, count=1, block=block_ms
        )
        if not resp:
            return None
        stream, entries = resp[0]
        if not entries:
            return None
        entry_id, payload = entries[0]
        job = Job.from_dict(json.loads(payload["job"]))
        # ack_token encodes which stream the entry came from so ack() can route.
        return job, f"{stream}|{entry_id}"

    def ack(self, route: str, ack_token: str) -> None:
        stream, _, entry_id = ack_token.partition("|")
        if not entry_id:
            stream = self._stream(route)
            entry_id = ack_token
        self._redis.xack(stream, self._group, entry_id)
        # Trim acked entry so the stream doesn't grow unboundedly.
        try:
            self._redis.xdel(stream, entry_id)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def publish_result(self, result: JobResult) -> None:
        self._redis.xadd(self._result_stream, {"result": json.dumps(result.to_dict())})

    def consume_result(self, timeout: float = 1.0) -> Optional[JobResult]:
        block_ms = int(timeout * 1000)
        resp = self._redis.xread({self._result_stream: self._result_last_id}, count=1, block=block_ms)
        if not resp:
            return None
        _, entries = resp[0]
        if not entries:
            return None
        entry_id, payload = entries[0]
        self._result_last_id = entry_id
        return JobResult.from_dict(json.loads(payload["result"]))

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------- factory


def build_backend(kind: str = "memory", **kwargs) -> QueueBackend:
    kind = kind.lower()
    if kind in ("memory", "in-memory", "mem"):
        return InMemoryQueueBackend()
    if kind == "redis":
        return RedisStreamBackend(**kwargs)
    raise ValueError(f"Unknown queue backend: {kind!r}")
