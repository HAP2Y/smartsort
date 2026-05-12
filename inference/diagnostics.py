"""Pre-flight checks + live progress + post-mortem for distributed runs.

The dispatcher walks the file set and pushes jobs into Redis. Without
visibility, every operator-facing failure mode looks the same: jobs went
in, time passed, results didn't come out. The helpers here make those
modes diagnosable in seconds rather than minutes.

Three concerns, three helpers:

* `preflight()` — before submitting, verify redis is reachable, the AI
  routes have at least one consumer subscribed, and (if probable) Ollama
  is healthy. Print a status panel and refuse to submit if anything
  blocks the run.

* `ProgressReporter` — feed it the orchestrator's results as they come in
  and it prints `[N/M] worker -> file` lines plus a periodic per-route
  summary, so you can watch the queue drain in real time.

* `postmortem()` — after collect returns, if anything failed, summarise
  per-route (how many timed out, how many errored) and dump the last few
  log lines from each worker container so the operator doesn't have to
  go hunting in `docker compose logs`.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from inference.queue import (
    DEFAULT_GROUP,
    DEFAULT_JOB_STREAM_PREFIX,
    QueueBackend,
)
from inference.types import JobResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- preflight


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str

    @property
    def status(self) -> str:
        return "✓" if self.ok else "✗"


@dataclass
class PreflightReport:
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def blockers(self) -> list[PreflightCheck]:
        return [c for c in self.checks if not c.ok]


def preflight_redis(redis_url: str) -> PreflightCheck:
    try:
        import redis  # type: ignore
        r = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        return PreflightCheck("redis", True, f"reachable at {redis_url}")
    except Exception as exc:
        return PreflightCheck("redis", False, f"unreachable at {redis_url}: {exc!s}")


def preflight_workers(redis_url: str, expected_routes: list[str]) -> list[PreflightCheck]:
    """One check per expected route — does any consumer in the worker
    group exist on that stream?"""
    out: list[PreflightCheck] = []
    try:
        import redis  # type: ignore
        r = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
    except Exception as exc:
        return [PreflightCheck(f"workers/{route}", False, f"redis client init failed: {exc!s}")
                for route in expected_routes]

    for route in expected_routes:
        stream = f"{DEFAULT_JOB_STREAM_PREFIX}{route}"
        try:
            consumers = r.xinfo_consumers(stream, DEFAULT_GROUP)
            count = len(consumers)
            if count == 0:
                out.append(PreflightCheck(
                    f"workers/{route}", False,
                    f"no consumers in group {DEFAULT_GROUP!r} (start one with "
                    f"`smartsort serve-worker --routes {route} --backend redis`)",
                ))
            else:
                names = ", ".join(c.get("name", "?") for c in consumers)
                out.append(PreflightCheck(
                    f"workers/{route}", True, f"{count} consumer(s): {names}",
                ))
        except Exception as exc:
            # Stream/group may not exist yet on a cold cluster — that's
            # only an actual problem if the route has expected jobs.
            out.append(PreflightCheck(
                f"workers/{route}", False,
                f"consumer-group not initialised yet ({type(exc).__name__})",
            ))
    return out


def preflight_ollama(url: str, timeout: float = 3.0) -> PreflightCheck:
    """Best-effort host-side probe. Workers may still hit Ollama via a
    different URL (e.g. host.docker.internal from inside a container);
    if `url` resolves to localhost we just probe localhost."""
    try:
        import requests  # already a dependency
        r = requests.get(url.rstrip("/"), timeout=timeout)
        if r.status_code == 200:
            return PreflightCheck("ollama", True, f"reachable at {url}")
        return PreflightCheck("ollama", False, f"HTTP {r.status_code} at {url}")
    except Exception as exc:
        return PreflightCheck(
            "ollama", False,
            f"unreachable at {url}: {type(exc).__name__} ({exc!s})",
        )


# -------------------------------------------------------------- progress


@dataclass
class ProgressReporter:
    """Updates a per-route counter as results stream in, and emits a
    one-line summary every `tick_seconds` so the operator sees movement.

    Plug into Orchestrator.collect via the `on_result` callback.
    """
    expected_total: int
    expected_by_route: dict[str, int]
    tick_seconds: float = 5.0
    completed: int = 0
    completed_by_route: dict[str, int] = field(default_factory=dict)
    failed: int = 0
    _started: float = field(default_factory=time.time)
    _last_tick: float = field(default_factory=time.time)
    _printer: callable = print

    def on_result(self, result: JobResult, _classification) -> None:
        self.completed += 1
        self.completed_by_route[result.route] = self.completed_by_route.get(result.route, 0) + 1
        if not result.ok:
            self.failed += 1
        if time.time() - self._last_tick >= self.tick_seconds:
            self._tick()
            self._last_tick = time.time()

    def _tick(self) -> None:
        elapsed = time.time() - self._started
        per_route = " | ".join(
            f"{r} {self.completed_by_route.get(r, 0)}/{self.expected_by_route.get(r, 0)}"
            for r in sorted(self.expected_by_route)
        )
        self._printer(
            f"[{elapsed:5.0f}s] {self.completed}/{self.expected_total} done | {per_route}"
        )

    def final(self) -> None:
        """Emit a final tick so the operator always sees one summary line
        even on workloads that finish before the first periodic tick."""
        self._tick()


# ------------------------------------------------------------- post-mortem


def queue_depths(backend: QueueBackend, routes: list[str]) -> dict[str, int]:
    """Return `{route: outstanding_entries}` — i.e. jobs still queued or
    in-flight (delivered to a consumer but not acked)."""
    out: dict[str, int] = {}
    redis_client = getattr(backend, "_redis", None)
    if redis_client is None:
        return out
    for route in routes:
        stream = f"{DEFAULT_JOB_STREAM_PREFIX}{route}"
        try:
            out[route] = redis_client.xlen(stream)
        except Exception:
            out[route] = -1
    return out


def tail_compose_logs(services: list[str], lines: int = 10) -> dict[str, str]:
    """Best-effort: return the last `lines` of each compose service's
    log. Empty dict if docker-compose isn't available."""
    out: dict[str, str] = {}
    base = _compose_cmd()
    if base is None:
        return out
    for svc in services:
        try:
            r = subprocess.run(
                base + ["logs", "--no-color", "--tail", str(lines), svc],
                capture_output=True, text=True, timeout=10,
            )
            out[svc] = r.stdout.strip() or r.stderr.strip()
        except Exception as exc:
            out[svc] = f"(log fetch failed: {exc!s})"
    return out


def _compose_cmd() -> list[str] | None:
    if shutil.which("docker"):
        try:
            r = subprocess.run(
                ["docker", "compose", "version"], capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return ["docker", "compose"]
        except Exception:
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None
