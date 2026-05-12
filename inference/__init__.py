"""Distributed inference primitives for SmartSort.

The pipeline that ships with SmartSort runs every classifier inline. This
package turns the same building blocks into a producer/worker system: a
``Router`` decides which queue a file goes to, ``Worker`` processes consume
jobs from those queues and publish ``JobResult`` envelopes, and an
``Orchestrator`` ties the two ends together.

The default ``QueueBackend`` is an in-memory implementation suitable for
single-process tests and local runs. ``RedisStreamBackend`` is a drop-in
replacement that scales the same topology across processes and hosts.
"""
from inference.types import Job, JobResult
from inference.queue import (
    InMemoryQueueBackend,
    QueueBackend,
    build_backend,
)
from inference.router import Router, RouteRule
from inference.worker import Worker
from inference.orchestrator import Orchestrator

__all__ = [
    "Job",
    "JobResult",
    "QueueBackend",
    "InMemoryQueueBackend",
    "build_backend",
    "Router",
    "RouteRule",
    "Worker",
    "Orchestrator",
]
