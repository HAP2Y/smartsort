"""Wire-format stability tests for Job / JobResult.

These types travel between processes (and hosts, via Redis). The dict
serialisation has to round-trip cleanly and the defaults have to make sense
without callers having to remember them.
"""
from __future__ import annotations

import time

from inference.types import Job, JobResult


def test_job_autogenerates_id_and_timestamp():
    before = time.time()
    job = Job(file_path="/tmp/a.pdf", route="rules")
    after = time.time()

    assert job.id and len(job.id) == 32      # uuid4 hex
    assert before <= job.enqueued_at <= after
    assert job.attempts == 0
    assert job.max_attempts == 3
    assert job.payload == {}


def test_job_round_trips_through_dict():
    original = Job(
        file_path="/tmp/b.pdf",
        route="ai-small",
        payload={"categories": ["A", "B"], "threshold": 80},
        attempts=2,
        max_attempts=5,
    )
    restored = Job.from_dict(original.to_dict())

    assert restored.id == original.id
    assert restored.file_path == original.file_path
    assert restored.route == original.route
    assert restored.payload == original.payload
    assert restored.attempts == original.attempts
    assert restored.max_attempts == original.max_attempts
    assert restored.enqueued_at == original.enqueued_at


def test_job_from_dict_supplies_missing_optional_fields():
    minimal = {"id": "abc", "file_path": "/tmp/c.pdf", "route": "rules"}
    job = Job.from_dict(minimal)

    assert job.id == "abc"
    assert job.payload == {}
    assert job.attempts == 0
    assert job.max_attempts == 3


def test_jobresult_ok_when_classification_present_and_no_error():
    res = JobResult(
        job_id="j1", file_path="/tmp/a.pdf", route="rules",
        worker_id="w1", duration_ms=5.0,
        classification={"category": "Foo", "confidence": 90, "method": "Rules", "reason": "x"},
    )
    assert res.ok is True


def test_jobresult_not_ok_when_error_set():
    res = JobResult(
        job_id="j1", file_path="/tmp/a.pdf", route="rules",
        worker_id="w1", duration_ms=5.0,
        error="boom",
    )
    assert res.ok is False


def test_jobresult_not_ok_when_classification_missing():
    res = JobResult(
        job_id="j1", file_path="/tmp/a.pdf", route="rules",
        worker_id="w1", duration_ms=5.0,
    )
    assert res.ok is False


def test_jobresult_round_trips_through_dict():
    original = JobResult(
        job_id="j-7", file_path="/tmp/d.pdf", route="ai-large",
        worker_id="w-2", duration_ms=123.4,
        classification={"category": "Travel_Transit", "confidence": 95, "method": "AI", "reason": "r"},
    )
    restored = JobResult.from_dict(original.to_dict())

    assert restored.job_id == original.job_id
    assert restored.file_path == original.file_path
    assert restored.route == original.route
    assert restored.worker_id == original.worker_id
    assert restored.duration_ms == original.duration_ms
    assert restored.classification == original.classification
    assert restored.error is None
    assert restored.ok
