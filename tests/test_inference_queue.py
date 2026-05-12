"""In-memory queue backend round-trips and factory."""
import threading

import pytest

from inference.queue import InMemoryQueueBackend, build_backend
from inference.types import Job, JobResult


def test_enqueue_dequeue_round_trip():
    qb = InMemoryQueueBackend()
    job = Job(file_path="/tmp/a.pdf", route="rules")
    qb.enqueue(job)

    got = qb.dequeue(["rules"], timeout=1.0)
    assert got is not None
    dequeued, token = got
    assert dequeued.id == job.id
    assert dequeued.file_path == "/tmp/a.pdf"
    assert token == job.id


def test_dequeue_times_out_when_empty():
    qb = InMemoryQueueBackend()
    assert qb.dequeue(["rules"], timeout=0.1) is None


def test_dequeue_polls_multiple_routes():
    qb = InMemoryQueueBackend()
    qb.enqueue(Job(file_path="/tmp/b.pdf", route="ai-small"))
    got = qb.dequeue(["rules", "ai-small"], timeout=1.0)
    assert got is not None
    assert got[0].route == "ai-small"


def test_results_round_trip():
    qb = InMemoryQueueBackend()
    res = JobResult(
        job_id="j1",
        file_path="/tmp/a.pdf",
        route="rules",
        worker_id="w1",
        duration_ms=12.3,
        classification={"category": "Foo", "confidence": 90, "method": "Rules", "reason": "x"},
    )
    qb.publish_result(res)
    got = qb.consume_result(timeout=1.0)
    assert got is not None
    assert got.job_id == "j1"
    assert got.ok


def test_consume_result_times_out_when_empty():
    qb = InMemoryQueueBackend()
    assert qb.consume_result(timeout=0.1) is None


def test_ack_and_close_are_noops_in_memory():
    """In-memory backend has nothing to ack or release; calls must not raise."""
    qb = InMemoryQueueBackend()
    qb.enqueue(Job(file_path="/tmp/a.pdf", route="rules"))
    job, token = qb.dequeue(["rules"], timeout=0.5)
    qb.ack("rules", token)
    qb.close()


def test_concurrent_enqueue_from_many_threads_is_safe():
    qb = InMemoryQueueBackend()
    threads_count = 8
    per_thread = 25
    barrier = threading.Barrier(threads_count)

    def producer():
        barrier.wait()
        for i in range(per_thread):
            qb.enqueue(Job(file_path=f"/tmp/f{i}.pdf", route="rules"))

    threads = [threading.Thread(target=producer) for _ in range(threads_count)]
    for t in threads: t.start()
    for t in threads: t.join()

    drained = 0
    while qb.dequeue(["rules"], timeout=0.1) is not None:
        drained += 1
    assert drained == threads_count * per_thread


def test_build_backend_returns_in_memory_for_default_kind():
    assert isinstance(build_backend("memory"), InMemoryQueueBackend)
    assert isinstance(build_backend("in-memory"), InMemoryQueueBackend)
    assert isinstance(build_backend("MEM"), InMemoryQueueBackend)


def test_build_backend_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown queue backend"):
        build_backend("kafka")
