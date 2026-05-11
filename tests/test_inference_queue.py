"""In-memory queue backend round-trips."""
from inference.queue import InMemoryQueueBackend
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
