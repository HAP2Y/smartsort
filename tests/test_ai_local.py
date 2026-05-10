"""Tests for the Ollama-backed local AI classifier.

Network is fully mocked. We test:

* ``parse_response`` against happy-path JSON, code-fenced JSON, invalid JSON,
  empty replies, out-of-bounds confidence, and categories outside the allowed
  set.
* ``OllamaClient.health`` against connection refused, server-not-OK, and a
  missing model.
* ``LocalAIClassifier.classify`` end-to-end with a mocked session, including
  HTTP 5xx and timeout paths.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from classifier.ai_local import (
    LocalAIClassifier,
    OllamaClient,
    build_prompt,
    parse_response,
)
from classifier.types import UNKNOWN_CATEGORY


CATEGORIES = ["Canadian_PR_Docs", "Financial_Taxes", "Resumes_Career_Tech", UNKNOWN_CATEGORY]


# ------------------------------------------------------------------ build_prompt


def test_build_prompt_includes_filename_snippet_and_categories():
    prompt = build_prompt("foo.pdf", "snippet text", CATEGORIES)
    assert "foo.pdf" in prompt
    assert "snippet text" in prompt
    for cat in CATEGORIES:
        assert cat in prompt


# ------------------------------------------------------------------ parse_response


def test_parse_response_clean_json():
    raw = '{"category": "Canadian_PR_Docs", "confidence": 95, "reason": "EVL"}'
    result = parse_response(raw, CATEGORIES)
    assert result.category == "Canadian_PR_Docs"
    assert result.confidence == 95
    assert "EVL" in result.reason
    assert result.method == "Local AI"


def test_parse_response_strips_json_code_fence():
    raw = '```json\n{"category":"Financial_Taxes","confidence":88,"reason":"bank statement"}\n```'
    result = parse_response(raw, CATEGORIES)
    assert result.category == "Financial_Taxes"
    assert result.confidence == 88


def test_parse_response_strips_plain_code_fence():
    raw = '```\n{"category":"Resumes_Career_Tech","confidence":90,"reason":"CV"}\n```'
    result = parse_response(raw, CATEGORIES)
    assert result.category == "Resumes_Career_Tech"


def test_parse_response_empty_string_is_unknown():
    result = parse_response("", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "empty" in result.reason.lower()


def test_parse_response_invalid_json_is_unknown():
    result = parse_response("not json at all", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "invalid json" in result.reason.lower()


def test_parse_response_non_object_json_is_unknown():
    result = parse_response("[1,2,3]", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY


def test_parse_response_disallowed_category_is_unknown():
    raw = '{"category":"made_up","confidence":99,"reason":"x"}'
    result = parse_response(raw, CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "unknown category" in result.reason.lower()


def test_parse_response_clamps_confidence():
    raw = '{"category":"Financial_Taxes","confidence":250,"reason":"x"}'
    result = parse_response(raw, CATEGORIES)
    assert result.confidence == 100

    raw = '{"category":"Financial_Taxes","confidence":-50,"reason":"x"}'
    result = parse_response(raw, CATEGORIES)
    assert result.confidence == 0


def test_parse_response_handles_non_int_confidence():
    raw = '{"category":"Financial_Taxes","confidence":"high","reason":"x"}'
    result = parse_response(raw, CATEGORIES)
    assert result.confidence == 0
    assert result.category == "Financial_Taxes"


# --------------------------------------------------------------- OllamaClient.health


def _resp(status_code: int, json_body=None):
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=(json_body if json_body is not None else {}))
    r.text = ""
    return r


def test_health_ok_when_server_up_and_model_present():
    session = MagicMock()
    session.get.side_effect = [
        _resp(200),  # base_url
        _resp(200, {"models": [{"name": "qwen2.5:14b"}, {"name": "llama3"}]}),
    ]
    client = OllamaClient(session=session)
    status = client.health("qwen2.5:14b")
    assert status.ok is True
    assert "qwen2.5:14b" in status.message


def test_health_reports_missing_model():
    session = MagicMock()
    session.get.side_effect = [
        _resp(200),
        _resp(200, {"models": [{"name": "llama3"}]}),
    ]
    client = OllamaClient(session=session)
    status = client.health("qwen2.5:14b")
    assert status.ok is False
    assert "ollama pull qwen2.5:14b" in status.message


def test_health_reports_connection_refused():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError()
    client = OllamaClient(session=session)
    status = client.health("any")
    assert status.ok is False
    assert "connection refused" in status.message.lower()


def test_health_reports_server_error():
    session = MagicMock()
    session.get.return_value = _resp(503)
    client = OllamaClient(session=session)
    status = client.health("any")
    assert status.ok is False
    assert "503" in status.message


# ----------------------------------------------------- LocalAIClassifier.classify


def _ai_with_session(session):
    return LocalAIClassifier(model="qwen2.5:14b", client=OllamaClient(session=session))


def test_classify_happy_path():
    session = MagicMock()
    session.post.return_value = _resp(
        200,
        {"response": '{"category":"Canadian_PR_Docs","confidence":97,"reason":"IMM form"}'},
    )
    ai = _ai_with_session(session)
    result = ai.classify("Happy_imm5476e.pdf", "IMM 5476 Use of a Representative", CATEGORIES)
    assert result.category == "Canadian_PR_Docs"
    assert result.confidence == 97
    assert result.method == "Local AI"


def test_classify_handles_http_500():
    session = MagicMock()
    err = _resp(500)
    err.text = "internal error"
    session.post.return_value = err
    ai = _ai_with_session(session)
    result = ai.classify("foo.pdf", "x", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "500" in result.reason


def test_classify_handles_timeout():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ReadTimeout()
    ai = _ai_with_session(session)
    result = ai.classify("foo.pdf", "x", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "timed out" in result.reason.lower()


def test_classify_handles_connection_refused():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError()
    ai = _ai_with_session(session)
    result = ai.classify("foo.pdf", "x", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
    assert "connection refused" in result.reason.lower()


def test_classify_handles_garbled_response_envelope():
    session = MagicMock()
    bad = MagicMock()
    bad.status_code = 200
    bad.json.side_effect = ValueError("not json")
    session.post.return_value = bad
    ai = _ai_with_session(session)
    result = ai.classify("foo.pdf", "x", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY


@pytest.mark.parametrize("body", [
    '   ',
    'not json',
    '{"category":"Outside","confidence":95,"reason":"r"}',
])
def test_classify_unknown_when_model_misbehaves(body):
    session = MagicMock()
    session.post.return_value = _resp(200, {"response": body})
    ai = _ai_with_session(session)
    result = ai.classify("foo.pdf", "x", CATEGORIES)
    assert result.category == UNKNOWN_CATEGORY
