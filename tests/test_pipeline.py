"""Tests for the classification pipeline runner.

We use fake classifiers so the test is independent of YAML, AI, or filesystem.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from classifier.pipeline import ClassificationPipeline
from classifier.types import Classification, FileItem, UNKNOWN_CATEGORY


class _Fake:
    def __init__(self, name, result):
        self.name = name
        self._result = result
        self.calls = 0

    def classify(self, file):
        self.calls += 1
        return self._result


class _Boom:
    name = "boom"

    def classify(self, file):
        raise RuntimeError("classifier exploded")


@pytest.fixture
def file_item(tmp_path):
    p = tmp_path / "x.pdf"
    p.touch()
    return FileItem(path=p)


def _ok(cat="Canadian_PR_Docs", conf=95, method="m"):
    return Classification(category=cat, confidence=conf, method=method, reason="r")


def test_first_known_classifier_wins(file_item):
    a = _Fake("a", _ok("Canadian_PR_Docs"))
    b = _Fake("b", _ok("Financial_Taxes"))
    pipe = ClassificationPipeline([a, b])
    result = pipe.classify(file_item)
    assert result.category == "Canadian_PR_Docs"
    assert b.calls == 0  # short-circuited


def test_none_lets_next_classifier_run(file_item):
    a = _Fake("a", None)
    b = _Fake("b", _ok("Financial_Taxes"))
    pipe = ClassificationPipeline([a, b])
    result = pipe.classify(file_item)
    assert result.category == "Financial_Taxes"
    assert a.calls == 1 and b.calls == 1


def test_unknown_category_treated_as_no_answer(file_item):
    a = _Fake("a", Classification.unknown())
    b = _Fake("b", _ok("Financial_Taxes"))
    pipe = ClassificationPipeline([a, b])
    assert pipe.classify(file_item).category == "Financial_Taxes"


def test_all_classifiers_silent_yields_unknown(file_item):
    pipe = ClassificationPipeline([_Fake("a", None), _Fake("b", None)])
    assert pipe.classify(file_item).category == UNKNOWN_CATEGORY


def test_classifier_exception_does_not_break_pipeline(file_item):
    pipe = ClassificationPipeline([_Boom(), _Fake("ok", _ok("Financial_Taxes"))])
    result = pipe.classify(file_item)
    assert result.category == "Financial_Taxes"


def test_pipeline_with_no_classifiers_yields_unknown(file_item):
    pipe = ClassificationPipeline([])
    assert pipe.classify(file_item).category == UNKNOWN_CATEGORY
