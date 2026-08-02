"""Regression guard for the worker's classifier ordering.

A real run produced two split-brain outcomes because the worker ran the LLM
ahead of the high-confidence filename rules:

* sector-ETF exports split across categories — ``xlc``/``xle`` to
  AstroQuant_Sidereal but ``xlb``/``xlf``/``xlp``/``xlu`` to Financial_Taxes,
  because the model reads their OHLC header as generic finance;
* ``PATEL_CAN_PAY_SLIP_*`` decided by the model rather than by the project's
  own pay-slip rule.

The worker now runs HC-rules → AI → keyword-rules. These tests pin both
halves of that contract: HC wins over a confident AI, and a *non*-HC file
still reaches the AI rather than being short-circuited by keyword rules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from classifier.classifiers import (
    HighConfidenceRulesClassifier,
    LocalAIPipelineClassifier,
    RulesClassifier,
)
from classifier.pipeline import ClassificationPipeline
from classifier.rules import RulesEngine
from classifier.types import Classification, FileItem

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "config" / "categories.yaml")


class _LoudAI:
    """Stands in for the LLM, always answering confidently and wrongly."""
    name = "Local AI"

    def __init__(self, category="Financial_Taxes"):
        self.category = category
        self.calls: list[str] = []

    def classify(self, file: FileItem):
        self.calls.append(file.name)
        return Classification(
            category=self.category, confidence=95,
            method="Local AI", reason="stub: reads OHLC header as finance",
        )


def _worker_pipeline(ai) -> ClassificationPipeline:
    """Mirror of _build_worker_classifier's ordering for ai-small/ai-large."""
    rules = RulesEngine(CONFIG)
    return ClassificationPipeline([
        HighConfidenceRulesClassifier(rules),
        ai,
        RulesClassifier(rules),
    ])


@pytest.mark.parametrize("name", [
    "xlb.csv", "xlc.csv", "xle.csv", "xlf.csv",
    "xlp.csv", "xlu.csv", "xlv.csv", "xlre.csv", "nfty.csv",
])
def test_ticker_exports_beat_a_confident_ai(tmp_path, name):
    ai = _LoudAI("Financial_Taxes")
    p = tmp_path / name
    p.touch()
    result = _worker_pipeline(ai).classify(FileItem(path=p))
    assert result.category == "AstroQuant_Sidereal", f"{name} -> {result.category}"
    assert ai.calls == [], "HC rules must short-circuit before the LLM is queried"


@pytest.mark.parametrize("name,expected", [
    ("PATEL_CAN_PAY_SLIP_JUN_2025.pdf", "Canadian_PR_Docs"),
    ("PATEL_CAN_FIRST_PAY_SLIP_AUG_2022.pdf", "Canadian_PR_Docs"),
    ("Payslip_May_2026.pdf", "Financial_Taxes"),
    ("Happy_imm5476e_Signed.pdf", "Canadian_PR_Docs"),
    ("Patel_Happy_2022-2023_T4.pdf", "Canadian_PR_Docs"),
])
def test_definitional_markers_beat_the_ai(tmp_path, name, expected):
    ai = _LoudAI("Resumes_Career_Tech")
    p = tmp_path / name
    p.touch()
    result = _worker_pipeline(ai).classify(FileItem(path=p))
    assert result.category == expected, f"{name} -> {result.category}"


@pytest.mark.parametrize("name", [
    "Happy Patel Resume.pdf",
    "IndiGo Itinerary - Y4947Y.pdf",
    "JioHome-May-2026.pdf",
    "CPIKOR-CORP-Customer-metadata.xml",
    "Untitled document.pdf",
])
def test_ordinary_documents_still_reach_the_ai(tmp_path, name):
    """HC must stay narrow — the whole point of AI-first was that the LLM
    should see the bulk of the corpus, not be starved by filename rules."""
    ai = _LoudAI("Financial_Taxes")
    p = tmp_path / name
    p.touch()
    result = _worker_pipeline(ai).classify(FileItem(path=p))
    assert ai.calls == [name], f"{name} should have been sent to the LLM"
    assert result.method == "Local AI"


def test_keyword_rules_remain_behind_the_ai(tmp_path):
    """A file that keyword rules *could* match must still go to the AI first."""
    ai = _LoudAI("Medical_Health")
    p = tmp_path / "Mobile Receipt.pdf"  # matches the 'receipt' keyword
    p.touch()
    result = _worker_pipeline(ai).classify(FileItem(path=p))
    assert result.category == "Medical_Health"
    assert ai.calls == ["Mobile Receipt.pdf"]
