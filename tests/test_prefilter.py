"""Tests for the dispatcher-side Prefilter.

Regression guard for a real run in which 54 files came back
``Unknown_Unsorted`` with reason "no AI / OCR worker available for this file
type" — including ``.DS_Store``, ``.localized`` and ``Ollama.dmg``, all of
which the filename rules classify correctly from the extension alone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from classifier.rules import RulesEngine
from classifier.types import FileItem
from inference.prefilter import Prefilter

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "config" / "categories.yaml")


@pytest.fixture(scope="module")
def prefilter():
    return Prefilter(RulesEngine(CONFIG))


@pytest.mark.parametrize("name,expected", [
    (".DS_Store", "System_Junk"),
    (".localized", "System_Junk"),
    ("Ollama.dmg", "Archives_and_Apps"),
    ("Docker.dmg", "Archives_and_Apps"),
    ("Claude.dmg", "Archives_and_Apps"),
    ("job_hunter.zip", "Archives_and_Apps"),
    ("Untitled design.zip", "Archives_and_Apps"),
    # Images: no OCR worker, so the prefilter is the only thing that can
    # classify these. Filenames carry plenty of signal.
    ("Screenshot 2026-06-08 at 15.12.07.png", "Media_Images"),
    ("screencapture-guidewire-hub-eu-admin-okta.png", "Media_Images"),
    ("image (5).png", "Media_Images"),
    ("PXL_20260401_184833356.jpg", "Media_Images"),
])
def test_prefilter_classifies_files_no_worker_can_read(prefilter, tmp_path, name, expected):
    p = tmp_path / name
    p.touch()
    result = prefilter.classify(FileItem(path=p))
    assert result.category == expected, f"{name} -> {result.category} ({result.reason})"
    assert result.is_known


def test_prefilter_returns_unknown_with_clear_reason(prefilter, tmp_path):
    p = tmp_path / "gw_totally_opaque_blob.bin"
    p.touch()
    result = prefilter.classify(FileItem(path=p))
    assert not result.is_known
    assert result.method == "Prefilter"
    assert "no rule matched" in result.reason


def test_prefilter_prefers_high_confidence_rules(prefilter, tmp_path):
    """An HC marker must win even for a file no worker could read."""
    p = tmp_path / "Happy_imm5476e_Signed.zip"
    p.touch()
    result = prefilter.classify(FileItem(path=p))
    assert result.category == "Canadian_PR_Docs"
    assert result.confidence == 100
