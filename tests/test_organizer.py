"""Tests for ``movers.organizer.Organizer``.

Cover: dry-run no-op, move into category folders, name-collision suffixing,
the round-trip move + undo, and graceful handling of a missing undo log.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from movers.organizer import Organizer


CATEGORIES = ["Canadian_PR_Docs", "Financial_Taxes", "Resumes_Career_Tech"]


def _plan(filepath: str, category: str) -> dict:
    return {filepath: {"category": category, "confidence": 100, "method": "Rules", "reason": "r"}}


def test_dry_run_does_not_move_or_log(tmp_path):
    f = tmp_path / "Happy_imm5476e.pdf"
    f.write_text("x")
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    org.move_files(_plan(str(f), "Canadian_PR_Docs"), apply=False)
    assert f.exists()
    assert not (tmp_path / ".smartsort_undo.json").exists()


def test_apply_moves_into_category_folder_and_writes_log(tmp_path):
    f = tmp_path / "Happy_imm5476e.pdf"
    f.write_text("x")
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    org.move_files(_plan(str(f), "Canadian_PR_Docs"), apply=True)

    moved = tmp_path / "Canadian_PR_Docs" / "Happy_imm5476e.pdf"
    assert moved.exists()
    assert not f.exists()
    log = tmp_path / ".smartsort_undo.json"
    assert log.exists()
    entries = json.loads(log.read_text())
    assert len(entries) == 1
    assert entries[0]["category"] == "Canadian_PR_Docs"


def test_apply_skips_unknown_and_metadata(tmp_path):
    f1 = tmp_path / "a.pdf"
    f1.write_text("x")
    f2 = tmp_path / ".DS_Store"
    f2.write_text("x")
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    plan = {
        str(f1): {"category": "Unknown_Unsorted", "confidence": 0, "method": "n", "reason": "r"},
        str(f2): {"category": "Metadata_System", "confidence": 100, "method": "n", "reason": "r"},
    }
    org.move_files(plan, apply=True)
    assert f1.exists() and f2.exists()  # nothing moved
    assert not (tmp_path / ".smartsort_undo.json").exists()


def test_collision_suffixes_filename(tmp_path):
    f1 = tmp_path / "doc.pdf"
    f1.write_text("a")
    cat_dir = tmp_path / "Canadian_PR_Docs"
    cat_dir.mkdir()
    (cat_dir / "doc.pdf").write_text("pre-existing")

    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    org.move_files(_plan(str(f1), "Canadian_PR_Docs"), apply=True)

    assert (cat_dir / "doc.pdf").read_text() == "pre-existing"
    assert (cat_dir / "doc_1.pdf").exists()


def test_round_trip_apply_then_undo(tmp_path):
    f = tmp_path / "Happy_imm5476e.pdf"
    f.write_text("x")

    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    org.move_files(_plan(str(f), "Canadian_PR_Docs"), apply=True)
    moved = tmp_path / "Canadian_PR_Docs" / "Happy_imm5476e.pdf"
    assert moved.exists() and not f.exists()

    restored, missing, errors = Organizer(str(tmp_path), category_names=CATEGORIES).undo()
    assert restored == 1
    assert missing == 0
    assert errors == []
    assert f.exists() and not moved.exists()
    assert not (tmp_path / "Canadian_PR_Docs").exists()  # empty dir cleaned up
    assert not (tmp_path / ".smartsort_undo.json").exists()


def test_undo_with_no_log(tmp_path):
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    restored, missing, errors = org.undo()
    assert restored == 0
    assert missing == 0
    assert any("no undo log" in e.lower() for e in errors)


def test_undo_handles_missing_source_gracefully(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("x")
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    org.move_files(_plan(str(f), "Canadian_PR_Docs"), apply=True)
    (tmp_path / "Canadian_PR_Docs" / "doc.pdf").unlink()  # user deleted it manually

    restored, missing, errors = Organizer(str(tmp_path), category_names=CATEGORIES).undo()
    assert restored == 0
    assert missing == 1
    assert errors == []


def test_is_already_organized(tmp_path):
    cat = tmp_path / "Canadian_PR_Docs"
    cat.mkdir()
    inside = cat / "x.pdf"
    inside.touch()
    outside = tmp_path / "y.pdf"
    outside.touch()
    org = Organizer(str(tmp_path), category_names=CATEGORIES)
    assert org.is_already_organized(str(inside)) is True
    assert org.is_already_organized(str(outside)) is False
