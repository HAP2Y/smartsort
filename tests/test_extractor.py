"""Tests for ``classifier.extractor.FileExtractor``.

Tests focus on the format dispatch, redaction integration, char-cap behaviour,
and graceful failure on missing / unsupported files. PDF parsing itself is
delegated to PyMuPDF and tested via a real (but tiny) generated PDF when the
library exposes the writer API; otherwise that test is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from classifier.extractor import FileExtractor


def test_unknown_extension_returns_empty(tmp_path):
    p = tmp_path / "weird.xyz"
    p.write_text("hello")
    assert FileExtractor().extract_text(str(p)) == ""


def test_missing_file_returns_empty(tmp_path):
    assert FileExtractor().extract_text(str(tmp_path / "ghost.pdf")) == ""


@pytest.mark.parametrize("ext", [".txt", ".log", ".json", ".xml", ".md", ".html", ".eml"])
def test_text_like_extensions_extract_and_redact(tmp_path, ext):
    p = tmp_path / f"doc{ext}"
    p.write_text("Contact happy@example.com about your invoice.\n")
    out = FileExtractor(max_chars=200).extract_text(str(p))
    assert "happy@example.com" not in out
    assert "[EMAIL_REDACTED]" in out
    assert "invoice" in out


def test_text_extraction_respects_max_chars(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("A" * 10_000)
    out = FileExtractor(max_chars=100).extract_text(str(p))
    assert len(out) <= 100


def test_csv_extracts_columns_and_sample_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("date,close,volume\n2026-01-01,100.0,1000\n2026-01-02,101.5,1100\n")
    out = FileExtractor(max_chars=500).extract_text(str(p))
    assert "Columns:" in out
    assert "date" in out and "close" in out and "volume" in out


def test_docx_extracts_paragraphs(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_paragraph("Employment Verification Letter")
    d.add_paragraph("This confirms Happy Patel's employment.")
    d.save(str(p))
    out = FileExtractor(max_chars=500).extract_text(str(p))
    assert "Employment Verification Letter" in out


def test_pdf_extracts_first_page_text(tmp_path):
    fitz = pytest.importorskip("pymupdf", reason="needs PyMuPDF >= 1.24")
    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Permanent Resident application")
    doc.save(str(p))
    doc.close()
    out = FileExtractor(max_chars=500).extract_text(str(p))
    assert "Permanent Resident" in out


def test_corrupt_pdf_returns_empty_string(tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a real pdf")
    assert FileExtractor().extract_text(str(p)) == ""
