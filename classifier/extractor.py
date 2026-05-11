"""File text extraction for the AI classifier path.

PyMuPDF and python-docx are heavy dependencies (~30 MB + native code). The
rules-only worker image doesn't need them, so they're imported lazily and
guarded: importing this module always succeeds, but extracting from a
format whose backend isn't installed quietly returns an empty string. CSV
is read with the stdlib `csv` module — pandas was ~125 MB of dependency
for what amounts to "show me the columns and the first two rows".
"""
import csv
import logging
import os

from classifier.redactor import Redactor

log = logging.getLogger(__name__)


def _try_import_fitz():
    try:
        import pymupdf as fitz  # PyMuPDF 1.24+
    except ImportError:
        try:
            import fitz  # PyMuPDF < 1.24
        except ImportError:
            return None
    fitz.TOOLS.mupdf_display_errors(False)
    return fitz


def _try_import_docx():
    try:
        import docx
        return docx
    except ImportError:
        return None


class FileExtractor:
    PDF_PAGES_TO_SCAN = 3
    DOCX_PARAGRAPHS_TO_SCAN = 30

    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def extract_text(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        text = ""
        try:
            if ext == '.pdf':
                fitz = _try_import_fitz()
                if fitz is None:
                    log.debug("pymupdf not installed; skipping %s", filepath)
                    return ""
                with fitz.open(filepath) as doc:
                    pages = []
                    for i, page in enumerate(doc):
                        if i >= self.PDF_PAGES_TO_SCAN:
                            break
                        pages.append(str(page.get_text()))
                        if sum(len(p) for p in pages) >= self.max_chars:
                            break
                    text = "\n".join(pages)
            elif ext == '.docx':
                docx_mod = _try_import_docx()
                if docx_mod is None:
                    log.debug("python-docx not installed; skipping %s", filepath)
                    return ""
                doc = docx_mod.Document(filepath)
                paras = [p.text for p in doc.paragraphs[:self.DOCX_PARAGRAPHS_TO_SCAN] if p.text.strip()]
                text = "\n".join(paras)
            elif ext == '.csv':
                text = _extract_csv_preview(filepath)
            elif ext in ('.txt', '.log', '.json', '.xml', '.md', '.html', '.eml'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(self.max_chars * 2)
        except Exception:
            return ""

        raw_text = text[:self.max_chars].strip()
        return Redactor.redact(raw_text)


def _extract_csv_preview(filepath: str, max_rows: int = 2) -> str:
    """Return "Columns: a, b, c\\nData:\\n<row1>\\n<row2>" — the same shape
    pandas produced, but using stdlib csv so the worker image doesn't need
    pandas + numpy installed."""
    with open(filepath, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return ""
        rows = []
        for _ in range(max_rows):
            try:
                rows.append(next(reader))
            except StopIteration:
                break
    cols = ", ".join(str(c) for c in header)
    data = "\n".join(", ".join(str(c) for c in r) for r in rows)
    return f"Columns: {cols}\nData:\n{data}"
