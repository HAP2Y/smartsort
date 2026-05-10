import os
import fitz  # PyMuPDF
import docx
import pandas as pd
from classifier.redactor import Redactor

fitz.TOOLS.mupdf_display_errors(False)


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
                doc = docx.Document(filepath)
                paras = [p.text for p in doc.paragraphs[:self.DOCX_PARAGRAPHS_TO_SCAN] if p.text.strip()]
                text = "\n".join(paras)
            elif ext == '.csv':
                df = pd.read_csv(filepath, nrows=5)
                text = f"Columns: {', '.join(map(str, df.columns))}\nData:\n{df.head(2).to_string()}"
            elif ext in ('.txt', '.log', '.json', '.xml', '.md', '.html', '.eml'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(self.max_chars * 2)
        except Exception:
            return ""

        raw_text = text[:self.max_chars].strip()
        return Redactor.redact(raw_text)
