import os
import fitz  # PyMuPDF
import docx
import pandas as pd
from classifier.redactor import Redactor

fitz.TOOLS.mupdf_display_errors(False)

class FileExtractor:
    def __init__(self, max_chars=1000):
        self.max_chars = max_chars

    def extract_text(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        text = ""
        try:
            if ext == '.pdf':
                doc = fitz.open(filepath)
                if len(doc) > 0:
                    # Wrap in str() to prevent Pylance "list[Unknown]" type checking errors
                    text = str(doc[0].get_text())
            elif ext == '.docx':
                doc = docx.Document(filepath)
                text = "\n".join([p.text for p in doc.paragraphs[:10]])
            elif ext in ['.csv']:
                df = pd.read_csv(filepath, nrows=5)
                text = f"Columns: {', '.join(df.columns)}\nData:\n{df.head(2).to_string()}"
            elif ext in ['.txt', '.log', '.json', '.xml']:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(self.max_chars)
        except Exception:
            return "" # Fail silently on extraction errors to keep workflow moving
        
        # The type checker now knows 'text' is 100% a string
        raw_text = text[:self.max_chars].strip()
        return Redactor.redact(raw_text)