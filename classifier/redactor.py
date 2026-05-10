import re

class Redactor:
    PATTERNS = {
        "EMAIL": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "PHONE": r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        "URL": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
        "JWT_TOKEN": r'eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',
        "AWS_KEY": r'(?i)AKIA[0-9A-Z]{16}',
    }

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return ""
        for entity, pattern in cls.PATTERNS.items():
            text = re.sub(pattern, f"[{entity}_REDACTED]", text)
        return text