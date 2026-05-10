"""Tests for ``classifier.redactor.Redactor``."""
from classifier.redactor import Redactor


def test_redact_empty_returns_empty():
    assert Redactor.redact("") == ""
    assert Redactor.redact(None) == ""  # type: ignore[arg-type]


def test_redact_email():
    out = Redactor.redact("Contact: happy.patel@example.com please.")
    assert "happy.patel@example.com" not in out
    assert "[EMAIL_REDACTED]" in out


def test_redact_url():
    out = Redactor.redact("See https://example.com/path?q=1 for details.")
    assert "https://example.com" not in out
    assert "[URL_REDACTED]" in out


def test_redact_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.signature"
    out = Redactor.redact(f"token={jwt}")
    assert jwt not in out
    assert "[JWT_TOKEN_REDACTED]" in out


def test_redact_aws_key():
    out = Redactor.redact("aws key AKIAIOSFODNN7EXAMPLE rotated")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[AWS_KEY_REDACTED]" in out


def test_redact_phone():
    out = Redactor.redact("Call +1 (415) 555-1234 today.")
    assert "555-1234" not in out
    assert "[PHONE_REDACTED]" in out


def test_redact_preserves_non_pii_content():
    out = Redactor.redact("Invoice #INV-2026-001 for HDFC bank statement.")
    # The bank statement context must survive — no PII patterns to redact.
    assert "Invoice" in out
    assert "HDFC" in out
    assert "bank statement" in out


def test_redact_handles_multiple_entities():
    out = Redactor.redact("From a@b.com see https://x.io and AKIAIOSFODNN7EXAMPLE")
    assert "[EMAIL_REDACTED]" in out
    assert "[URL_REDACTED]" in out
    assert "[AWS_KEY_REDACTED]" in out
