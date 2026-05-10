"""Local AI classifier backed by Ollama.

Split into three concerns so each can be tested in isolation:

* ``OllamaClient`` — raw HTTP. Pluggable session for tests.
* ``PROMPT_TEMPLATE`` and ``build_prompt`` — pure string construction.
* ``parse_response`` — pure JSON parsing with error categorisation.

``LocalAIClassifier`` glues them together and is the public API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from classifier.types import UNKNOWN_CATEGORY, Classification

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60
DEFAULT_HEALTH_TIMEOUT = 3
METHOD = "Local AI"

PROMPT_TEMPLATE = """You are a document classification AI. Classify the file into EXACTLY ONE category:
{categories}

Filename: {filename}
Extracted Text Snippet: {snippet}

DISAMBIGUATION RULES (apply in order):
1. Canadian_PR_Docs is the bucket for everything the user is collecting for a Canadian Permanent Residence application. This INCLUDES: employment verification letters (any employer, any country, including Guidewire), reference letters used as PR proof, T4 slips and pay slips when used as proof-of-employment, IMM forms, IRCC paperwork, IELTS / WES / ECA results, NOC references, LMIA, PCC (Police Clearance Certificates), ITA, retainer agreements with immigration consultants, and Express Entry profiles.
2. Resumes_Career_Tech is ONLY for actual resumes / CVs (career-summary documents listing the person's own skills and experience), cover letters, certifications, interview prep, and portfolios. It is NOT for employment verification letters, even if they describe a role.
3. Guidewire_PSE_Work is strictly for internal Guidewire operational artifacts: JIRA tickets, support cases, stack traces, logs, platform/system specs, customer SAML metadata, internal policies, and EA / provisioning emails. It is NOT for the user's own employment verification letters or T4s, even when the employer is Guidewire — those go to Canadian_PR_Docs.
4. Financial_Taxes covers personal banking, credit-card statements, EMI, receipts, invoices, vouchers, and utility bills. T4s and pay slips submitted as PR proof go to Canadian_PR_Docs (rule 1) instead.
5. Travel_Transit is for itineraries, boarding passes, e-tickets, and hotel reservations only. Business / market-research reports about a location are NOT travel.
6. Use the filename as a strong tie-breaker, especially prefixes like 'PR_', 'Canada_', or form codes like IMM####, T4, PCC, NOC.
7. If the snippet is mostly redacted or generic and you cannot confidently classify, return Unknown_Unsorted with a confidence below 80.

Return ONLY valid JSON in this exact format, with no markdown formatting or backticks:
{{"category": "Category_Name", "confidence": 95, "reason": "brief explanation grounded in the text content"}}
"""


def build_prompt(filename: str, snippet: str, categories: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        categories=", ".join(categories),
        filename=filename,
        snippet=snippet,
    )


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def parse_response(raw_reply: str, categories: list[str]) -> Classification:
    """Parse a raw Ollama ``response`` string into a Classification.

    Returns Unknown_Unsorted if the model returned empty / invalid JSON or
    a category outside the allowed set.
    """
    if not raw_reply or not raw_reply.strip():
        return Classification.unknown(reason="Ollama returned an empty string", method=METHOD)

    cleaned = _strip_code_fence(raw_reply)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return Classification.unknown(reason="AI returned invalid JSON", method=METHOD)

    if not isinstance(data, dict):
        return Classification.unknown(reason="AI returned non-object JSON", method=METHOD)

    cat = str(data.get("category", UNKNOWN_CATEGORY))
    reason = str(data.get("reason", "AI classification"))
    try:
        confidence = int(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    if cat not in categories:
        return Classification.unknown(
            reason=f"AI suggested unknown category {cat!r}", method=METHOD
        )

    return Classification(category=cat, confidence=confidence, method=METHOD, reason=reason)


@dataclass
class HealthStatus:
    ok: bool
    message: str


class OllamaClient:
    """Thin HTTP client. Pluggable session/url so tests can mock it."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        session: Optional[requests.Session] = None,
        timeout: int = DEFAULT_TIMEOUT,
        health_timeout: int = DEFAULT_HEALTH_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.health_timeout = health_timeout

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def health(self, model: str) -> HealthStatus:
        try:
            health = self.session.get(self.base_url, timeout=self.health_timeout)
            if health.status_code != 200:
                return HealthStatus(False, f"Ollama server returned HTTP {health.status_code}")

            tags = self.session.get(self.tags_url, timeout=self.health_timeout)
            if tags.status_code == 200:
                names = [m.get("name", "") for m in tags.json().get("models", [])]
                if not any(model in n for n in names):
                    return HealthStatus(
                        False,
                        f"Model '{model}' not pulled. Run: ollama pull {model}",
                    )
            return HealthStatus(True, f"Ollama is running and {model} is available.")
        except requests.exceptions.ConnectionError:
            return HealthStatus(False, "Ollama connection refused. Is the Ollama app open?")
        except requests.exceptions.Timeout:
            return HealthStatus(False, "Ollama health check timed out.")
        except Exception as e:  # pragma: no cover - defensive
            return HealthStatus(False, f"Ollama health check failed: {e!s}")

    def generate(self, model: str, prompt: str) -> tuple[bool, str]:
        """Returns (ok, response_text_or_error_message)."""
        payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
        try:
            r = self.session.post(self.generate_url, json=payload, timeout=self.timeout)
        except requests.exceptions.ReadTimeout:
            return False, "Ollama timed out"
        except requests.exceptions.ConnectionError:
            return False, "Ollama connection refused"
        except Exception as e:  # pragma: no cover - defensive
            return False, f"Error: {type(e).__name__}"

        if r.status_code != 200:
            body_excerpt = r.text[:80] if r.text else ""
            return False, f"HTTP {r.status_code}: {body_excerpt}"

        try:
            return True, r.json().get("response", "")
        except ValueError:
            return False, "Ollama returned non-JSON envelope"


class LocalAIClassifier:
    """High-level classifier: health-checks Ollama, then prompts the model."""

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        client: Optional[OllamaClient] = None,
    ):
        self.model = model
        self.client = client or OllamaClient()

    def is_running(self) -> tuple[bool, str]:
        status = self.client.health(self.model)
        return status.ok, status.message

    def classify(self, filename: str, snippet: str, categories: list[str]) -> Classification:
        prompt = build_prompt(filename, snippet, categories)
        ok, body = self.client.generate(self.model, prompt)
        if not ok:
            return Classification.unknown(reason=body, method=METHOD)
        return parse_response(body, categories)
