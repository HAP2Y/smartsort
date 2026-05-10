"""Concrete pipeline classifiers wrapping the existing engines."""
from __future__ import annotations

import logging
from typing import Optional

from classifier.ai_local import LocalAIClassifier as _LocalAI
from classifier.extractor import FileExtractor
from classifier.rules import RulesEngine
from classifier.types import Classification, FileItem, UNKNOWN_CATEGORY

log = logging.getLogger(__name__)


class HighConfidenceRulesClassifier:
    """Unambiguous filename markers (IMM forms, T4, EVL, PR_ prefix, ...)."""
    name = "Rules (HC)"

    def __init__(self, rules: RulesEngine):
        self._rules = rules

    def classify(self, file: FileItem) -> Optional[Classification]:
        return self._rules.high_confidence_match(str(file.path))


class RulesClassifier:
    """System / hidden + keyword + archive-extension fallback rules."""
    name = "Rules"

    def __init__(self, rules: RulesEngine):
        self._rules = rules

    def classify(self, file: FileItem) -> Optional[Classification]:
        result = self._rules.classify(str(file.path))
        return result if result.category != UNKNOWN_CATEGORY else None


class LocalAIPipelineClassifier:
    """Extract text, prompt the local model, accept above-threshold results."""
    name = "Local AI"

    def __init__(
        self,
        ai: _LocalAI,
        extractor: FileExtractor,
        categories: list[str],
        threshold: int,
        enabled: bool = True,
    ):
        self._ai = ai
        self._extractor = extractor
        self._categories = categories
        self._threshold = threshold
        self.enabled = enabled

    def classify(self, file: FileItem) -> Optional[Classification]:
        if not self.enabled:
            return None
        snippet = self._extractor.extract_text(str(file.path))
        if not (snippet and snippet.strip()):
            log.debug("AI: no extractable text for %s", file.name)
            return None
        result = self._ai.classify(file.name, snippet, self._categories)
        if not result.is_known:
            return None
        if result.confidence < self._threshold:
            log.debug(
                "AI: %s below threshold (%d < %d)",
                file.name, result.confidence, self._threshold,
            )
            return None
        return result
