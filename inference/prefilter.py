"""Dispatcher-side cheap classification.

The distributed path is deliberately **AI-first**: anything an AI worker can
read goes to a worker so the LLM weighs in. But some files can never reach a
worker — archives, installers, spreadsheets, extensionless files, and (until
an OCR worker exists) images. Previously those were stamped
``Unknown_Unsorted`` on the dispatcher, which threw away easy wins: ``.DS_Store``
and ``Ollama.dmg`` are trivially classifiable by extension alone.

``Prefilter`` runs the same filename rules the inline pipeline uses, but only
for files the router could not hand to a worker. Files that *can* reach a
worker are untouched, so AI-first is preserved for everything AI can read.
"""
from __future__ import annotations

import logging

from classifier.classifiers import HighConfidenceRulesClassifier, RulesClassifier
from classifier.pipeline import ClassificationPipeline
from classifier.rules import RulesEngine
from classifier.types import Classification, FileItem

log = logging.getLogger(__name__)


class Prefilter:
    """Filename-rules classification for files no worker can handle."""

    def __init__(self, rules: RulesEngine):
        self._pipeline = ClassificationPipeline(
            [HighConfidenceRulesClassifier(rules), RulesClassifier(rules)]
        )

    def classify(self, file: FileItem) -> Classification:
        """Classify by filename rules. Unknown_Unsorted if nothing matches."""
        result = self._pipeline.classify(file)
        if result.is_known:
            log.debug("prefilter: %s -> %s", file.name, result.category)
            return result
        return Classification.unknown(
            reason="no worker can read this file type and no rule matched",
            method="Prefilter",
        )
