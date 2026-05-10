"""Pluggable classification pipeline.

A ``Classifier`` is anything with a ``name`` attribute and a
``classify(file: FileItem) -> Optional[Classification]`` method. The
``ClassificationPipeline`` runs registered classifiers in order, and the
first one that returns a non-None, known-category ``Classification`` wins.

Adding a new classification source (OCR, hash-dedupe, ML model) becomes a
single new class implementing the protocol.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from classifier.types import Classification, FileItem, UNKNOWN_CATEGORY

log = logging.getLogger(__name__)


@runtime_checkable
class Classifier(Protocol):
    name: str

    def classify(self, file: FileItem) -> Optional[Classification]:
        ...


class ClassificationPipeline:
    """Runs each classifier in registration order, returning the first hit."""

    def __init__(self, classifiers: list[Classifier]):
        self.classifiers = classifiers

    def classify(self, file: FileItem) -> Classification:
        for classifier in self.classifiers:
            try:
                result = classifier.classify(file)
            except Exception:
                log.exception("Classifier %s raised on %s", classifier.name, file.name)
                continue

            if result is None or result.category == UNKNOWN_CATEGORY:
                continue

            log.debug(
                "%s -> %s (%d%%) via %s",
                file.name, result.category, result.confidence, classifier.name,
            )
            return result

        return Classification.unknown()
