"""Filename-rules classification engine.

Two entry points:

* ``high_confidence_match`` — unambiguous filename markers (IMM forms,
  T4, EVL, ``PR_`` prefix, ...). Returns ``Classification`` at confidence
  100, or ``None``.
* ``classify`` — system/hidden file detection, then keyword matches against
  the tokenised filename, then an extension-only fallback for archives
  and installers. Returns a ``Classification``.

The tokeniser splits on underscores, dashes, dots, parentheses, whitespace
*and* camelCase boundaries so ``IMM5476e``, ``previewFormPCCDetail.pdf``,
and ``MSUBARODA_WESEducationalCredentialsForwarding.pdf`` all surface their
intended tokens.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import yaml

from classifier.types import UNKNOWN_CATEGORY, Classification


# (regex, category, reason). Patterns are searched against a tokenised,
# space-separated, lowercased version of the filename so ``\b`` works
# regardless of underscores, dashes, or camelCase in the original name.
HIGH_CONFIDENCE_PATTERNS: list[tuple[str, str, str]] = [
    (r'\bimm\d{4,}\w*', 'Canadian_PR_Docs', "IMM form number in filename"),
    (r'\bircc\b', 'Canadian_PR_Docs', "IRCC marker in filename"),
    (r'\bielts\b', 'Canadian_PR_Docs', "IELTS marker in filename"),
    (r'\bwes\b', 'Canadian_PR_Docs', "WES (credential evaluation) marker"),
    (r'\bnoc\d*\b', 'Canadian_PR_Docs', "NOC code in filename"),
    (r'\blmia\b', 'Canadian_PR_Docs', "LMIA marker in filename"),
    (r'\bpcc\b', 'Canadian_PR_Docs', "PCC (Police Clearance) marker"),
    (r'\beca\b', 'Canadian_PR_Docs', "ECA (credential evaluation) marker"),
    (r'\bita\b', 'Canadian_PR_Docs', "ITA (Invitation to Apply) marker"),
    (r'\bt4\b', 'Canadian_PR_Docs', "T4 (Canadian tax form) used as PR proof"),
    (r'\bexpress entry\b', 'Canadian_PR_Docs', "Express Entry reference"),
    (r'\bemployment verification\b', 'Canadian_PR_Docs', "Employment verification letter (PR proof)"),
    (r'\bpolice clearance\b', 'Canadian_PR_Docs', "Police clearance certificate"),
    (r'\bpermanent resident\b', 'Canadian_PR_Docs', "Permanent resident document"),
    (r'^pr\b', 'Canadian_PR_Docs', "Filename prefix 'PR_' indicates PR document"),
    (r'\bpay slip\b', 'Financial_Taxes', "Pay slip"),
    (r'\bbalance certificate\b', 'Financial_Taxes', "Bank balance certificate"),
]


class RulesEngine:
    HC_METHOD = "Rules (HC)"
    KEYWORD_METHOD = "Rules"

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.categories: dict = self.config['categories']

    @staticmethod
    def _tokenize(filename: str) -> tuple[list[str], str]:
        # Split camelCase: aB -> a B, then ABCd -> AB Cd.
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', filename)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        # Collapse any non-alphanumeric run into a single space.
        s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
        tokens = [t.lower() for t in s.split() if t]
        return tokens, ' '.join(tokens)

    @staticmethod
    def _keyword_matches(keyword: str, tokens: list[str], flat: str) -> bool:
        kw = keyword.lower().strip()
        if not kw:
            return False
        if ' ' in kw:
            return kw in flat
        if kw in tokens:
            return True
        # Allow simple suffix tolerance for longer keywords (e.g. doctor -> doctors).
        if len(kw) >= 5:
            return any(t.startswith(kw) for t in tokens)
        return False

    # ------------------------------------------------------------------ public

    def high_confidence_match(self, filepath: str) -> Optional[Classification]:
        """Return a Classification for unambiguous filename markers, else None."""
        filename = os.path.basename(filepath)
        _, flat = self._tokenize(filename)
        for pattern, cat, reason in HIGH_CONFIDENCE_PATTERNS:
            if re.search(pattern, flat):
                return Classification(category=cat, confidence=100, method=self.HC_METHOD, reason=reason)
        return None

    def classify(self, filepath: str) -> Classification:
        """Classify by system extension, keyword, or archive fallback."""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        # 1. System / hidden files
        meta_exts = self.categories.get('Metadata_System', {}).get('extensions', [])
        if ext in meta_exts or filename.startswith('.'):
            return Classification(
                category="Metadata_System",
                confidence=100,
                method=self.KEYWORD_METHOD,
                reason="System/Hidden file detected by extension",
            )

        # 2. Tokenised keyword match. Multi-word phrases first so the more
        #    specific signal wins (e.g. "air india" beats a generic "ticket").
        tokens, flat = self._tokenize(filename)
        for phrase_pass in (True, False):
            for cat, data in self.categories.items():
                if ext and ext not in data.get('extensions', []):
                    continue
                for keyword in data.get('keywords', []):
                    is_phrase = ' ' in str(keyword).strip()
                    if is_phrase != phrase_pass:
                        continue
                    if self._keyword_matches(keyword, tokens, flat):
                        return Classification(
                            category=cat,
                            confidence=95,
                            method=self.KEYWORD_METHOD,
                            reason=f"Matched '{keyword}' in filename",
                        )

        # 3. Extension-only fallback for archives / installers
        archives_exts = self.categories.get('Archives_and_Apps', {}).get('extensions', [])
        if ext in archives_exts:
            return Classification(
                category="Archives_and_Apps",
                confidence=75,
                method=self.KEYWORD_METHOD,
                reason=f"Archive/installer fallback by extension '{ext}'",
            )

        return Classification.unknown(reason="No rule matched", method=self.KEYWORD_METHOD)
