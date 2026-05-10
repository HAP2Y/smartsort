import os
import re
import yaml


# (regex, category, reason). Patterns are searched against a tokenized,
# space-separated, lowercased version of the filename so `\b` works reliably
# regardless of underscores, dashes, or camelCase in the original name.
HIGH_CONFIDENCE_PATTERNS = [
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
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.categories = self.config['categories']

    @staticmethod
    def _tokenize(filename: str) -> tuple[list[str], str]:
        # Split camelCase: aB -> a B, and ABCd -> AB Cd.
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', filename)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        # Collapse any non-alphanumeric run (underscores, dashes, dots, parens,
        # commas, whitespace, etc.) into a single space.
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
        # Allow simple suffix tolerance for longer keywords (e.g., "doctor" -> "doctors")
        if len(kw) >= 5:
            return any(t.startswith(kw) for t in tokens)
        return False

    def high_confidence_match(self, filepath: str):
        """Return (category, 100, reason) for unambiguous filename markers, else None."""
        filename = os.path.basename(filepath)
        _, flat = self._tokenize(filename)
        for pattern, cat, reason in HIGH_CONFIDENCE_PATTERNS:
            if re.search(pattern, flat):
                return cat, 100, reason
        return None

    def classify(self, filepath: str) -> tuple[str, int, str]:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        # 1. System/hidden files
        meta_exts = self.categories.get('Metadata_System', {}).get('extensions', [])
        if ext in meta_exts or filename.startswith('.'):
            return "Metadata_System", 100, "System/Hidden file detected by extension"

        # 2. Tokenized filename keyword match. Multi-word phrases are checked
        #    before single-word keywords so the more specific signal wins
        #    (e.g. "air india" beats a generic "ticket").
        tokens, flat = self._tokenize(filename)
        for phrase_pass in (True, False):
            for cat, data in self.categories.items():
                if ext and ext not in data.get('extensions', []):
                    continue
                for keyword in data.get('keywords', []):
                    is_phrase = ' ' in keyword.strip()
                    if is_phrase != phrase_pass:
                        continue
                    if self._keyword_matches(keyword, tokens, flat):
                        return cat, 95, f"Matched '{keyword}' in filename"

        # 3. Extension-only fallback for archives/installers so generic
        #    .zip / .dmg / .pkg files don't end up in Unknown.
        archives_exts = self.categories.get('Archives_and_Apps', {}).get('extensions', [])
        if ext in archives_exts:
            return "Archives_and_Apps", 75, f"Archive/installer fallback by extension '{ext}'"

        return "Unknown_Unsorted", 0, "No rule matched"
