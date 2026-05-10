import os
import yaml
import re

class RulesEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.categories = self.config['categories']

    def classify(self, filepath: str) -> tuple[str, int, str]:
        filename = os.path.basename(filepath).lower()
        ext = os.path.splitext(filename)[1].lower()

        # 1. Check Metadata/Junk first
        if ext in self.categories.get('Metadata_System', {}).get('extensions', []) or filename.startswith('.'):
            return "Metadata_System", 100, "System/Hidden file detected by extension"

        # 2. Check strict filename keywords using word boundaries
        for cat, data in self.categories.items():
            if ext and ext not in data.get('extensions', []):
                continue
            
            for keyword in data.get('keywords', []):
                # \b means "word boundary". This ensures 'pr' only matches the exact word 'pr', 
                # and NOT substrings inside 'print' or 'april'
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, filename):
                    return cat, 95, f"Matched exact word '{keyword}' in filename"

        return "Unknown_Unsorted", 0, "No rule matched"