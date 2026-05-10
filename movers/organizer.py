import os
import shutil
import json
from datetime import datetime


class Organizer:
    UNDO_FILENAME = '.smartsort_undo.json'

    def __init__(self, target_dir: str, category_names=None):
        self.target_dir = target_dir
        self.undo_log = os.path.join(target_dir, self.UNDO_FILENAME)
        self.history = []
        # Set of folder names that are managed by SmartSort. Files already
        # nested inside one of these are considered "already organized" and
        # are skipped on subsequent runs.
        self.category_names = set(category_names or [])

    def is_already_organized(self, filepath: str) -> bool:
        try:
            rel = os.path.relpath(filepath, self.target_dir)
        except ValueError:
            return False
        head = rel.split(os.sep, 1)[0]
        return head in self.category_names

    def move_files(self, classification_plan: dict, apply: bool = False):
        if not apply:
            return

        for filepath, data in classification_plan.items():
            category = data['category']
            if category in ("Unknown_Unsorted", "Metadata_System"):
                continue

            dest_dir = os.path.join(self.target_dir, category)
            os.makedirs(dest_dir, exist_ok=True)

            filename = os.path.basename(filepath)
            dest_path = os.path.join(dest_dir, filename)

            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(filepath, dest_path)
            self.history.append({
                "original": filepath,
                "new": dest_path,
                "category": category,
                "timestamp": datetime.now().isoformat(),
            })

        self._save_undo_log()

    def _save_undo_log(self):
        if not self.history:
            return
        existing = []
        if os.path.exists(self.undo_log):
            try:
                with open(self.undo_log, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.extend(self.history)
        with open(self.undo_log, 'w') as f:
            json.dump(existing, f, indent=2)

    def undo(self) -> tuple[int, int, list[str]]:
        """Reverse the most recent sort. Returns (restored, missing, errors)."""
        if not os.path.exists(self.undo_log):
            return 0, 0, ["No undo log found in this directory."]

        with open(self.undo_log, 'r') as f:
            history = json.load(f)

        restored, missing = 0, 0
        errors: list[str] = []
        # Reverse so the most recent move is undone first.
        for entry in reversed(history):
            src = entry.get('new')
            dst = entry.get('original')
            if not src or not dst:
                continue
            if not os.path.exists(src):
                missing += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                # Don't clobber a file that was re-created at the original path.
                if os.path.exists(dst):
                    base, ext = os.path.splitext(dst)
                    n = 1
                    while os.path.exists(f"{base}_restored_{n}{ext}"):
                        n += 1
                    dst = f"{base}_restored_{n}{ext}"
                shutil.move(src, dst)
                restored += 1
            except OSError as e:
                errors.append(f"{src} -> {dst}: {e}")

        # Clean up empty category directories left behind.
        for entry in history:
            cat_dir = os.path.dirname(entry.get('new', ''))
            if cat_dir and os.path.isdir(cat_dir) and not os.listdir(cat_dir):
                try:
                    os.rmdir(cat_dir)
                except OSError:
                    pass

        os.remove(self.undo_log)
        return restored, missing, errors
