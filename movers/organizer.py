import os
import shutil
import json
from datetime import datetime

class Organizer:
    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        self.undo_log = os.path.join(target_dir, '.smartsort_undo.json')
        self.history = []

    def move_files(self, classification_plan: dict, apply: bool = False):
        if not apply:
            return # Dry run, do nothing
            
        for filepath, data in classification_plan.items():
            category = data['category']
            if category == "Unknown_Unsorted" or category == "Metadata_System":
                continue # Optionally, we don't move unknown/system files
                
            dest_dir = os.path.join(self.target_dir, category)
            os.makedirs(dest_dir, exist_ok=True)
            
            filename = os.path.basename(filepath)
            dest_path = os.path.join(dest_dir, filename)
            
            # Handle collisions
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1
                
            shutil.move(filepath, dest_path)
            self.history.append({"original": filepath, "new": dest_path, "timestamp": datetime.now().isoformat()})
            
        self._save_undo_log()

    def _save_undo_log(self):
        with open(self.undo_log, 'w') as f:
            json.dump(self.history, f, indent=2)