import json
import os
from collections import defaultdict

FEEDBACK_PATH = "artifacts/feedback.json"


class FeedbackStore:
    def __init__(self):
        self.feedback = defaultdict(lambda: defaultdict(int))
        self._load()

    def _load(self):
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r") as f:
                raw = json.load(f)
                for shape, items in raw.items():
                    for image_id, count in items.items():
                        self.feedback[shape][int(image_id)] = count

    def _save(self):
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(self.feedback, f, indent=2)

     
    def add_feedback(self, shape: str, image_id: int):
        self.feedback[shape][image_id] += 1
        self._save()

    # Used during ranking
    def get_boost(self, shape: str, image_id: int) -> float:
        return 0.02 * self.feedback[shape].get(image_id, 0)
