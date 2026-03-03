import json
import os
from typing import Dict


class LabelManager:
    """Manage mapping between species names and label indices.

    Labels are persisted in a JSON file so that the model knows which
    output index corresponds to each species. When new species are added
    (e.g. during training), the JSON file is updated and the model's
    final linear layer can be reconfigured accordingly.
    """

    def __init__(self, path: str):
        self.path = path
        self.labels: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            # start with empty mapping
            self.labels = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, sort_keys=True)

    def add_label(self, name: str) -> int:
        """Return an index for the given species name, adding it if new."""
        if name in self.labels:
            return self.labels[name]
        idx = len(self.labels)
        self.labels[name] = idx
        self._save()
        return idx

    def get_index(self, name: str) -> int:
        return self.labels[name]

    def get_name(self, index: int) -> str:
        for k, v in self.labels.items():
            if v == index:
                return k
        raise KeyError(f"No species found for index {index}")

    def num_classes(self) -> int:
        return len(self.labels)

    def all_labels(self):
        return list(self.labels.keys())
