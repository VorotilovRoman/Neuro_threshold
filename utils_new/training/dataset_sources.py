"""Ultralytics bridge for per-dataset approved label paths, including missing TXT."""

import hashlib
from pathlib import Path

from runtime_torch import require_torch

require_torch()
from ultralytics.data.dataset import YOLODataset


class SelectedYOLODataset(YOLODataset):
    @property
    def label_files(self):
        return self._selected_label_files

    @label_files.setter
    def label_files(self, defaults):
        selection = self.data.get("_dataset_sources") if self.data else None
        if not selection:
            self._selected_label_files = defaults
            return
        labels = selection["annotations"]
        missing = Path(selection["missing_dir"])
        missing.mkdir(parents=True, exist_ok=True)
        self._selected_label_files = [
            labels.get(str(Path(image).resolve()))
            or str(
                missing
                / (
                    hashlib.sha256(str(Path(image).resolve()).encode()).hexdigest()
                    + ".txt"
                )
            )
            for image in self.im_files
        ]
