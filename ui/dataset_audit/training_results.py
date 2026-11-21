from ui import dataset_source as source_dialogs

"""Dataset analysis: training_results Qt controller."""

import os

from PyQt5.QtGui import QPixmap

from utils_new.dataset_audit.analysis import TrainingResultsLoader


class DatasetTrainingResultsController:
    def browse_training_results(self):
        folder = source_dialogs.get_directory(self, "training_run")
        if folder:
            self.training_results_edit.setText(folder)

    def load_training_results(self):
        folder = self.training_results_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            folder = source_dialogs.get_directory(self, "training_run")
            if not folder:
                return
            self.training_results_edit.setText(folder)

        data = TrainingResultsLoader.scan(folder)
        if data.get("error"):
            self.training_results_summary_label.setText(f"❌ {data['error']}")
            self._last_training_results = data
            return

        self._last_training_results = data
        self.training_results_summary_label.setText(
            TrainingResultsLoader.summary_text(data).replace("\n", "<br>")
        )

        keys_map = {
            "results": "results.png",
            "labels": "labels.jpg",
            "confusion": "confusion_matrix_normalized.png",
        }
        for key, fname in keys_map.items():
            widget = getattr(self, f"training_img_{key}", None)
            if widget is None:
                continue
            path = data["main"].get(fname)
            if not path:
                widget.set_image(None)
                continue
            pix = QPixmap(path)
            widget.set_image(pix)

        self._audit_log(
            f"Результаты обучения загружены из {folder}: "
            f"main={sorted(data['main'].keys())}, "
            f"curves={len(data['curves'])}, "
            f"batches={len(data['batches'])}, "
            f"weights={sorted(data['weights'].keys())}",
            tag="TRAIN",
            level="OK",
        )
