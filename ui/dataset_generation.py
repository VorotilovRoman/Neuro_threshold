"""Qt adapter around the headless dataset generator."""

from PyQt5.QtCore import QThread, pyqtSignal
from utils_new.import_export.dataset_generator import DatasetGeneratorCoreNew


class DatasetGeneratorThread(QThread):
    """Qt signals and cancellation for the dataset preparation tab."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["log_cb"] = self.log_signal.emit
        kwargs["progress_cb"] = self.progress_signal.emit
        self.engine = DatasetGeneratorCoreNew(*args, **kwargs)

    def cancel(self):
        self.engine.cancel()

    def run(self):
        success, message = self.engine.execute()
        self.finished_signal.emit(success, message)
