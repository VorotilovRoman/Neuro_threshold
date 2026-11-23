"""Wait for observable asynchronous dataset work while allowing Qt to dispatch events."""


def wait_dataset(owner, timeout=15000):
    from time import monotonic
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtTest import QTest

    def busy():
        tasks = getattr(owner, "_dataset_tasks", None)
        if tasks is not None and tasks.busy():
            return True
        for name in ("overlap_review", "_overlap_review_panel"):
            worker = getattr(getattr(owner, name, None), "_worker", None)
            if worker is not None and worker.isRunning():
                return True
        return False

    deadline = monotonic() + timeout / 1000
    while monotonic() < deadline:
        QApplication.processEvents()
        if not busy():
            QApplication.processEvents()
            if not busy():
                return
        QTest.qWait(5)
    raise AssertionError("Dataset background tasks did not settle")
