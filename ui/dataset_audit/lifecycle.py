"""Guard re-entry during legacy synchronous scans and reject stale ML results."""

from functools import wraps
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractButton,
    QComboBox,
    QLineEdit,
    QAbstractSpinBox,
    QSlider,
)


def serialized_scan(function):
    @wraps(function)
    def run(owner, *args, **kwargs):
        if owner._scan_busy or owner._closing:
            return
        owner._scan_busy = True
        revision = owner._dataset_revision
        controls = owner.findChildren(
            (
                QAbstractButton,
                QComboBox,
                QLineEdit,
                QAbstractSpinBox,
                QSlider,
                QAbstractItemView,
            )
        )
        stops = {
            getattr(owner, name, None)
            for name in ("btn_stop_quality", "btn_stop_duplicates", "btn_stop_heatmap")
        }
        previous = [
            (widget, widget.isEnabled()) for widget in controls if widget not in stops
        ]
        for widget, _ in previous:
            widget.setEnabled(False)
        try:
            return function(owner, *args, **kwargs)
        finally:
            for widget, enabled in previous:
                widget.setEnabled(enabled)
            owner._scan_busy = False
            if owner._dataset_revision != revision:
                for name in (
                    "btn_run_quality",
                    "btn_run_duplicates",
                    "btn_run_heatmap",
                    "btn_run_audit",
                    "btn_run_cluster",
                    "btn_run_validation",
                    "btn_generate_report",
                ):
                    getattr(owner, name).setEnabled(True)
            if owner._closing:
                QTimer.singleShot(0, owner.window().close)

    return run


def stale(owner, result=None):
    if (
        result is not None
        and result.get("_dataset_revision", owner._dataset_revision)
        != owner._dataset_revision
    ):
        return True
    worker = owner.sender()
    return owner._closing or (
        worker is not None
        and getattr(worker, "revision", owner._dataset_revision)
        != owner._dataset_revision
    )


def close_workers(owner):
    owner._closing = True
    owner._stop_requested = True
    for name in ("_audit_thread", "_cluster_thread", "_report_thread", "_val_thread"):
        worker = getattr(owner, name, None)
        if worker is not None:
            worker.close()
