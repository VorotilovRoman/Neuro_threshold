"""One visible error contract; a failure never becomes an empty successful mask."""

from PyQt5.QtWidgets import QMessageBox

from utils_new.segmentation.result import ResultStatus, SegmentationResult


def require(owner, condition, message):
    if condition:
        return True
    QMessageBox.warning(owner, "Недостаточно данных", message)
    return False


def report(owner, context, error, *, modal=True):
    from ui.segmentation import presentation

    message = f"{context}: {error}"
    owner.result_state = SegmentationResult(ResultStatus.ERROR, error=message)
    owner._raw_mask = None
    owner._raw_signature = None
    owner.final_mask = None
    for name in ("mask", "prediction_mask"):
        if hasattr(owner, name):
            setattr(owner, name, None)
    if hasattr(owner, "current_threshold"):
        owner.current_threshold = None
    if getattr(owner, "current_gray", None) is not None:
        from ui.segmentation.histogram import update_histogram

        update_histogram(owner, owner.current_gray)
    owner.log(f"Ошибка — {message}")
    presentation.show_unavailable(owner, ResultStatus.ERROR, message)
    key = (getattr(owner, "_frame_key", None), message)
    if modal and key != getattr(owner, "_reported_error", None):
        owner._reported_error = key
        QMessageBox.critical(owner, "Ошибка обработки", message)
