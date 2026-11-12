"""Shared Qt bindings, staged refresh and deterministic resource cleanup."""

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from ui.segmentation import presentation
from ui.segmentation.jobs import TaskController
from ui.settings_service import settings
from utils_new.segmentation.result import ResultStatus, SegmentationResult


def initialize(owner):
    owner._raw_mask = None
    owner._raw_signature = None
    owner._frame_key = None
    owner._data_revision = 0
    owner.final_mask = None
    owner.result_state = SegmentationResult(ResultStatus.NO_IMAGE)
    owner.jobs = TaskController(owner)
    owner.update_timer = QTimer(owner)
    owner.update_timer.setSingleShot(True)
    owner.update_timer.timeout.connect(owner.display_current_image)
    nav = owner.nav_widget
    for signal, slot in [
        (nav.load_images, owner.load_images_from_dialog),
        (nav.load_folder, owner.load_folder),
        (nav.prev, owner.prev_image),
        (nav.next, owner.next_image),
        (nav.goto_page, owner.goto_image),
        (nav.resize_toggled, owner.on_resize_mode_changed),
    ]:
        signal.connect(slot)
    owner.preset_combo.activated.connect(owner.on_preset_activated)
    owner.save_preset_btn.clicked.connect(owner.save_as_preset)
    owner.delete_preset_btn.clicked.connect(owner.delete_current_preset)
    owner.toggle_log_btn.clicked.connect(owner.toggle_log)
    owner.toggle_hist_btn.clicked.connect(owner.toggle_histogram)
    owner.reset_zoom_button.clicked.connect(owner.reset_all_zooms)
    owner.save_button.clicked.connect(owner.save_current_annotations)
    owner.object_list.itemChanged.connect(owner.on_object_selection_changed)
    owner.draw_combo.currentIndexChanged.connect(owner.on_draw_mode_changed)
    owner.hull_checkbox.stateChanged.connect(owner.on_hull_changed)
    owner.morphology_widget.parameters_changed.connect(
        lambda: schedule(owner, "postprocess")
    )
    owner.invert_checkbox.stateChanged.connect(lambda _: schedule(owner, "postprocess"))
    settings.settings_changed.connect(owner.on_global_settings_changed)
    presentation.set_status(owner, ResultStatus.NO_IMAGE)


def schedule(owner, stage="prediction"):
    owner.update_timer.start(50)


def invalidate_prediction(owner):
    owner._pending_apply = False
    owner.jobs.invalidate()
    for name in ("mask", "prediction_mask"):
        if hasattr(owner, name):
            setattr(owner, name, None)
    owner._raw_signature = None
    owner._raw_mask = None
    owner.result_state = SegmentationResult()


def resize_changed(owner, enabled):
    if (
        owner.image_paths
        and QMessageBox.question(
            owner,
            "Изменение размера",
            "Перезагрузить изображения с новым ограничением размера?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        == QMessageBox.Yes
    ):
        owner.reload_current_images()
    elif owner.image_paths:
        owner.clear_images()


def close(owner):
    owner.update_timer.stop()
    owner.jobs.close()
    cache = getattr(owner, "_lazy_cache", None)
    if cache is not None:
        cache.shutdown()
        owner._lazy_cache = None
