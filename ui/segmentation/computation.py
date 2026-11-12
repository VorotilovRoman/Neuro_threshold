"""GUI-side snapshots and acceptance of numerical tasks. No QWidget reaches workers."""

from functools import wraps
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ui.segmentation import errors, presentation, session
from utils_new.segmentation import operations
from utils_new.segmentation.result import ResultStatus, SegmentationResult, binary_mask


def gui_action(function):
    @wraps(function)
    def run(owner, *args, **kwargs):
        try:
            return function(owner, *args, **kwargs)
        except Exception as exc:
            errors.report(owner, "Параметры операции", exc)

    return run


def frame_for(owner):
    try:
        frame = session.current_frame(owner)
        if not errors.require(owner, frame is not None, "Загрузите изображение."):
            return None
        return frame
    except Exception as exc:
        errors.report(owner, "Чтение изображения", exc)
        return None


def accept_mask(owner, attribute, mask, shape):
    setattr(owner, attribute, binary_mask(mask, shape))
    owner._raw_signature = None
    owner.display_current_image()


@gui_action
def run_interactive(owner):
    frame = frame_for(owner)
    if frame is None or owner.jobs.busy:
        return
    method = owner.method_combo.currentText()
    widget = owner.params_widget
    if method in ("SuperCut", "OneCut", "Active Contours") and not errors.require(
        owner, owner.rect is not None, "Нарисуйте ROI."
    ):
        return
    fg = (
        np.zeros(frame.gray.shape, np.uint8)
        if owner.fg_scribbles is None
        else owner.fg_scribbles.copy()
    )
    bg = (
        np.zeros(frame.gray.shape, np.uint8)
        if owner.bg_scribbles is None
        else owner.bg_scribbles.copy()
    )
    if method == "Random Walker" and not errors.require(
        owner, bool((fg == 255).any() and (bg == 255).any()), "Нанесите штрихи FG и BG."
    ):
        return
    getters = {
        "GrabCut": widget.get_grabcut_params,
        "Watershed (marker)": widget.get_watershed_params,
        "Lazy Snapping": widget.get_superpixel_params,
        "SuperCut": widget.get_supercut_params,
        "OneCut": widget.get_onecut_params,
        "Random Walker": widget.get_random_walker_params,
        "Active Contours": widget.get_active_contour_params,
    }
    params = getters[method]()
    rect = owner.rect
    image, gray = frame.image.copy(), frame.gray.copy()
    owner.jobs.start(
        "Интерактивная сегментация",
        lambda cancel: operations.interactive(
            method, image, gray, rect, fg, bg, params, cancel
        ),
        lambda value: accept_mask(owner, "mask", value, gray.shape),
    )


@gui_action
def run_dl(owner):
    frame = frame_for(owner)
    if (
        frame is None
        or owner.jobs.busy
        or not errors.require(owner, owner.segmentor is not None, "Загрузите модель.")
    ):
        return
    name = owner.model_combo.currentText()
    parameters = {}
    overrides = {}
    if not errors.require(
        owner, name not in ("SegFormer", "SAM"), f"Модель {name} ещё не реализована."
    ):
        return
    if name == "YOLO-seg":
        widget = owner.yolo_settings
        parameters = dict(
            conf=widget._conf.value(),
            iou=widget._iou.value(),
            imgsz=widget._imgsz.value(),
            save=widget._save.isChecked(),
        )
    if name in ("U-Net", "DeepLabV3+", "DeepLabV3"):
        widget = owner.unet_settings if name == "U-Net" else owner.deeplab_settings
        valid, message = widget.validate_for_prediction()
        if not errors.require(owner, valid, message):
            return
        parameters["threshold"] = widget._threshold.value()
        overrides = dict(widget.get_override_params())
    model = owner.segmentor
    image = frame.image.copy()
    owner.jobs.start(
        "Сегментация модели",
        lambda cancel: operations.predict_dl(
            model, image, parameters, overrides, cancel
        ),
        lambda value: accept_mask(owner, "prediction_mask", value, image.shape),
    )


def cluster_result(owner, result):
    mask, unique, means = result
    owner.current_cluster_labels = mask
    owner.cluster_means = means
    previous = owner.cluster_combo.blockSignals(True)
    try:
        owner.cluster_combo.clear()
        for label, mean in zip(unique, means):
            owner.cluster_combo.addItem(
                f"Cluster {label} (mean intensity = {mean:.1f})", int(label)
            )
        labels = [int(value) for value in unique]
        selected = (
            labels.index(owner._last_selected_cluster_label)
            if owner._last_selected_cluster_label in labels
            else int(np.argmin(means))
        )
        owner._last_selected_cluster_label = labels[selected]
        owner.cluster_combo.setCurrentIndex(selected)
        owner.cluster_combo.setEnabled(True)
    finally:
        owner.cluster_combo.blockSignals(previous)
    accept_mask(
        owner,
        "prediction_mask",
        (mask == unique[selected]).astype(np.uint8) * 255,
        mask.shape,
    )


@gui_action
def predict_ml(owner, cluster=False):
    frame = frame_for(owner)
    if (
        frame is None
        or owner.jobs.busy
        or not errors.require(
            owner, owner.model is not None, "Обучите или загрузите модель."
        )
    ):
        return
    if cluster and not hasattr(owner.model, "cluster_centers_"):
        return
    model = owner.model
    metadata = dict(owner.model_metadata or {})
    spec = owner._get_current_settings_dict()
    image, gray = frame.image.copy(), frame.gray.copy()
    owner.jobs.start(
        "Применение ML",
        lambda cancel: operations.predict_ml(
            image, gray, model, metadata, spec, cluster, cancel
        ),
        lambda value: (
            cluster_result(owner, value)
            if cluster
            else accept_mask(owner, "prediction_mask", value, gray.shape)
        ),
    )


@gui_action
def train_ml(owner):
    if owner.jobs.busy:
        return
    if not errors.require(
        owner,
        any(image is not None for image in owner.display_images),
        "Загрузите корректные изображения для обучения.",
    ):
        return
    images = [None if image is None else image.copy() for image in owner.display_images]
    grays = [None if gray is None else gray.copy() for gray in owner.gray_images]
    spec = owner._get_current_settings_dict()

    def prepared(values):
        x, y = values
        large = len(x) > 500000 or (
            spec["model_type"] == "MeanShift"
            and not spec["use_superpixels"]
            and len(x) > 50000
        )
        if (
            large
            and QMessageBox.question(
                owner,
                "Большой объём данных",
                f"Обучение на {len(x)} образцах может быть долгим. Продолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            presentation.show_unavailable(owner, ResultStatus.NOT_RUN)
            return

        def trained(value):
            owner.model, owner.model_metadata = value
            owner.model_paths[spec["model_type"]] = ""
            owner._last_selected_cluster_label = None
            owner._update_model_path_display()
            owner._update_apply_button_status()
            owner.check_settings_match_model()
            owner.log(f"Модель {spec['model_type']} обучена на {len(x)} образцах.")
            owner._reset_for_new_image()

        owner.jobs.start(
            "Обучение ML",
            lambda cancel: operations.train_prepared(x, y, spec, cancel),
            trained,
            frame_sensitive=False,
        )

    owner.jobs.start(
        "Подготовка обучающих данных",
        lambda cancel: operations.prepare_training(images, grays, spec, cancel),
        prepared,
        frame_sensitive=False,
    )


@gui_action
def find_k(owner):
    if owner.jobs.busy:
        return
    if not errors.require(
        owner,
        owner.model_combo.currentText() == "K-Means Clustering",
        "Выберите K-Means.",
    ):
        return
    images = [None if image is None else image.copy() for image in owner.display_images]
    spec = owner._get_current_settings_dict()

    def accepted(value):
        owner.kmeans_clusters.setValue(value)
        owner.log(f"Оптимальное K: {value}")
        train_ml(owner)

    owner.jobs.start(
        "Поиск K",
        lambda cancel: operations.optimal_k(images, spec, cancel),
        accepted,
        frame_sensitive=False,
    )


@gui_action
def load_dl(owner, name):
    if owner.jobs.busy:
        return
    widgets = {
        "U-Net": owner.unet_settings,
        "DeepLabV3+": owner.deeplab_settings,
        "DeepLabV3": owner.deeplab_settings,
        "YOLO-seg": owner.yolo_settings,
        "Custom ONNX": owner.custom_settings,
        "SAM": owner.sam_settings,
        "SegFormer": owner.segformer_settings,
    }
    if not errors.require(
        owner,
        name in widgets and name not in ("SAM", "SegFormer"),
        "Выбранная модель ещё не реализована.",
    ):
        return
    path = widgets[name]._model_path.text().strip()
    if not errors.require(
        owner, bool(path) and Path(path).is_file(), "Выберите существующий файл модели."
    ):
        return
    device = owner.device
    owner.segmentor = None
    owner.prediction_mask = None

    def loaded(model):
        owner.segmentor = model
        owner.ignore_ui_changes = True
        try:
            metadata = getattr(model, "metadata", None)
            if metadata and name == "U-Net":
                owner._update_unet_ui_from_metadata(metadata)
            elif metadata and name in ("DeepLabV3", "DeepLabV3+"):
                owner._update_deeplab_ui_from_metadata(metadata)
        finally:
            owner.ignore_ui_changes = False
        owner._raw_signature = None
        owner.result_state = SegmentationResult()
        owner._update_segment_button_color()
        owner.display_current_image()

    owner.jobs.start(
        "Загрузка модели",
        lambda cancel: operations.load_dl(name, path, device, cancel),
        loaded,
        frame_sensitive=False,
        dataset_sensitive=False,
    )


@gui_action
def load_ml(owner, path=None):
    if owner.jobs.busy or getattr(owner, "_loading_model", False):
        return
    if path is None:
        path = QFileDialog.getOpenFileName(
            owner, "Загрузить модель", "", "Model files (*.pkl *.joblib)"
        )[0]
    if not path:
        return
    if not errors.require(owner, Path(path).is_file(), "Файл модели не найден."):
        return
    owner.model = None
    owner.prediction_mask = None

    def loaded(data):
        model, metadata = data
        name = owner._get_model_type_from_class(model)
        if name is None:
            raise ValueError("Тип модели не поддерживается.")
        index = owner.model_combo.findText(name)
        if index < 0:
            raise ValueError("Тип модели отсутствует в интерфейсе.")
        owner._loading_model = True
        owner.ignore_ui_changes = True
        previous = owner.model_combo.blockSignals(True)
        try:
            owner.model_combo.setCurrentIndex(index)
            owner.model = model
            owner.model_metadata = metadata
            owner.model_paths[name] = path
            owner._loaded_model_path = (name, path)
            if metadata:
                owner._apply_model_metadata(metadata)
            owner.on_model_changed(index)
        finally:
            owner.model_combo.blockSignals(previous)
            owner.ignore_ui_changes = False
            owner._loading_model = False
        owner._update_model_path_display()
        owner._update_apply_button_status()
        owner.check_settings_match_model()
        if owner.image_paths:
            owner._reset_for_new_image()
        else:
            owner.display_current_image()

    owner.jobs.start(
        "Загрузка ML",
        lambda cancel: operations.load_ml(path, cancel),
        loaded,
        frame_sensitive=False,
        dataset_sensitive=False,
    )
