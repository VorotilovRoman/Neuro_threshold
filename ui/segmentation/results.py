"""Object inspector / overlay synchronization shared by segmentation tabs."""

from PyQt5.QtCore import QRectF, Qt

from ui.image_conversion import numpy_to_qpixmap
from utils_new.annotations.object_info import ObjectInfo


def current_image(owner):
    image = getattr(owner, "original_image", None)
    if image is None and getattr(owner, "display_images", None):
        image = owner.display_images[owner.current_index]
    return image


def update_object_list(owner):
    owner._focused_object_index = -1
    owner.current_selected_indices = []
    image = current_image(owner)
    if image is None:
        owner.objects_widget.clear()
        return
    h, w = image.shape[:2]
    frame = getattr(owner, "_current_frame", None)
    source_size = frame.source_size if frame is not None else (w, h)
    owner.objects_widget.set_objects(owner.current_objects_full, source_size, (w, h))
    owner.current_selected_indices = owner.objects_widget.checked_indices
    owner.update_annotated_view()


def update_overlay(owner):
    if owner.current_base_image is None:
        return
    image = owner.current_base_image
    h, w = image.shape[:2]
    # Keep pixels free of old OpenCV labels; checkbox changes update vectors only.
    if getattr(owner.annotated_view, "_segmentation_base", None) is not image:
        owner.annotated_view.set_pixmap(numpy_to_qpixmap(image))
        owner.annotated_view._segmentation_base = image
    visible = [
        owner.current_objects_full[i]
        for i in owner.current_selected_indices
        if 0 <= i < len(owner.current_objects_full)
    ]
    owner.annotated_view.set_annotations(visible, w, h)
    owner.annotated_view.set_annotation_labels(
        [
            f"№{i + 1} · класс {owner.current_objects_full[i][1]}"
            for i in owner.current_selected_indices
            if 0 <= i < len(owner.current_objects_full)
        ]
    )
    focused = getattr(owner, "_focused_object_index", -1)
    visible_indices = [
        i
        for i in owner.current_selected_indices
        if 0 <= i < len(owner.current_objects_full)
    ]
    owner._syncing_object_focus = True
    try:
        owner.annotated_view.set_selected_index(
            visible_indices.index(focused) if focused in visible_indices else -1
        )
    finally:
        owner._syncing_object_focus = False
    owner.annotated_view.refresh_annotations()


def on_checks_changed(owner, indices):
    owner.current_selected_indices = sorted(
        i for i in indices if 0 <= i < len(owner.current_objects_full)
    )
    owner.update_annotated_view()


def prepare_frame(owner, image):
    """Scene geometry always uses working pixels, even if QPixmap is capped."""
    h, w = image.shape[:2]
    for name in owner.segmentation_view_names:
        getattr(owner, name).set_image_data(w, h)


def clear_views(owner):
    for name in owner.segmentation_view_names:
        view = getattr(owner, name)
        view.set_pixmap(numpy_to_qpixmap(None))
        view.set_annotations([], 0, 0)
        view._segmentation_base = None


def refresh_styles(owner):
    """Settings only affect Qt paint; preserve pixels, masks, selection and text."""
    for name in owner.segmentation_view_names:
        getattr(owner, name).refresh_annotations()


def focus_row(owner, row):
    owner._focused_object_index = (
        row if 0 <= row < len(owner.current_objects_full) else -1
    )
    update_overlay(owner)


def focus_annotation(owner, visible_index):
    if getattr(owner, "_syncing_object_focus", False):
        return
    indices = [
        i
        for i in owner.current_selected_indices
        if 0 <= i < len(owner.current_objects_full)
    ]
    owner.objects_widget.focus_object(
        indices[visible_index] if 0 <= visible_index < len(indices) else -1, reveal=True
    )


def zoom_object(owner, index):
    image = current_image(owner)
    if image is None or index not in owner.current_selected_indices:
        return
    h, w = image.shape[:2]
    info = ObjectInfo.from_tuple(index, owner.current_objects_full[index])
    x0, y0, x1, y1 = info.geometry((w, h)).bounds
    margin = max(x1 - x0, y1 - y0) * 0.15 + 2
    owner.annotated_view.fitInView(
        QRectF(x0 - margin, y0 - margin, x1 - x0 + 2 * margin, y1 - y0 + 2 * margin),
        Qt.KeepAspectRatio,
    )
