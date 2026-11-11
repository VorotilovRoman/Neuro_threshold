"""Selection/list/coordinate/overlay presentation shared by segmentation tabs."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

from ui.image_conversion import numpy_to_qpixmap
from utils_new.annotation_processing import (
    format_annotation_display,
    format_object_for_list,
)


def current_image(owner):
    image = getattr(owner, 'original_image', None)
    if image is None and getattr(owner, 'display_images', None):
        image = owner.display_images[owner.current_index]
    return image


def update_object_list(owner):
    previous = owner.object_list.blockSignals(True)
    try:
        owner.object_list.clear()
        owner.current_selected_indices = []
        image = current_image(owner)
        if image is None:
            return
        h, w = image.shape[:2]
        for index, obj in enumerate(owner.current_objects_full):
            item = QListWidgetItem(format_object_for_list(index, obj, w, h))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            owner.object_list.addItem(item)
            owner.current_selected_indices.append(index)
    finally:
        owner.object_list.blockSignals(previous)
    owner.update_annotated_view()


def update_overlay(owner):
    if owner.current_base_image is None:
        return
    image = owner.current_base_image
    h, w = image.shape[:2]
    # Keep pixels free of old OpenCV labels; checkbox changes update vectors only.
    if getattr(owner.annotated_view, '_segmentation_base', None) is not image:
        owner.annotated_view.set_pixmap(numpy_to_qpixmap(image))
        owner.annotated_view._segmentation_base = image
    visible = [owner.current_objects_full[i] for i in owner.current_selected_indices
               if 0 <= i < len(owner.current_objects_full)]
    owner.annotated_view.set_annotations(visible, w, h)
    owner.annotated_view.refresh_annotations()
    owner.update_coordinates_display()


def update_coordinates(owner):
    owner.coord_text.clear()
    image = current_image(owner)
    if image is None or not owner.current_selected_indices:
        owner.coord_text.append('No objects selected.')
        return
    h, w = image.shape[:2]
    for index in owner.current_selected_indices:
        if 0 <= index < len(owner.current_objects_full):
            owner.coord_text.append(format_annotation_display(index, owner.current_objects_full[index], w, h))


def on_selection_changed(owner, item):
    index = owner.object_list.row(item)
    checked = set(owner.current_selected_indices)
    if item.checkState() == Qt.Checked:
        checked.add(index)
    else:
        checked.discard(index)
    owner.current_selected_indices = sorted(i for i in checked if 0 <= i < len(owner.current_objects_full))
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
