"""One explicit save flow for all five segmentation windows."""
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from utils_new.dataset_files_new import image_size_from_file
from utils_new.dataset_yolo_labels_new import (
    build_label_save_plan,
    format_label_save_confirmation,
)
from utils_new.segmentation.export import write_annotations_atomic


def save_selected(owner):
    if not owner.image_paths:
        QMessageBox.warning(owner, 'Нет изображения', 'Нет загруженных изображений.')
        return
    objects = owner.current_objects_full
    selected = [objects[i] for i in owner.current_selected_indices if 0 <= i < len(objects)]
    image = owner.image_paths[owner.current_index]
    destination = str(Path(image).with_suffix('.txt'))
    try:
        plan = build_label_save_plan([(image, destination)])
        summary = format_label_save_confirmation(plan) + f'\nВыбрано объектов: {len(selected)} из {len(objects)}'
        if QMessageBox.question(owner, 'Сохранение', summary, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        width, height = image_size_from_file(image)
        write_annotations_atomic(selected, destination, width, height)
    except (OSError, ValueError, TypeError) as exc:
        owner.log(f'Ошибка сохранения {destination}: {exc}')
        QMessageBox.critical(owner, 'Ошибка', str(exc))
        return
    message = f'Сохранено {len(selected)} из {len(objects)} объектов в {destination}'
    if not selected:
        message += '\nПустой TXT — отрицательный пример YOLO.'
    owner.log(message)
    QMessageBox.information(owner, 'Сохранение', message)
