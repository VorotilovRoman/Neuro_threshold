"""Qt adapters for image selection, progress and legacy-shaped window sessions."""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from ui.image_conversion import numpy_to_qpixmap
from utils_new.segmentation.image_session import (
    create_cache,
    image_paths,
    load_one_image_item,
)


def choose_images(owner, load):
    paths, _ = QFileDialog.getOpenFileNames(owner, 'Select Images', '',
                                           'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)')
    if paths:
        owner.log(f'Loading {len(paths)} images...')
        load(paths)


def choose_folder(owner, load):
    path = QFileDialog.getExistingDirectory(owner, 'Select Folder')
    if path:
        owner.log(f'Loading folder: {path}')
        load(path)


def load_cached(owner, source, view_names, *, max_side=1024, reset=None):
    paths = image_paths(source)
    if not paths:
        owner.log('No images loaded.')
        return
    if owner._lazy_cache is not None:
        owner._lazy_cache.shutdown()
    owner._lazy_cache = create_cache(paths, resize_enabled=owner.nav_widget.is_resize_enabled(),
                                     max_side=max_side)
    owner.image_paths = paths
    owner.current_index = 0
    owner.original_image = owner.current_gray = owner.current_base_image = None
    owner.current_objects_full = []
    owner.current_selected_indices = []
    if reset is not None:
        reset()
    for name in view_names:
        view = getattr(owner, name)
        view.set_pixmap(numpy_to_qpixmap(None))
        view.set_annotations([], 0, 0)
    owner.object_list.clear()
    try:
        first = owner._lazy_cache.get(0)
    except (OSError, ValueError, TypeError) as exc:
        owner.log(f'Ошибка чтения изображения: {exc}')
        first = None
    if first is None:
        owner.log('Не удалось загрузить первое изображение.')
        owner._lazy_cache.shutdown()
        owner._lazy_cache = None
        owner.image_paths = []
        owner.nav_widget.set_current_index(0, 0)
        owner.update_navigation_state()
        QMessageBox.warning(owner, 'Ошибка чтения', 'Не удалось загрузить первое изображение.')
        return
    total = len(paths)
    owner._lazy_cache.prefetch(range(1, min(1+owner._lazy_cache.prefetch_ahead+1, total)))
    owner.log(f'Загружено {total} изображений (первое — сразу, остальное в фоне).')
    owner.display_current_image()
    owner.nav_widget.set_current_index(owner.current_index, total)
    owner.update_navigation_state()


def load_images_universal(source, require_annotations=False, resize_enabled=True,
                          max_side=1024, progress_callback=None, parent=None,
                          format_hint='auto'):
    """Temporary eager adapter for ML training; no Qt in the actual reader."""
    paths = image_paths(source)
    progress = None
    if progress_callback is None and parent is not None:
        progress = QProgressDialog('Загрузка изображений...', 'Отмена', 0, len(paths), parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
    output = ([], [], [], [])
    try:
        for index, path in enumerate(paths):
            if progress is not None:
                if progress.wasCanceled():
                    break
                progress.setValue(index)
            elif progress_callback is not None and not progress_callback(index, len(paths)):
                break
            if require_annotations and not Path(path).with_suffix('.txt').exists():
                continue
            frame = load_one_image_item(path, resize_enabled, max_side, format_hint=format_hint)
            if frame is not None:
                for target, value in zip(output, (path, frame['image'], frame['gray'], frame['annotations'])):
                    target.append(value)
    finally:
        if progress is not None:
            progress.close()
    return output


def load_eager(owner, source, *, max_side=640):
    values = load_images_universal(source, resize_enabled=owner.nav_widget.is_resize_enabled(),
                                   max_side=max_side, parent=owner)
    if values[0]:
        owner._update_loaded_data(*values)
        return True
    owner.log('No images loaded.')
    return False


def reload_images(owner, load):
    if not owner.image_paths:
        owner.log('Нет загруженных изображений для перезагрузки.')
        return
    owner.log('Перезагрузка изображений с новыми настройками ресайза...')
    load(list(owner.image_paths))


def clear_session(owner, *, reset=None, empty_fields=(), none_fields=()):
    from ui.segmentation.results import clear_views
    cache = getattr(owner, '_lazy_cache', None)
    if cache is not None:
        cache.shutdown()
        owner._lazy_cache = None
    owner.image_paths = []
    owner.current_index = 0
    for name in ('original_image','current_gray'):
        if hasattr(owner, name):
            setattr(owner, name, None)
    owner.current_objects_full = []
    owner.current_selected_indices = []
    owner.current_base_image = None
    for name in empty_fields:
        setattr(owner, name, [])
    for name in none_fields:
        setattr(owner, name, None)
    if reset is not None:
        reset()
    clear_views(owner)
    owner.info_label.setText('No images')
    owner.object_list.clear()
    owner.coord_text.clear()
    owner.nav_widget.set_current_index(0,0)
    owner.update_navigation_state()
