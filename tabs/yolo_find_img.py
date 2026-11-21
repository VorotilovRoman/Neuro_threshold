# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# yolo_find_img.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import QDir, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QCursor, QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFileSystemModel,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation.jobs import BackgroundTask
from ui_tabs.yolo_find_img_ui import setup_yolo_find_img_ui
from utils_new.annotation_processing import yolo_result_to_detections
from utils_new.dataset_files_new import (
    load_one_image_item,
    normalize_to_uint8,
    read_image_with_fallback,
    resize_to_max_side,
)


# ------------------------------------------------------------
# ClickableLabel — одиночный/двойной клик + контекстное меню
# ------------------------------------------------------------
class ClickableLabel(QLabel):
    """
    QLabel, который:
      • по одиночному клику — вызывает click callback,
      • по двойному клику — вызывает open callback,
      • по правому клику — вызывает context callback.
    Разделение одиночного и двойного клика — через QTimer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path = ""
        self._open_callback = None
        self._click_callback = None
        self._context_callback = None

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._emit_clicked)

        self._suppress_next_release = False

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._emit_context)

    def set_file_path(self, file_path: str):
        self._file_path = file_path

    def set_open_callback(self, callback):
        self._open_callback = callback

    def set_click_callback(self, callback):
        self._click_callback = callback

    def set_context_callback(self, callback):
        self._context_callback = callback

    def _emit_clicked(self):
        if self._file_path and self._click_callback:
            self._click_callback(self._file_path)

    def _emit_context(self, pos):
        if self._file_path and self._context_callback:
            self._context_callback(self._file_path, self.mapToGlobal(pos))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._file_path:
            if self._suppress_next_release:
                self._suppress_next_release = False
            else:
                self._click_timer.start()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._click_timer.stop()
        self._suppress_next_release = True
        if self._open_callback and self._file_path:
            self._open_callback(self._file_path)
        super().mouseDoubleClickEvent(event)


# ------------------------------------------------------------
# Поток для сканирования
# ------------------------------------------------------------
class ScanThread(BackgroundTask):
    progress = pyqtSignal(int, int)
    log_msg = pyqtSignal(str)
    file_done = pyqtSignal(str, float, bool)
    results_ready = pyqtSignal(list)

    def __init__(
        self,
        root_dir,
        model_path,
        target_class,
        conf,
        iou,
        imgsz,
        device,
        edge_margin=0,
        min_object_size_percent=0.0,
        max_object_size_percent=0.0,
    ):
        """
        edge_margin              — мин. расстояние (px) от бокса до каждого края кадра;
                                   0 — критерий отключён.
        min_object_size_percent  — мин. площадь бокса в % от площади кадра;
                                   0 — критерий отключён.
        max_object_size_percent  — макс. площадь бокса в % от площади кадра;
                                   0 — критерий отключён.
        """
        super().__init__()
        self.root_dir = root_dir
        self.model_path = model_path
        self.target_class = target_class
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.edge_margin = int(edge_margin)
        self.min_object_size_percent = float(min_object_size_percent)
        self.max_object_size_percent = float(max_object_size_percent)
        self.error = None
        self.checked = 0
        self.unreadable = 0
        self.processing_errors = 0

    def cancel(self):
        super().cancel()
        self.log_msg.emit("Отмена сканирования...")

    def _box_meets_margin(self, x1, y1, x2, y2, w, h):
        """True, если бокс отстоит от всех четырёх краёв кадра >= edge_margin px."""
        m = self.edge_margin
        if m <= 0:
            return True
        return (x1 >= m) and (y1 >= m) and (w - x2 >= m) and (h - y2 >= m)

    def _box_meets_min_size(self, x1, y1, x2, y2, w, h):
        """True, если площадь бокса >= min_object_size_percent от площади кадра."""
        m = self.min_object_size_percent
        if m <= 0:
            return True
        img_area = float(w) * float(h)
        if img_area <= 0:
            return True
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh
        percent = 100.0 * box_area / img_area
        return percent >= m

    def _box_meets_max_size(self, x1, y1, x2, y2, w, h):
        """True, если площадь бокса <= max_object_size_percent от площади кадра."""
        m = self.max_object_size_percent
        if m <= 0:
            return True
        img_area = float(w) * float(h)
        if img_area <= 0:
            return True
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh
        percent = 100.0 * box_area / img_area
        return percent <= m

    def run(self):
        results = []
        try:
            if self.cancelled:
                return
            from runtime_policy import configure_ultralytics

            configure_ultralytics()
            from ultralytics import YOLO

            all_images = []
            for root, dirs, files in os.walk(self.root_dir):
                if self.cancelled:
                    break
                dirs.sort()
                all_images.extend(
                    os.path.join(root, file)
                    for file in sorted(files)
                    if file.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
                    )
                )
            total = len(all_images)
            self.publish(
                self.log_msg,
                f"Найдено снимков: {total}. Отступы измеряются на исходном снимке.",
            )
            if self.cancelled or not total:
                return
            model = YOLO(self.model_path)
            for idx, path in enumerate(all_images):
                if self.cancelled:
                    break
                try:
                    # Do not resize: box coordinates/margins must remain in source pixels.
                    item = load_one_image_item(
                        path,
                        resize_enabled=False,
                        load_annotations_flag=False,
                        hard_max_side=None,
                    )
                    if item is None:
                        self.unreadable += 1
                        self.publish(self.log_msg, f"Не читается: {path}")
                        continue
                    image = item["image"]
                    h, w = image.shape[:2]
                    predictions = model(
                        image,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                    if self.cancelled:
                        break
                    detections = (
                        yolo_result_to_detections(
                            predictions[0], getattr(model, "names", {})
                        )
                        if predictions
                        else []
                    )
                    accepted = []
                    for det in detections:
                        if det["cls_id"] != self.target_class:
                            continue
                        x1, y1, x2, y2 = det["xyxy"]
                        if (
                            self._box_meets_margin(x1, y1, x2, y2, w, h)
                            and self._box_meets_min_size(x1, y1, x2, y2, w, h)
                            and self._box_meets_max_size(x1, y1, x2, y2, w, h)
                        ):
                            accepted.append(det["conf"])
                    confidence = max(accepted, default=0.0)
                    if accepted:
                        results.append((path, confidence))
                    self.checked += 1
                    self.publish(self.file_done, path, confidence, bool(accepted))
                except Exception as exc:
                    self.processing_errors += 1
                    self.publish(self.log_msg, f"Ошибка обработки {path}: {exc}")
                finally:
                    self.publish(self.progress, idx + 1, total)
        except Exception as exc:
            self.error = str(exc)
            self.publish(self.log_msg, f"Ошибка запуска поиска: {exc}")
        finally:
            self.publish(self.results_ready, results)


# ------------------------------------------------------------
# Главное окно
# ------------------------------------------------------------
class FindImagesWindow(QMainWindow, setup_yolo_find_img_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self._closing = False
        self.scan_thread = None
        self._class_task = None
        self._thumb_task = None
        self._thumb_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="yolo-thumbnails"
        )
        self.current_root_dir = ""
        self.result_file_paths = []  # список (file_path, max_conf)
        self.thumbnail_widgets = {}
        self.thumbnail_pixmaps = {}

        # Очередь миниатюр: обрабатывается по одному элементу за итерацию
        # цикла событий, чтобы UI оставался отзывчивым во время сканирования.
        self._thumb_queue = []

        self.select_all_checkbox.setTristate(False)

        # Подключение сигналов
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.scan_btn.clicked.connect(self.start_scan)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.browse_target_btn.clicked.connect(self.browse_target_folder)
        self.copy_btn.clicked.connect(self.copy_selected_files)
        self.move_btn.clicked.connect(self.move_selected_files)
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        self.toggle_view_btn.clicked.connect(self.toggle_view_mode)
        self.sort_btn.clicked.connect(self.sort_by_confidence)
        self.results_list.itemDoubleClicked.connect(self.open_file_from_list)
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.thumb_size_combo.currentIndexChanged.connect(self._on_thumb_size_changed)

        # Текущий размер миниатюры (px) — синхронизируется с combo
        self._thumb_size_px = 180

        # Настройка дерева файлов
        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.file_model.setNameFilters(
            ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
        )
        self.file_model.setNameFilterDisables(False)
        self.file_model.setRootPath("")
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(QDir.rootPath()))
        self.file_tree.doubleClicked.connect(self.open_file_from_tree)

        # Подключение виджета настроек YOLO
        self.yolo_settings.model_path_changed.connect(self.on_model_path_changed)
        self.log = self.log_widget.log

        # Добавляем комбобокс устройства (если его нет в UI)
        if hasattr(self, "device_combo"):
            self.device_combo = getattr(self, "device_combo")
        else:
            self.device_combo = QComboBox()
            self.device_combo.addItems(["auto", "0", "cpu", "mps"])
            parent_layout = self.yolo_settings.layout()
            if parent_layout:
                parent_layout.addLayout(
                    self.yolo_settings._row("Устройство", self.device_combo)
                )
            else:
                device_widget = QWidget()
                device_layout = QHBoxLayout(device_widget)
                device_layout.addLayout(QLabel("Устройство:"))
                device_layout.addWidget(self.device_combo)
                self.yolo_settings.layout().addWidget(device_widget)

        # Режим отображения
        self.is_thumbnail_mode = False
        self.stacked_view.setCurrentIndex(0)
        self.toggle_view_btn.setText("Режим: миниатюры")
        self.log_widget.setVisible(False)

        # Загрузка классов текущей модели
        current_model = self.yolo_settings._model_path.text().strip()
        if current_model:
            self.update_model_classes(current_model)

        self.thumbnail_container.resizeEvent = self.on_thumbnail_container_resize

    # --------------------------------------------------------
    # Вспомогательные
    # --------------------------------------------------------

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def _on_thumb_size_changed(self, idx):
        sizes = [120, 180, 260]
        self._thumb_size_px = sizes[max(0, min(idx, len(sizes) - 1))]
        if getattr(self, "is_thumbnail_mode", False) and getattr(
            self, "thumbnail_widgets", None
        ):
            self._rerender_thumbnails()

    def _rerender_thumbnails(self):
        """Меняет масштаб уже загруженных миниатюр без перечитывания файлов."""
        size = self._thumb_size_px
        for fp, (cb, label, container) in self.thumbnail_widgets.items():
            pixmap = self.thumbnail_pixmaps.get(fp)
            if pixmap and not pixmap.isNull():
                label.setPixmap(
                    pixmap.scaled(
                        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            container.setMaximumSize(size + 30, size + 90)
        self.relayout_thumbnails()

    # --------------------------------------------------------
    # Классы модели
    # --------------------------------------------------------
    def update_model_classes(self, model_path):
        if self._class_task:
            self._class_task.close()
        if not model_path or not os.path.isfile(model_path):
            self.class_list_widget.clear()
            return

        def compute(cancel):
            from runtime_policy import configure_ultralytics

            configure_ultralytics()
            from ultralytics import YOLO

            return dict(YOLO(model_path).names)

        worker = BackgroundTask(compute)
        self._class_task = worker
        self.class_list_widget.clear()
        self.class_list_widget.addItem("Чтение классов…")

        def current():
            return not self._closing and self._class_task is worker

        def accept(names):
            if not current():
                return
            self.class_list_widget.clear()
            for index, name in sorted(names.items()):
                self.class_list_widget.addItem(f"{index}: {name}")

        def failed(message):
            if current():
                self.class_list_widget.clear()
                self.class_list_widget.addItem("Классы не загружены")
                self.log(message)

        worker.result_ready.connect(accept)
        worker.failed.connect(failed)
        worker.start()

    def on_model_path_changed(self, model_path):
        self.update_model_classes(model_path)

    # --------------------------------------------------------
    # Выбор папки и дерево
    # --------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите корневую папку для поиска"
        )
        if not folder:
            return
        self.current_root_dir = folder
        self.file_tree.setRootIndex(self.file_model.index(folder))
        self.current_folder_label.setText(folder)
        self.current_folder_label.setStyleSheet(
            "QLabel { color: #444; font-size: 11px; }"
        )
        self.current_folder_label.setToolTip(folder)
        self.log(f"Выбрана папка: {folder}")

    def open_file_from_tree(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            self.open_file(file_path)

    def open_file_from_list(self, item):
        if item is None:
            return
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            self.open_file(file_path)
        else:
            QMessageBox.warning(
                self, "Файл не найден", f"Файл {file_path} не существует."
            )

    def open_file(self, file_path):
        self.log(f"Открытие файла: {file_path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def _open_in_explorer(self, file_path):
        try:
            folder = os.path.dirname(os.path.abspath(file_path))
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception as e:
            self.log(f"Не удалось открыть проводник: {e}")

    # --------------------------------------------------------
    # Сканирование
    # --------------------------------------------------------
    def start_scan(self):
        if self._closing or (self.scan_thread and self.scan_thread.isRunning()):
            return
        if (
            self.min_object_size_spin.value() > 0
            and self.max_object_size_spin.value() > 0
            and self.min_object_size_spin.value() > self.max_object_size_spin.value()
        ):
            QMessageBox.warning(
                self, "Диапазон площади", "Минимальная площадь больше максимальной."
            )
            return
        if not self.current_root_dir:
            QMessageBox.warning(self, "Нет папки", "Сначала выберите папку для поиска.")
            return

        model_path = self.yolo_settings._model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(
                self, "Нет модели", "Укажите существующий файл модели YOLO."
            )
            return

        target_class = self.target_class_spin.value()
        conf = self.yolo_settings._conf.value()
        iou = self.yolo_settings._iou.value()
        imgsz = self.yolo_settings._imgsz.value()
        device = self.device_combo.currentText()
        device = self.resolve_device(device)

        edge_margin = self.edge_margin_spin.value()
        min_object_size = float(self.min_object_size_spin.value())
        max_object_size = float(self.max_object_size_spin.value())

        self.log("=" * 50)
        self.log("ЗАПУСК СКАНИРОВАНИЯ (режим онлайн-миниатюр)")
        self.log(f"Модель: {model_path}")
        self.log(f"Целевая метка: {target_class}")
        self.log(f"Параметры: conf={conf}, iou={iou}, imgsz={imgsz}")
        self.log(f"Используется устройство: {device}")
        self.log(f"Корневая папка: {self.current_root_dir}")
        if edge_margin > 0:
            self.log(f"Мин. расстояние от бокса до края кадра: {edge_margin} px")
        else:
            self.log("Критерий отступа от краёв кадра отключён")
        if min_object_size > 0:
            self.log(
                f"Минимальная площадь объекта: {min_object_size:.2f}% от площади кадра"
            )
        else:
            self.log("Критерий минимальной площади объекта отключён")
        if max_object_size > 0:
            self.log(
                f"Максимальная площадь объекта: {max_object_size:.2f}% от площади кадра"
            )
        else:
            self.log("Критерий максимальной площади объекта отключён")

        self.results_list.clear()
        self._clear_thumbnail_grid()
        self.result_file_paths = []
        self._thumb_queue = []
        self.select_all_checkbox.setChecked(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Сканирование: 0%")

        # Управление кнопками: список/сортировка недоступны во время сканирования,
        # переключение вида — доступно, чтобы можно было смотреть список
        # или миниатюры в реальном времени.
        self.scan_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.toggle_view_btn.setEnabled(True)
        self.sort_btn.setEnabled(False)

        # Автопереключение в режим миниатюр: пользователь сразу видит,
        # как появляются найденные снимки.
        if not self.is_thumbnail_mode:
            self.toggle_view_btn.setChecked(True)
            self.is_thumbnail_mode = True
            self.stacked_view.setCurrentIndex(1)
            self.toggle_view_btn.setText("Режим: список")
            self.log("Автопереключение в режим миниатюр (генерация по мере поиска)")

        self.scan_thread = ScanThread(
            root_dir=self.current_root_dir,
            model_path=model_path,
            target_class=target_class,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            edge_margin=edge_margin,
            min_object_size_percent=min_object_size,
            max_object_size_percent=max_object_size,
        )
        self.scan_thread.progress.connect(self.update_progress)
        self.scan_thread.log_msg.connect(self.log)
        self.scan_thread.file_done.connect(self.on_file_done)
        self.scan_thread.results_ready.connect(self.on_scan_finished)
        self._set_scan_controls(True)
        self.scan_thread.start()

    def stop_scan(self):
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.cancel()
            self.stop_btn.setEnabled(False)
            self.log("Запрошена остановка сканирования...")

    def resolve_device(self, device_str):
        import torch

        if device_str == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        elif device_str == "mps":
            return "cpu" if not torch.backends.mps.is_available() else "mps"
        return device_str

    def update_progress(self, current, total):
        if not self._accept_scan():
            return
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"Сканирование: {percent}% ({current}/{total})")

    # --------------------------------------------------------
    # Поступление файла во время сканирования
    # --------------------------------------------------------
    def on_file_done(self, file_path, max_conf, contains_target):
        if not self._accept_scan():
            return
        if not contains_target:
            return

        # 1) Сразу добавляем в список
        self.result_file_paths.append((file_path, max_conf))
        item = QListWidgetItem(f"{os.path.basename(file_path)} (conf: {max_conf:.3f})")
        item.setData(Qt.UserRole, file_path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        self.results_list.addItem(item)
        self.results_list.scrollToBottom()

        # 2) Ставим миниатюру в очередь. Если очередь была пуста — стартуем
        #    обработку через цикл событий, чтобы UI не подвисал.
        was_empty = len(self._thumb_queue) == 0
        self._thumb_queue.append((file_path, max_conf))
        if was_empty:
            QTimer.singleShot(0, self._process_thumb_queue)

        self.update_select_all_state()

    def _process_thumb_queue(self):
        if self._closing or self._thumb_task is not None or not self._thumb_queue:
            return
        path, confidence = self._thumb_queue.pop(0)
        max_side = max(self._thumb_size_px, 260)

        def compute(cancel):
            image = read_image_with_fallback(path)
            if image is None:
                raise OSError(f"Не читается миниатюра: {path}")
            return resize_to_max_side(normalize_to_uint8(image), max_side)

        worker = BackgroundTask(compute)
        self._thumb_task = worker

        def current():
            return not self._closing and self._thumb_task is worker

        def accept(image):
            if current() and any(file == path for file, _ in self.result_file_paths):
                self.add_thumbnail_widget(path, numpy_to_qpixmap(image), confidence)

        def failed(message):
            if current():
                self.log(message)

        def settled():
            if current():
                self._thumb_task = None
                QTimer.singleShot(0, self._process_thumb_queue)

        worker.result_ready.connect(accept)
        worker.failed.connect(failed)
        worker.settled.connect(settled)
        worker.start(self._thumb_executor)

    def on_scan_finished(self, result_files):
        if not self._accept_scan():
            return
        worker = self.scan_thread
        self._set_scan_controls(False)
        state = (
            "Ошибка" if worker.error else "Отменено" if worker.cancelled else "Готово"
        )
        self.progress_bar.setFormat(f"{state}: найдено {len(result_files)}")
        summary = f"{state}. Найдено: {len(result_files)}; проверено: {worker.checked}; не читается: {worker.unreadable}; ошибок обработки: {worker.processing_errors}."
        if worker.error:
            summary += " " + worker.error
        self.scan_status.setText(summary)
        self.log(summary)
        if self.is_thumbnail_mode and self.thumbnail_widgets:
            self.relayout_thumbnails()

    # --------------------------------------------------------
    # Сортировка
    # --------------------------------------------------------
    def sort_by_confidence(self):
        if not self.result_file_paths:
            return

        selected = set(self.get_selected_files())
        self.result_file_paths.sort(key=lambda x: x[1], reverse=True)

        # Перестроить список в новом порядке
        self.results_list.blockSignals(True)
        self.results_list.clear()
        for file_path, conf in self.result_file_paths:
            item = QListWidgetItem(f"{os.path.basename(file_path)} (conf: {conf:.3f})")
            item.setData(Qt.UserRole, file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if file_path in selected else Qt.Unchecked)
            self.results_list.addItem(item)
        self.results_list.blockSignals(False)

        # Синхронизируем чекбоксы списка с чекбоксами миниатюр
        self.sync_selection_to_thumbnails()

        # Пересобрать сетку миниатюр в новом порядке, используя уже
        # сгенерированные виджеты (файлы миниатюр уже на диске).
        if self.thumbnail_widgets:
            ordered = []
            for fp, conf in self.result_file_paths:
                entry = self.thumbnail_widgets.get(fp)
                if entry is not None:
                    ordered.append((fp, entry))

            # Снять виджеты с сетки, не удаляя их
            for i in reversed(range(self.thumbnail_grid.count())):
                w = self.thumbnail_grid.itemAt(i).widget()
                if w is not None:
                    w.setParent(None)

            size = self._thumb_size_px
            item_w = size + 30
            item_h = size + 90
            width = max(1, self.thumbnail_container.width())
            cols = max(1, width // item_w)

            for idx, (fp, (cb, label, container)) in enumerate(ordered):
                row = idx // cols
                col = idx % cols
                self.thumbnail_grid.addWidget(container, row, col)

            rows = (len(ordered) + cols - 1) // cols if ordered else 0
            self.thumbnail_container.setMinimumHeight(rows * item_h + 20)
            self.thumbnail_grid.update()

        self.log("Список отсортирован по убыванию уверенности.")

    # --------------------------------------------------------
    # Миниатюры (полная генерация — оставлена для совместимости)
    # --------------------------------------------------------

    def add_thumbnail_widget(self, file_path, thumb_path, confidence):
        if file_path in self.thumbnail_widgets:
            return

        pixmap = thumb_path if isinstance(thumb_path, QPixmap) else QPixmap(thumb_path)
        if pixmap.isNull():
            self.log(f"Ошибка загрузки миниатюры для {os.path.basename(file_path)}")
            return

        self.thumbnail_pixmaps[file_path] = pixmap

        size = self._thumb_size_px

        container = QFrame()
        container.setFrameShape(QFrame.Box)
        container.setStyleSheet(
            "QFrame { border: 1px solid #bbb; border-radius: 4px; "
            "background: palette(base); }"
        )
        container.setMaximumSize(size + 30, size + 90)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        cb = QCheckBox()
        cb.setChecked(file_path in self.get_selected_files())
        cb.setFocusPolicy(Qt.NoFocus)
        cb.setToolTip("Отметить файл для копирования/перемещения")
        cb.stateChanged.connect(
            lambda state, fp=file_path: self.on_thumbnail_checkbox_changed(fp, state)
        )
        layout.addWidget(cb, alignment=Qt.AlignTop | Qt.AlignRight)

        label = ClickableLabel()
        label.setPixmap(
            pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label.setAlignment(Qt.AlignCenter)
        label.setToolTip(
            f"{file_path}\nУверенность: {confidence:.3f}\n\n"
            f"Клик — выбрать/снять, двойной клик — открыть, ПКМ — меню."
        )
        label.setCursor(QCursor(Qt.PointingHandCursor))
        label.set_file_path(file_path)
        label.set_open_callback(self.open_file)
        label.set_click_callback(self._toggle_thumbnail_by_path)
        label.set_context_callback(self._show_thumbnail_context_menu)
        layout.addWidget(label, alignment=Qt.AlignCenter)

        name_label = QLabel(os.path.basename(file_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setStyleSheet("QLabel { font-size: 10px; }")
        name_label.setToolTip(file_path)
        layout.addWidget(name_label)

        conf_label = QLabel(f"conf: {confidence:.3f}")
        conf_label.setAlignment(Qt.AlignCenter)
        conf_label.setStyleSheet("QLabel { font-size: 10px; color: #888; }")
        layout.addWidget(conf_label)

        self.thumbnail_widgets[file_path] = (cb, label, container)

        # Подтянуть чекбокс из списка (пользователь мог успеть отметить строку)
        for i in range(self.results_list.count()):
            it = self.results_list.item(i)
            if it.data(Qt.UserRole) == file_path and it.checkState() == Qt.Checked:
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                break

        # Позиция в сетке — по текущей ширине контейнера
        size = self._thumb_size_px
        item_w = size + 30
        width = max(1, self.thumbnail_container.width())
        cols = max(1, width // item_w)
        idx = len(self.thumbnail_widgets) - 1
        row = idx // cols
        col = idx % cols
        self.thumbnail_grid.addWidget(container, row, col)

    def _toggle_thumbnail_by_path(self, file_path):
        """Переключает чекбокс миниатюры и связанной строки в списке."""
        entry = self.thumbnail_widgets.get(file_path)
        if not entry:
            return
        cb, _, _ = entry
        cb.blockSignals(True)
        cb.setChecked(not cb.isChecked())
        cb.blockSignals(False)
        self.on_thumbnail_checkbox_changed(file_path, cb.checkState())

    def _show_thumbnail_context_menu(self, file_path, global_pos):
        menu = QMenu(self)
        act_open = menu.addAction("Открыть файл")
        act_open_ext = menu.addAction("Открыть в проводнике")
        menu.addSeparator()
        act_copy_path = menu.addAction("Скопировать путь")
        menu.addSeparator()
        act_toggle = menu.addAction("Отметить / снять")
        action = menu.exec_(global_pos)
        if action == act_open:
            self.open_file(file_path)
        elif action == act_open_ext:
            self._open_in_explorer(file_path)
        elif action == act_copy_path:
            QApplication.clipboard().setText(file_path)
            self.log(f"Путь скопирован: {file_path}")
        elif action == act_toggle:
            self._toggle_thumbnail_by_path(file_path)

    def _clear_thumbnail_grid(self):
        if self._thumb_task:
            self._thumb_task.close()
        self._thumb_task = None
        for i in reversed(range(self.thumbnail_grid.count())):
            widget = self.thumbnail_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.thumbnail_widgets.clear()
        self.thumbnail_pixmaps.clear()
        # Сбрасываем минимальную высоту — иначе пустой контейнер
        # сохранит большой размер после предыдущего заполнения.
        self.thumbnail_container.setMinimumHeight(0)

    def on_thumbnail_container_resize(self, event):
        if self.thumbnail_widgets:
            self.relayout_thumbnails()
        event.accept()

    def relayout_thumbnails(self):
        if not self.thumbnail_widgets:
            return
        size = self._thumb_size_px
        item_w = size + 30
        item_h = size + 90

        width = max(1, self.thumbnail_container.width())
        cols = max(1, width // item_w)
        rows = (len(self.thumbnail_widgets) + cols - 1) // cols
        total_height = rows * item_h + 20
        self.thumbnail_container.setMinimumHeight(total_height)

        for i in reversed(range(self.thumbnail_grid.count())):
            widget = self.thumbnail_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for idx, (file_path, (cb, label, container)) in enumerate(
            self.thumbnail_widgets.items()
        ):
            row = idx // cols
            col = idx % cols
            self.thumbnail_grid.addWidget(container, row, col)

        self.thumbnail_grid.update()

    # --------------------------------------------------------
    # Переключение режима
    # --------------------------------------------------------
    def toggle_view_mode(self):
        # Переключение доступно и во время сканирования — пользователь
        # может в любой момент посмотреть список или сетку миниатюр.
        self.is_thumbnail_mode = self.toggle_view_btn.isChecked()
        if self.is_thumbnail_mode:
            self.stacked_view.setCurrentIndex(1)
            self.toggle_view_btn.setText("Режим: список")
            self.log("Переключение в режим миниатюр")
            self.sync_selection_to_thumbnails()
            if self.thumbnail_widgets:
                self.relayout_thumbnails()
        else:
            self.stacked_view.setCurrentIndex(0)
            self.toggle_view_btn.setText("Режим: миниатюры")
            self.log("Переключение в режим списка")
            self.sync_selection_to_list()

    def sync_selection_to_thumbnails(self):
        selected_set = set()
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.checkState() == Qt.Checked:
                file_path = item.data(Qt.UserRole)
                selected_set.add(file_path)
        for file_path, (cb, _, _) in self.thumbnail_widgets.items():
            cb.blockSignals(True)
            cb.setChecked(file_path in selected_set)
            cb.blockSignals(False)

    def sync_selection_to_list(self):
        if not self.thumbnail_widgets:
            return
        thumb_checked = {
            fp: cb.isChecked() for fp, (cb, _, _) in self.thumbnail_widgets.items()
        }
        self.results_list.blockSignals(True)
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            fp = item.data(Qt.UserRole)
            if fp in thumb_checked:
                item.setCheckState(Qt.Checked if thumb_checked[fp] else Qt.Unchecked)
        self.results_list.blockSignals(False)
        self.update_select_all_state()

    def on_thumbnail_checkbox_changed(self, file_path, state):
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.data(Qt.UserRole) == file_path:
                item.setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
                break
        self.update_select_all_state()

    def update_select_all_state(self):
        total = self.results_list.count()
        if total == 0:
            self.select_all_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            return

        checked_count = 0
        for i in range(total):
            if self.results_list.item(i).checkState() == Qt.Checked:
                checked_count += 1

        new_checked = checked_count == total
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(new_checked)
        self.select_all_checkbox.blockSignals(False)

    def toggle_select_all(self, state):
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        self.results_list.blockSignals(True)
        for i in range(self.results_list.count()):
            self.results_list.item(i).setCheckState(check_state)
        self.results_list.blockSignals(False)
        for _, (cb, _, _) in self.thumbnail_widgets.items():
            cb.blockSignals(True)
            cb.setChecked(state == Qt.Checked)
            cb.blockSignals(False)

    def get_selected_files(self):
        # The list is the selection source even while thumbnails are still loading.
        return [
            self.results_list.item(i).data(Qt.UserRole)
            for i in range(self.results_list.count())
            if self.results_list.item(i).checkState() == Qt.Checked
        ]

    # --------------------------------------------------------
    # Копирование/перемещение
    # --------------------------------------------------------
    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите целевую папку")
        if folder:
            self.target_folder_edit.setText(folder)
            self.log(f"Выбрана целевая папка: {folder}")

    def copy_selected_files(self):
        self._copy_move_files(copy=True)

    def move_selected_files(self):
        self._copy_move_files(copy=False)

    @staticmethod
    def _unique_dst(dst):
        """
        Возвращает первое свободное имя файла:
            /path/cat.jpg  →  /path/cat.jpg (если свободно)
                            →  /path/cat_1.jpg
                            →  /path/cat_2.jpg ...
        """
        if not os.path.exists(dst):
            return dst
        base, ext = os.path.splitext(dst)
        i = 1
        while True:
            candidate = f"{base}_{i}{ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _copy_move_files(self, copy=True):
        if self.scan_thread and self.scan_thread.isRunning():
            QMessageBox.information(
                self,
                "Сканирование",
                "Завершите или остановите поиск перед копированием/перемещением.",
            )
            return
        target_dir = self.target_folder_edit.text().strip()
        if not target_dir:
            QMessageBox.warning(self, "Нет папки", "Укажите целевую папку.")
            return
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
                self.log(f"Создана целевая папка: {target_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось создать папку: {e}")
                return

        selected = self.get_selected_files()
        if not selected:
            QMessageBox.information(self, "Нет файлов", "Не выбрано ни одного файла.")
            return

        operation = "Копирование" if copy else "Перемещение"
        self.log(f"{operation} {len(selected)} файлов в {target_dir}")

        success = 0
        renamed = 0
        errors = 0
        successfully_processed = []

        for src in selected:
            base_name = os.path.basename(src)
            dst_initial = os.path.join(target_dir, base_name)
            if os.path.realpath(dst_initial) == os.path.realpath(src):
                self.log(f"Пропущен (источник уже в целевой папке): {base_name}")
                continue
            dst = self._unique_dst(dst_initial)

            try:
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
                success += 1
                successfully_processed.append(src)

                if dst != dst_initial:
                    renamed += 1
                    self.log(
                        f"{'Скопирован' if copy else 'Перемещён'} "
                        f"(переименован → {os.path.basename(dst)}): {base_name}"
                    )
                else:
                    self.log(f"{'Скопирован' if copy else 'Перемещён'}: {base_name}")
            except Exception as e:
                self.log(f"Ошибка {base_name}: {e}")
                errors += 1

        msg_lines = [
            f"{operation} завершено.",
            f"Успешно: {success}",
            f"Переименовано (во избежание перезаписи): {renamed}",
            f"Ошибок: {errors}",
        ]
        QMessageBox.information(self, "Результат", "\n".join(msg_lines))
        self.log(
            f"{operation} завершено. Успешно: {success}, "
            f"переименовано: {renamed}, ошибок: {errors}"
        )

        if not copy:
            processed_set = set(successfully_processed)
            self.result_file_paths = [
                (fp, conf)
                for fp, conf in self.result_file_paths
                if fp not in processed_set
            ]

            for i in range(self.results_list.count() - 1, -1, -1):
                item = self.results_list.item(i)
                if item.data(Qt.UserRole) in processed_set:
                    self.results_list.takeItem(i)

            for fp in processed_set:
                if fp in self.thumbnail_widgets:
                    _, _, container = self.thumbnail_widgets[fp]
                    container.deleteLater()
                    del self.thumbnail_widgets[fp]
                    if fp in self.thumbnail_pixmaps:
                        del self.thumbnail_pixmaps[fp]

            if self.is_thumbnail_mode and self.thumbnail_widgets:
                self.relayout_thumbnails()
            self.update_select_all_state()

    # --------------------------------------------------------
    # Завершение
    # --------------------------------------------------------
    def closeEvent(self, event):
        self.shutdown_processing()
        event.accept()

    def _accept_scan(self):
        return not self._closing and (
            self.sender() is None or self.sender() is self.scan_thread
        )

    def _set_scan_controls(self, busy):
        for widget in (
            self.scan_btn,
            self.select_folder_btn,
            self.yolo_settings,
            self.target_class_spin,
            self.edge_margin_spin,
            self.min_object_size_spin,
            self.max_object_size_spin,
            self.copy_btn,
            self.move_btn,
            self.sort_btn,
        ):
            widget.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        if busy:
            self.scan_status.setText(
                "Поиск выполняется. Результаты появляются по мере обработки; их можно просматривать и отмечать."
            )

    def shutdown_processing(self):
        if self._closing:
            return
        self._closing = True
        if self.scan_thread:
            self.scan_thread.close()
        if self._class_task:
            self._class_task.close()
        self._thumb_queue = []
        if self._thumb_task:
            self._thumb_task.close()
        self._thumb_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FindImagesWindow()
    window.show()
    from ui.application import exec_window

    sys.exit(exec_window(window, app))
