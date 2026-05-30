from import_libs_internal import *
from import_libs_methods_ui import setup_demo_ensemble_methods_ui

import traceback

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
# ------------------------------------------------------------
# Получение версии Ultralytics YOLO (для информационных целей)
# ------------------------------------------------------------
ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    print(f"Ultralytics YOLO version: {version}")
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    print("Не удалось определить версию YOLO (ultralytics не установлена или повреждена)")


class YoloInferThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object, list)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_path, img, conf, iou, imgsz, device, save):
        super().__init__()
        self.model_path = model_path
        self.img = img
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.save = save
        self.cancel_requested = False
        sys.excepthook = lambda exctype, value, tb: self.log(
            f"Unhandled exception: {value}\n{''.join(traceback.format_tb(tb))}")
        sys.excepthook = lambda exctype, value, tb: print_flush(
            f"Unhandled exception: {value}\n{''.join(traceback.format_tb(tb))}")

    def log(self, message):
        self.log_signal.emit(message)

    def stop_inference(self):
        self.cancel_requested = True
        self.log("Получен сигнал остановки...")

    def run(self):
        if not ULTRALYTICS_AVAILABLE:
            self.finished_signal.emit(False, "Библиотека ultralytics не установлена.")
            return

        try:
            self.log("=== Загрузка модели ===")
            model = YOLO(self.model_path)
            self.log(f"Модель загружена: {self.model_path}")
            self.progress_signal.emit(30)

            if self.cancel_requested:
                self.finished_signal.emit(False, "Инференс отменён")
                return

            self.log("=== Выполнение инференса ===")
            results = model(self.img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                            device=self.device, save=self.save)

            result = results[0]
            img_with_boxes = result.plot()

            yolo_lines = []
            if result.boxes is not None:
                h, w = self.img.shape[:2]
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")

            self.result_signal.emit(img_with_boxes, yolo_lines)
            self.progress_signal.emit(100)
            self.log("=== Инференс завершён ===")
            self.finished_signal.emit(True, "Инференс успешно завершён!")

        except Exception as e:
            self.log(f"Ошибка: {e}")
            self.finished_signal.emit(False, str(e))


class EnsembleThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, image, gray, yolo_detections, yolo_weight, presets_with_weights,
                 iou_threshold, conf_threshold, use_containment, method):
        super().__init__()
        self.image = image
        self.gray = gray
        self.yolo_detections = yolo_detections
        self.yolo_weight = yolo_weight
        self.presets_with_weights = presets_with_weights
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.use_containment = use_containment
        self.method = method
        self.cancel_requested = False

    def log(self, message):
        self.log_signal.emit(message)

    def stop(self):
        self.cancel_requested = True
        self.log("Остановка ансамбля...")

    def run(self):
        try:
            self.log("=== Запуск ансамбля ===")
            total_presets = len(self.presets_with_weights)
            detections_list = [self.yolo_detections]
            weights = [self.yolo_weight]

            for idx, (preset, weight, preset_conf) in enumerate(self.presets_with_weights):
                if self.cancel_requested:
                    self.finished_signal.emit(False, "Ансамбль прерван пользователем")
                    return
                self.log(f"Обработка пресета {idx+1}/{total_presets}...")
                preset_dets = get_detections_from_preset(self.image, self.gray, preset, confidence=preset_conf)
                detections_list.append(preset_dets)
                weights.append(weight)
                progress = int(70 * (idx + 1) / total_presets)
                self.progress_signal.emit(progress)
                self.log(f"  найдено {len(preset_dets)} объектов, уверенность {preset_conf:.2f}")

            self.progress_signal.emit(70)
            self.log(f"Объединение детекций (метод: {self.method})...")
            final_detections = ensemble_detections(
                detections_list, weights,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
                use_containment=self.use_containment,
                method=self.method,
                verbose=True
            )
            self.progress_signal.emit(100)
            self.result_signal.emit(final_detections)
            self.log(f"Ансамбль завершён, найдено объектов: {len(final_detections)}")
            self.finished_signal.emit(True, "Ансамбль выполнен успешно")
        except Exception as e:
            self.log(f"Ошибка в ансамбле: {e}")
            self.finished_signal.emit(False, str(e))


class YoloDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Model Verification (Detection)")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(700)

        setup_demo_ensemble_methods_ui(self)

        self.current_folder = None
        self.image_paths = []
        self.images = []
        self.gray_images = []
        self.annotations_list = []
        self.current_index = 0

        self.current_image = None
        self.current_gray = None
        self.current_image_path = None

        self.preset_manager = PresetManager(main_window=None)
        self.active_preset_name = "default"
        self.update_preset_combo()

        self.preset_weight_spins = []
        self.preset_weight_layouts = []

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_preset_to_image)

        self.preset_combo.activated.connect(self.on_preset_activated)
        self.yolo_weight_spin.valueChanged.connect(self.update_total_weight)
        self.nav_widget.load_images.connect(self.load_images)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)

        # Подключаем сигнал изменения пути модели (без дублирования кнопки)
        self.yolo_settings.model_path_changed.connect(self.on_model_path_changed)

        self.run_yolo_btn.clicked.connect(self.on_run_yolo)
        self.run_ensemble_btn.clicked.connect(self.on_run_ensemble)
        self.reset_zoom_all_btn.clicked.connect(self.reset_all_zooms)
        self.on_ensemble_method_changed(self.ensemble_method.currentText())

        # Выбор устройства
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        self.on_device_changed(0)

        self.infer_thread = None
        self.ensemble_thread = None
        self.last_save_folder = None
        self.ensemble_mode = False
        self.ensemble_yolo_detections = []
        self.device = "auto"

        # Флаги для автоматического запуска YOLO
        self.auto_run_yolo = True
        self.yolo_inference_running = False

        self.update_ensemble_weights_ui()

    # ---------- Управление устройством ----------
    def on_device_changed(self, idx):
        device_text = self.device_combo.currentText()
        if device_text.startswith("CPU"):
            self.device = "cpu"
        elif device_text.startswith("CUDA"):
            self.device = "cuda"
        elif "MPS" in device_text:
            self.device = "mps"
        elif device_text.startswith("Auto"):
            self.device = "auto"
        self.log(f"Выбрано устройство: {self.device}")

    def resolve_device(self):
        if self.device == 'auto':
            return '0' if torch.cuda.is_available() else 'cpu'
        elif self.device == 'mps':
            return 'cpu' if not torch.backends.mps.is_available() else 'mps'
        return self.device

    # ---------- Модель YOLO ----------
    def on_model_path_changed(self, path):
        if path:
            self.log(f"Выбрана модель: {path}")
            # При изменении модели автоматически запускаем инференс на текущем изображении, если включено
            if self.auto_run_yolo and self.current_image is not None:
                self._auto_run_yolo_if_model_loaded()

    # ---------- Автоматический запуск YOLO ----------
    def _stop_current_inference(self):
        """Останавливает текущий инференс, если он выполняется."""
        if self.infer_thread and self.infer_thread.isRunning():
            self.log("Останавливаем предыдущий инференс...")
            self.infer_thread.stop_inference()
            self.infer_thread.wait(1000)  # ждём завершения
        self.infer_thread = None
        self.yolo_inference_running = False

    def _auto_run_yolo_if_model_loaded(self):
        print_flush("=== _auto_run_yolo_if_model_loaded START ===")
        if not self.auto_run_yolo:
            print_flush("auto_run_yolo disabled")
            return
        if self.current_image is None:
            print_flush("current_image is None")
            return
        model_path = self.yolo_settings._model_path.text().strip()
        if not model_path:
            print_flush("No model selected")
            return
        if not ULTRALYTICS_AVAILABLE:
            print_flush("Ultralytics not available")
            return
        if self.yolo_inference_running:
            print_flush("Inference already running, stopping previous")
            self._stop_current_inference()
        print_flush("Starting YOLO inference")
        self._run_yolo_internal(manual=False)
        print_flush("=== _auto_run_yolo_if_model_loaded END ===")

    def _run_yolo_internal(self, manual=True):
        print_flush(f"=== _run_yolo_internal START (manual={manual}) ===")
        model_path = self.yolo_settings._model_path.text().strip()
        if not model_path:
            if manual:
                QMessageBox.warning(self, "Ошибка", "Не выбрана модель YOLO.")
            return
        if self.current_image is None:
            if manual:
                QMessageBox.warning(self, "Ошибка", "Нет загруженного изображения.")
            return
        if not ULTRALYTICS_AVAILABLE:
            if manual:
                QMessageBox.critical(self, "Ошибка", "Библиотека ultralytics не установлена.")
            return

        img = self.prepare_image_for_inference(self.current_image)
        conf = self.yolo_settings._conf.value()
        iou = self.yolo_settings._iou.value()
        imgsz = self.yolo_settings._imgsz.value()
        save = self.yolo_settings._save.isChecked()
        device = self.resolve_device()
        print_flush(f"params: conf={conf}, iou={iou}, imgsz={imgsz}, device={device}")

        if manual:
            self.run_yolo_btn.setEnabled(False)
            self.run_ensemble_btn.setEnabled(False)
        self.infer_progress.setValue(0)
        self.infer_progress.setFormat("YOLO: 0%")

        self.yolo_inference_running = True
        self.infer_thread = YoloInferThread(model_path, img, conf, iou, imgsz, device, save)
        self.infer_thread.log_signal.connect(self.log)
        self.infer_thread.progress_signal.connect(self.update_progress)
        self.infer_thread.result_signal.connect(self.display_yolo_result)
        self.infer_thread.result_signal.connect(self.on_yolo_result_for_ensemble)
        self.infer_thread.finished_signal.connect(self.on_inference_finished)
        self.infer_thread.start()
        print_flush("=== _run_yolo_internal END ===")


    def on_run_yolo(self):
        """Ручной запуск YOLO инференса (через кнопку)."""
        self._run_yolo_internal(manual=True)

    def on_inference_finished(self, success, message):
        self.yolo_inference_running = False
        if success:
            self.log("✅ Инференс завершён!")
        else:
            self.log("❌ Инференс прерван")
        self.log(message)

        # Восстанавливаем кнопки только если это был ручной запуск (или если они были заблокированы)
        # Проще всегда восстанавливать, если они отключены – но могут быть отключены из‑за ансамбля.
        # Проверим, не заблокированы ли они ансамблем (ensemble_mode). Если ensemble_mode, то кнопки не восстанавливаем.
        if not self.ensemble_mode:
            self.run_yolo_btn.setEnabled(True)
            self.run_ensemble_btn.setEnabled(True)

        if success and self.ensemble_mode:
            self._run_ensemble()
        else:
            self.ensemble_mode = False

    # ---------- Ансамбль ----------
    def on_run_ensemble(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала загрузите изображение.")
            return
        model_path = self.yolo_settings._model_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Ошибка", "Не выбрана модель YOLO.")
            return
        if not ULTRALYTICS_AVAILABLE:
            QMessageBox.critical(self, "Ошибка", "Библиотека ultralytics не установлена.")
            return

        # Если инференс YOLO уже выполняется, останавливаем его и запускаем заново с новыми параметрами
        if self.yolo_inference_running:
            self._stop_current_inference()

        self.ensemble_mode = True
        self.ensemble_yolo_detections = []
        # Запускаем YOLO инференс, который после завершения автоматически вызовет _run_ensemble
        self._run_yolo_internal(manual=True)

    def _run_ensemble(self):
        presets_with_weights = []
        for name, weight_spin, conf_spin in self.preset_weight_spins:
            preset = self.preset_manager.presets.get(name, {})
            weight = weight_spin.value()
            preset_conf = conf_spin.value()
            if preset and weight > 0:
                presets_with_weights.append((preset, weight, preset_conf))

        yolo_weight = self.yolo_weight_spin.value()
        iou_threshold = self.ensemble_iou.value()
        conf_threshold = self.ensemble_conf.value()
        use_containment = self.ensemble_use_containment.isChecked()
        method_text = self.ensemble_method.currentText()
        method = "wbf" if "WBF" in method_text else "nms"

        self.log(f"YOLO: {len(self.ensemble_yolo_detections)} детекций, вес {yolo_weight:.3f}")
        self.log(f"Пресетов участвует: {len(presets_with_weights)}")
        self.log(f"IoU={iou_threshold}, conf_threshold={conf_threshold}, use_containment={use_containment}, метод={method}")

        self.infer_progress.setValue(0)
        self.infer_progress.setFormat("Ансамбль: 0%")

        self.ensemble_thread = EnsembleThread(
            image=self.current_image,
            gray=self.current_gray,
            yolo_detections=self.ensemble_yolo_detections,
            yolo_weight=yolo_weight,
            presets_with_weights=presets_with_weights,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            use_containment=use_containment,
            method=method
        )
        self.ensemble_thread.log_signal.connect(self.log)
        self.ensemble_thread.progress_signal.connect(self.update_progress)
        self.ensemble_thread.result_signal.connect(self.on_ensemble_result)
        self.ensemble_thread.finished_signal.connect(self.on_ensemble_finished)
        self.ensemble_thread.start()

    def on_yolo_result_for_ensemble(self, img_with_boxes, yolo_lines):
        self.ensemble_yolo_detections = []
        for line in yolo_lines:
            parts = line.split()
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = map(float, parts)
                self.ensemble_yolo_detections.append((int(cls), cx, cy, w, h, conf))
            else:
                cls, cx, cy, w, h = map(float, parts)
                self.ensemble_yolo_detections.append((int(cls), cx, cy, w, h, 1.0))
        self.log(f"YOLO детекции получены (для ансамбля): {len(self.ensemble_yolo_detections)} объектов")

    def on_ensemble_result(self, final_detections):
        self.log(f"Результат ансамбля: {len(final_detections)} объектов")
        if final_detections:
            self.log(f"Пример детекции: {final_detections[0]}")

        img = self.prepare_image_for_inference(self.current_image)
        annotations = []
        lines = []

        for d in final_detections:
            # Пытаемся определить структуру кортежа
            if len(d) == 6:
                # Вариант: (cls, cx, cy, w, h, conf)
                cls, cx, cy, w, h, conf = d
            elif len(d) == 5:
                # Вариант: (cls, cx, cy, w, h) без confidence
                cls, cx, cy, w, h = d
                conf = 1.0
            elif len(d) == 4:
                # Вариант: (cx, cy, w, h) – класс по умолчанию 0
                cx, cy, w, h = d
                cls = 0
                conf = 1.0
            else:
                self.log(f"Неизвестный формат детекции: {d}")
                continue

            # Приводим к унифицированному формату ('detect', cls, cx, cy, w, h)
            annotations.append(('detect', int(cls), cx, cy, w, h))
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} (conf={conf:.3f})")

        # Рисуем аннотации на изображении
        annotated = draw_yolo_annotations(img, annotations, color=(255, 0, 0))
        pixmap = numpy_to_qpixmap(annotated)
        self.ensemble_result_view.set_pixmap(pixmap)

        if lines:
            self.ensemble_analysis_text.setText("\n".join(lines))
        else:
            self.ensemble_analysis_text.setText("Нет объектов")


    def on_ensemble_finished(self, success, message):
        self.run_yolo_btn.setEnabled(True)
        self.run_ensemble_btn.setEnabled(True)
        if success:
            self.log("✅ Ансамбль завершён!")
            self.infer_progress.setFormat("Готово")
        else:
            self.log(f"❌ Ансамбль прерван: {message}")
            QMessageBox.warning(self, "Ошибка ансамбля", message)
        self.ensemble_thread = None
        self.ensemble_mode = False

    # ---------- Навигация с автоматическим запуском YOLO ----------
    def prev_image(self):
        if not self.images:
            return
        self.current_index = (self.current_index - 1) % len(self.images)
        self._update_current_image_and_run()

    def next_image(self):
        if not self.images:
            return
        self.current_index = (self.current_index + 1) % len(self.images)
        self._update_current_image_and_run()

    def goto_image(self, page_num):
        if not self.images:
            return
        total = len(self.images)
        page_num = max(1, min(page_num, total))
        self.current_index = page_num - 1
        self._update_current_image_and_run()

    def _update_current_image_and_run(self):
        print_flush("=== _update_current_image_and_run START ===")
        if not self.images:
            print_flush("No images")
            return
        if self.current_index >= len(self.images):
            print_flush(f"Index {self.current_index} out of range, reset to 0")
            self.current_index = 0
        self.current_image = self.images[self.current_index]
        self.current_gray = self.gray_images[self.current_index]
        self.current_image_path = self.image_paths[self.current_index]
        print_flush(f"Current image: {self.current_image_path}, shape {self.current_image.shape}")
        try:
            self.display_current_groundtruth()
        except Exception as e:
            print_flush(f"Error in display_current_groundtruth: {e}")
            traceback.print_exc()
        try:
            self.schedule_preset_update()
        except Exception as e:
            print_flush(f"Error in schedule_preset_update: {e}")
        try:
            self.update_navigation_state()
        except Exception as e:
            print_flush(f"Error in update_navigation_state: {e}")
        if hasattr(self.nav_widget, 'set_current_index'):
            self.nav_widget.set_current_index(self.current_index, len(self.images))
        print_flush("Calling _auto_run_yolo_if_model_loaded")
        self._auto_run_yolo_if_model_loaded()
        print_flush("=== _update_current_image_and_run END ===")

    def display_current_groundtruth(self):
        print_flush("=== display_current_groundtruth START ===")
        if not self.images or self.current_index >= len(self.images):
            print_flush("No images or index out of range")
            return
        img = self.images[self.current_index].copy()
        ann = self.annotations_list[self.current_index] if self.current_index < len(self.annotations_list) else []
        print_flush(f"Annotations count: {len(ann)}")
        if ann:
            try:
                img = draw_yolo_annotations(img, ann)
                print_flush("  -> draw_yolo_annotations succeeded")
            except Exception as e:
                print_flush(f"  -> Exception in draw_yolo_annotations: {e}")
                traceback.print_exc()
        try:
            pixmap = numpy_to_qpixmap(img)
            if not pixmap.isNull():
                self.preview_view.set_pixmap(pixmap)
                print_flush("  -> pixmap set to preview_view")
            else:
                self.preview_view.set_pixmap(QPixmap())
                print_flush("  -> pixmap is null")
        except Exception as e:
            print_flush(f"  -> Exception setting pixmap: {e}")
            traceback.print_exc()

        # Формируем текстовое представление для всех типов аннотаций
        lines = []
        for a in ann:
            typ = a[0]
            if typ == 'detect':
                _, cls, cx, cy, w, h = a
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            elif typ == 'obb':
                _, cls, points = a  # 8 нормализованных координат
                pts_str = ' '.join(f"{p:.6f}" for p in points)
                lines.append(f"OBB {cls} {pts_str}")
            elif typ == 'segment':
                _, cls, points = a  # список нормализованных координат
                pts_str = ' '.join(f"{p:.6f}" for p in points[:6]) + (' ...' if len(points) > 6 else '')
                lines.append(f"SEG {cls} {pts_str}")
        text = "\n".join(lines) if lines else "Нет истинных аннотаций"
        self.groundtruth_analysis_text.setText(text)

        total = len(self.images)
        if total > 0:
            self.current_file_label.setText(
                f"{self.current_index + 1} / {total}: {os.path.basename(self.image_paths[self.current_index])}")
        print_flush("=== display_current_groundtruth END ===")


    # ---------- Загрузка изображений ----------
    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not paths:
            return
        self.current_folder = None
        self.load_images_from_paths(paths)

    def load_images_from_paths(self, paths):
        print_flush("=== load_images_from_paths START ===")
        print_flush(f"paths count: {len(paths)}")
        total = len(paths)
        progress = QProgressDialog("Loading images...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.image_paths = []
        self.images = []
        self.gray_images = []
        self.annotations_list = []

        for idx, path in enumerate(paths):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"Loading {os.path.basename(path)}...")
            try:
                print_flush(f"Loading {path}")
                img_paths, imgs, grays, anns = load_images_universal(
                    source=[path],
                    require_annotations=False,
                    resize_enabled=self.nav_widget.is_resize_enabled(),
                    parent=self
                )
                if img_paths:
                    self.image_paths.append(img_paths[0])
                    self.images.append(imgs[0])
                    self.gray_images.append(grays[0])
                    ann = anns[0] if anns and anns[0] is not None else []
                    self.annotations_list.append(ann)
                    print_flush(f"  -> Loaded OK, image shape {imgs[0].shape}")
                else:
                    print_flush(f"  -> Failed to load {path}")
                    self.log(f"Не удалось загрузить: {path}")
            except Exception as e:
                print_flush(f"  -> Exception: {e}")
                traceback.print_exc()
                self.log(f"Ошибка загрузки {path}: {e}")
                continue
            QApplication.processEvents()
        progress.close()
        print_flush(f"Loaded {len(self.images)} images")
        if self.images:
            self.current_index = 0
            print_flush("Calling _update_current_image_and_run")
            self._update_current_image_and_run()
        else:
            print_flush("No images loaded, resetting nav")
            if hasattr(self.nav_widget, 'set_current_index'):
                self.nav_widget.set_current_index(0, 0)
        print_flush("=== load_images_from_paths END ===")

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return
        self.current_folder = folder

        def progress_callback(processed, total):
            QApplication.processEvents()
            return True

        try:
            paths, imgs, grays, anns = load_images_universal(
                source=folder,
                require_annotations=False,
                resize_enabled=self.nav_widget.is_resize_enabled(),
                progress_callback=progress_callback,
                parent=self  # <--- добавлено
            )
        except Exception as e:
            self.log(f"Ошибка при загрузке папки: {e}")
            return

        if not paths:
            self.log("В папке нет подходящих изображений.")
            self.nav_widget.set_current_index(0, 0)
            return
        self.image_paths = paths
        self.images = imgs
        self.gray_images = grays
        # Преобразуем аннотации: стандартизируем None -> []
        self.annotations_list = [ann if ann is not None else [] for ann in anns]
        self.current_index = 0
        self._update_current_image_and_run()


    # ---------- Остальные методы (без изменений) ----------
    def update_navigation_state(self):
        has_images = len(self.images) > 0
        self.nav_widget.set_prev_enabled(has_images and self.current_index > 0)
        self.nav_widget.set_next_enabled(has_images and self.current_index < len(self.images) - 1)

    def on_resize_mode_changed(self, state):
        if not self.image_paths and not self.current_folder:
            return
        reply = QMessageBox.question(
            self, "Resize Mode Changed",
            "Resize mode changed. To apply, you need to reload images.\n"
            "Do you want to reload images now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            if self.current_folder:
                self.load_folder()
            elif self.image_paths:
                self.load_images_from_paths(self.image_paths)
            else:
                self.log("Нечего перезагружать.")
        else:
            self.clear_images()

    def clear_images(self):
        self.image_paths = []
        self.images = []
        self.gray_images = []
        self.annotations_list = []
        self.current_index = 0
        self.current_image = None
        self.current_gray = None
        self.current_image_path = None
        self.current_folder = None
        self.preview_view.set_pixmap(QPixmap())
        self.groundtruth_analysis_text.clear()
        self.update_navigation_state()
        self.log("Resize mode changed, images cleared. Please load images again.")
        self.nav_widget.set_current_index(0, 0)
        # Останавливаем текущий инференс
        if self.yolo_inference_running:
            self._stop_current_inference()

    def reset_all_zooms(self):
        """Сбрасывает масштаб во всех четырёх графических представлениях."""
        for view in self.view_widgets.values():
            if hasattr(view, 'reset_view'):
                view.reset_view()
        self.log("Масштаб всех изображений сброшен")

    # ---------- Пресеты и прочее ----------
    def update_ensemble_weights_ui(self):
        for layout in self.preset_weight_layouts:
            self._clear_layout(layout)
            layout.deleteLater()
        self.preset_weight_layouts.clear()
        self.preset_weight_spins.clear()

        preset_names = self.preset_manager.get_preset_names()
        for name in preset_names:
            preset = self.preset_manager.presets.get(name, {})
            preset_conf = preset.get('preset_conf', 0.7)

            row_layout = QHBoxLayout()

            label = QLabel(name)
            label.setMinimumWidth(120)
            row_layout.addWidget(label)

            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.0, 1.0)
            weight_spin.setSingleStep(0.05)
            weight_spin.setValue(0.1)
            weight_spin.valueChanged.connect(self.update_total_weight)
            row_layout.addWidget(weight_spin)

            conf_spin = QDoubleSpinBox()
            conf_spin.setRange(0.0, 1.0)
            conf_spin.setSingleStep(0.05)
            conf_spin.setValue(preset_conf)
            conf_spin.valueChanged.connect(self.update_total_weight)
            row_layout.addWidget(conf_spin)

            row_layout.addStretch()

            self.preset_weights_container.addLayout(row_layout)
            self.preset_weight_layouts.append(row_layout)
            self.preset_weight_spins.append((name, weight_spin, conf_spin))

        if not preset_names:
            empty_label = QLabel("Нет доступных пресетов. Сохраните пресеты в главном окне.")
            empty_label.setStyleSheet("QLabel { color: #888; font-style: italic; }")
            self.preset_weights_container.addWidget(empty_label)
            self.preset_weight_layouts.append(empty_label)

        self.update_total_weight()

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout:
                    self._clear_layout(sub_layout)

    def update_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for name in self.preset_manager.get_preset_names():
            self.preset_combo.addItem(name)
        if self.active_preset_name not in self.preset_manager.presets:
            self.active_preset_name = self.preset_manager.get_preset_names()[0] if self.preset_manager.presets else "default"
        self.preset_combo.setCurrentText(self.active_preset_name)
        self.preset_combo.blockSignals(False)

    def update_presets(self):
        self.preset_manager.load_presets()
        self.update_preset_combo()
        self.update_ensemble_weights_ui()
        self.log("Список пресетов обновлён")

    def on_preset_activated(self, index):
        name = self.preset_combo.itemText(index)
        if name:
            self.active_preset_name = name
            self.log(f"Применён пресет: {name}")
            self._update_preset_info_label(name)
            self.schedule_preset_update()

    def _update_preset_info_label(self, preset_name):
        if hasattr(self, 'preset_info_label'):
            preset = self.preset_manager.presets.get(preset_name, {})
            if 'params' in preset:
                flat = preset.copy()
                flat.update(preset['params'])
                del flat['params']
                preset = flat
            method = preset.get('method', 'Simple Threshold')
            draw_mode = preset.get('draw_mode', 'Contours (simple)')
            info_text = f"Метод: {method} | Отрисовка: {draw_mode}"
            self.preset_info_label.setText(info_text)

    def get_current_preset_settings(self):
        preset = self.preset_manager.presets.get(self.active_preset_name, {})
        if 'params' in preset:
            flat = preset.copy()
            flat.update(preset['params'])
            del flat['params']
            return flat
        return preset

    def schedule_preset_update(self):
        self.update_timer.start(50)

    def apply_preset_to_image(self):
        if self.current_image is None:
            return
        try:
            original = self.current_image
            gray = self.current_gray
            preset = self.get_current_preset_settings()
            method = preset.get('method', 'Simple Threshold')
            params = extract_threshold_params(preset, method)

            # 1. Бинаризация
            binary, thresh = apply_threshold_method(gray, method, params)

            # 2. Принудительная инверсия (как в ThresholdWindow)
            binary = cv2.bitwise_not(binary)

            # 3. Морфология
            close_factor = preset.get('close_factor', 0.0)
            open_factor = preset.get('open_factor', 0.0)
            kernel_shape = preset.get('kernel_shape', 'Rectangle')
            processed = apply_morphology(binary, close_factor, open_factor, kernel_shape, gray.shape)

            # 4. Опциональная инверсия результата (аналог invert_checkbox)
            if preset.get('invert', False):
                processed = cv2.bitwise_not(processed)

            # 5. Получение объектов без рисования (draw=False)
            draw_mode = preset.get('draw_mode', 'Segmentation (Polygon)')
            use_hull = preset.get('use_hull', False)

            if draw_mode == "None":
                objects = []
            elif draw_mode == "Segmentation (Polygon)":
                _, objects = segment_contours(original.copy(), processed, use_hull, draw=False)
            elif draw_mode == "Bounding Box (Detect)":
                _, objects = segment_projections(original.copy(), processed, draw=False)
            elif draw_mode == "OBB (Oriented Box)":
                _, objects = segment_min_area_rect(original.copy(), processed, use_hull, draw=False)
            else:
                objects = []

            self.current_objects = objects
            self.display_preset_result(original, processed)
            self.update_preset_analysis(thresh, len(objects))
        except Exception as e:
            self.log(f"Ошибка при применении пресета: {e}")
            traceback.print_exc()

    def display_preset_result(self, original, binary):
        # Подготовка цветного изображения
        if len(original.shape) == 2:
            result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.shape[2] == 1:
            result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.shape[2] == 4:
            result = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        else:
            result = original.copy()

        # Инвертируем бинарную маску: фон становится белым
        if len(binary.shape) == 3:
            binary = binary[:, :, 0]
        result[binary == 0] = [255, 255, 255]

        if self.current_objects:
            thickness, font_scale, font_thickness, _ = get_display_params(result.shape)
            color_rect = settings.get_color('annotation')
            color_label = settings.get_color('label_text')


            # Выбираем все индексы объектов
            all_indices = list(range(len(self.current_objects)))
            result = draw_selected_objects(
                result, self.current_objects, all_indices,
                draw_mode="",  # не используется, т.к. тип определён в объекте
                use_hull=False,  # не используется
                color_rect=color_rect,
                color_label=color_label,
                thickness=thickness,
                font_scale=font_scale,
                font_thickness=font_thickness
            )

        pixmap = numpy_to_qpixmap(result)
        if not pixmap.isNull():
            self.preset_result_view.set_pixmap(pixmap)





    def showEvent(self, event):
        """Переопределяем показ вкладки, чтобы обновить пресеты."""
        self.update_presets()
        super().showEvent(event)

    def focusInEvent(self, event):
        self.update_presets()
        super().focusInEvent(event)

    def update_preset_analysis(self, threshold, object_count):
        preset = self.get_current_preset_settings()
        method = preset.get('method', 'Simple Threshold')
        draw_mode = preset.get('draw_mode', 'Contours (simple)')

        yolo_lines = []
        if self.current_image is not None:
            img_h, img_w = self.current_image.shape[:2]
            for obj in self.current_objects:
                # Унифицированный формат
                if isinstance(obj, (tuple, list)) and len(obj) >= 2 and obj[0] in ('detect', 'obb', 'segment'):
                    typ = obj[0]
                    if typ == 'detect':
                        _, cls, cx, cy, w, h = obj
                        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    elif typ == 'obb':
                        _, cls, points = obj
                        # Для OBB выводим все 8 координат
                        pts_str = ' '.join(f"{p:.6f}" for p in points)
                        yolo_lines.append(f"{cls} {pts_str} (OBB)")
                    elif typ == 'segment':
                        _, cls, points = obj
                        # Для сегментации выводим все координаты (может быть много, ограничим первыми 6)
                        if len(points) > 6:
                            pts_str = ' '.join(f"{p:.6f}" for p in points[:6]) + ' ...'
                        else:
                            pts_str = ' '.join(f"{p:.6f}" for p in points)
                        yolo_lines.append(f"{cls} {pts_str} (SEG)")
                else:
                    # Старый формат (для обратной совместимости)
                    if len(obj) == 4:
                        x, y, w, h = obj
                        cx_norm = (x + w / 2) / img_w
                        cy_norm = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        yolo_lines.append(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                    # len(obj)==5 для OBB старого формата не обрабатываем, но при желании можно добавить

        analysis_text = ""
        if yolo_lines:
            analysis_text += "\n".join(yolo_lines)
        else:
            analysis_text += "Нет объектов"

        self.preset_analysis_text.setText(analysis_text)


    def display_yolo_result(self, img_with_boxes, yolo_lines):
        print_flush("=== display_yolo_result START ===")
        try:
            pixmap = numpy_to_qpixmap(img_with_boxes)
            if not pixmap.isNull():
                self.model_result_view.set_pixmap(pixmap)
                print_flush("  -> pixmap set to model_result_view")
            else:
                self.model_result_view.set_pixmap(QPixmap())
                print_flush("  -> pixmap is null")
        except Exception as e:
            print_flush(f"  -> Exception setting pixmap: {e}")
            traceback.print_exc()
        lines = []
        for line in yolo_lines:
            parts = line.split()
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = map(float, parts)
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} (conf={conf:.3f})")
            elif len(parts) == 5:
                cls, cx, cy, w, h = map(float, parts)
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        text = "\n".join(lines) if lines else "Нет детекций"
        self.model_analysis_text.setText(text)
        print_flush(f"YOLO detections: {len(lines)}")
        print_flush("=== display_yolo_result END ===")


    def prepare_image_for_inference(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            img = img[:, :, :3]
        return img

    def update_progress(self, percent):
        self.infer_progress.setValue(percent)
        QApplication.processEvents()

    def update_total_weight(self):
        total = self.yolo_weight_spin.value()
        for name, weight_spin, conf_spin in self.preset_weight_spins:
            total += weight_spin.value()
        self.total_weight_label.setText(f"Сумма весов: {total:.2f}")
        if abs(total - 1.0) > 0.01:
            self.total_weight_label.setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; }")
        else:
            self.total_weight_label.setStyleSheet("QLabel { font-weight: bold; color: #27ae60; }")

    def normalize_weights(self):
        total = self.yolo_weight_spin.value()
        for name, weight_spin, conf_spin in self.preset_weight_spins:
            total += weight_spin.value()
        if total > 0:
            self.yolo_weight_spin.setValue(self.yolo_weight_spin.value() / total)
            for name, weight_spin, conf_spin in self.preset_weight_spins:
                weight_spin.setValue(weight_spin.value() / total)
        self.log("Веса нормализованы")

    def log(self, message):
        self.log_widget.log(message)
        QApplication.processEvents()

    def on_stop(self):
        if self.infer_thread and self.infer_thread.isRunning():
            self.log("Останавливаем инференс...")
            self.stop_btn.setEnabled(False)
            self.infer_thread.stop_inference()

    def on_ensemble_method_changed(self, method_text):
        is_wbf = "WBF" in method_text
        self.ensemble_use_containment.setEnabled(is_wbf)
        if not is_wbf and self.ensemble_use_containment.isChecked():
            self.ensemble_use_containment.setChecked(False)
            self.log("Режим NMS не поддерживает учёт вложенности. Флаг сброшен.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YoloDemoWindow()
    window.show()
    sys.exit(app.exec_())