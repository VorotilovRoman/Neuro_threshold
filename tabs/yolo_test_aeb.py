# yolo_test_aeb.py
"""
Вкладка YOLO + AE/VAE.

Раскладка: сверху — YOLO-результат (2-й снимок), снизу слева —
обработанный снимок (3-й), снизу справа — карта ошибок AE/VAE (4-й).
Боксы рисуются только на 2-м снимке и только если включён
соответствующий чекбокс.

Морфологические операции (эрозия/дилатация) применяются внутри маски
выбранного объекта.

Отступ (padding) и смещения X/Y задают окно обрезки вокруг выбранного
объекта при авто-обрезке и авто-довороте. Дополнительно есть точный
доворот кадра в градусах (-90..+90) вокруг центра уже обрезанного
кадра.

Изображения загружаются лениво: в памяти держится окно вокруг текущего
индекса, остальное подгружается/выгружается через LazyImageCache.

Вся «тяжёлая» логика (модели AE/VAE, препроцессинг, работа с масками,
кроп/доворот, ленивый кэш) вынесена в utils_aeb/yolo_aeb_utils.py.
"""
from import_libs_internal import *
from import_libs_methods_ui import setup_yolo_ae_vae_ui


from utils_aeb.yolo_aeb_utils import (
    # модели AE/VAE
    ConvEncoder, ConvDecoder, ConvAutoencoder, ConvVAE,
    UNetEncoder, UNetDecoder, UNetAE, UNetVAE,
    detect_arch, build_ae_model_from_cfg, preprocess_image_for_ae,
    # геометрия
    longest_edge_angle_deg,
    # работа с детекциями
    build_detection_mask, morph_image_in_mask, apply_object_morphology,
    fill_outside_object, crop_to_detection, rotate_with_optional_crop,
    cls_suffix_from_detection, limit_image_size
)

# Защитный лимит для QPixmap/QImage и GDI на Windows.
MAX_SAFE_SIDE = 8192


# ============================================================
# Пресет-менеджер вкладки YOLO + AE/VAE
# ============================================================
class YoloAEBPresetManager(PresetManager):
    def get_current_settings(self):
        m = self.main
        return {
            "yolo_model_path": m.model_edit.text(),
            "ae_model_path": m.ae_model_edit.text(),

            "conf": float(m.conf_spin.value()),
            "iou": float(m.iou_spin.value()),
            "imgsz": int(m.imgsz_spin.value()),
            "device": m.device_combo.currentText(),
            "chk_show_boxes": bool(m.chk_show_boxes.isChecked()),

            "ae_perc": float(m.ae_perc_spin.value()),
            "chk_ae_use_ckpt_thr": bool(m.chk_ae_use_ckpt_thr.isChecked()),
            "chk_ae_overlay": bool(m.chk_ae_overlay.isChecked()),
            "chk_ae_show_mask": bool(m.chk_ae_show_mask.isChecked()),
            "ae_alpha": int(m.ae_alpha_slider.value()),

            "crop_padding": int(m.crop_padding_spin.value()),
            "crop_offset_x": int(m.crop_offset_x_spin.value()),
            "crop_offset_y": int(m.crop_offset_y_spin.value()),
            "crop_rotation": float(m.crop_rotation_spin.value()),
            "chk_auto_crop": bool(m.chk_auto_crop.isChecked()),
            "chk_auto_rotate": bool(m.chk_auto_rotate.isChecked()),
            "fill_mode": m.fill_combo.currentData(),

            "kernel_shape": m.kernel_shape_combo.currentText(),
            "erosion": int(m.erosion_slider.value()),
            "dilation": int(m.dilation_slider.value()),
            "invert_image": bool(m.invert_image_checkbox.isChecked()),
            "contour_only": bool(m.chk_contour_only.isChecked()),
        }

    def apply_preset(self, name):
        if name not in self.presets:
            return
        s = self.presets[name]
        m = self.main

        if "conf" in s:      m.conf_spin.setValue(float(s["conf"]))
        if "iou" in s:       m.iou_spin.setValue(float(s["iou"]))
        if "imgsz" in s:     m.imgsz_spin.setValue(int(s["imgsz"]))
        if "device" in s:
            idx = m.device_combo.findText(str(s["device"]))
            if idx >= 0:
                m.device_combo.setCurrentIndex(idx)
        if "chk_show_boxes" in s:
            m.chk_show_boxes.setChecked(bool(s["chk_show_boxes"]))

        if "ae_perc" in s:   m.ae_perc_spin.setValue(float(s["ae_perc"]))
        if "chk_ae_use_ckpt_thr" in s:
            m.chk_ae_use_ckpt_thr.setChecked(bool(s["chk_ae_use_ckpt_thr"]))
        if "chk_ae_overlay" in s:
            m.chk_ae_overlay.setChecked(bool(s["chk_ae_overlay"]))
        if "chk_ae_show_mask" in s:
            m.chk_ae_show_mask.setChecked(bool(s["chk_ae_show_mask"]))
        if "ae_alpha" in s:
            m.ae_alpha_slider.setValue(int(s["ae_alpha"]))

        if "crop_padding" in s:
            m.crop_padding_spin.setValue(int(s["crop_padding"]))
        if "crop_offset_x" in s:
            m.crop_offset_x_spin.setValue(int(s["crop_offset_x"]))
        if "crop_offset_y" in s:
            m.crop_offset_y_spin.setValue(int(s["crop_offset_y"]))
        if "crop_rotation" in s:
            m.crop_rotation_spin.setValue(float(s["crop_rotation"]))
        if "chk_auto_crop" in s:
            m.chk_auto_crop.setChecked(bool(s["chk_auto_crop"]))
        if "chk_auto_rotate" in s:
            m.chk_auto_rotate.setChecked(bool(s["chk_auto_rotate"]))
        if "fill_mode" in s:
            idx = m.fill_combo.findData(s["fill_mode"])
            if idx >= 0:
                m.fill_combo.setCurrentIndex(idx)

        if "kernel_shape" in s:
            idx = m.kernel_shape_combo.findText(str(s["kernel_shape"]))
            if idx >= 0:
                m.kernel_shape_combo.setCurrentIndex(idx)
        if "erosion" in s:
            m.erosion_slider.setValue(int(s["erosion"]))
        if "dilation" in s:
            m.dilation_slider.setValue(int(s["dilation"]))
        if "invert_image" in s:
            m.invert_image_checkbox.setChecked(bool(s["invert_image"]))
        if "contour_only" in s:
            m.chk_contour_only.setChecked(bool(s["contour_only"]))

        yolo_path = s.get("yolo_model_path", "") or ""
        ae_path = s.get("ae_model_path", "") or ""
        if yolo_path or ae_path:
            m.apply_preset_models(yolo_path, ae_path)


# ============================================================
# Класс окна вкладки
# ============================================================
class YoloInspectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO + AE/VAE")

        # --- Данные изображений ---
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # --- YOLO ---
        self.model = None

        # --- AE/VAE ---
        self.ae_model = None
        self.ae_cfg = None
        self.ae_device = torch.device("cpu")
        self.ae_arch = "conv"
        self._ae_ckpt_threshold = None
        self._ae_ckpt_percentile = None
        self._ae_err_hw = None
        self._ae_valid_region = None
        self._ae_base_bgr = None
        self._ae_threshold = float("nan")
        self._error_bgr = None

        # --- Текущее изображение ---
        self.original_image = None
        self.displayed_image = None
        self.display_mode = "original"
        self.current_detections = []
        self._manual_ops = []
        self._updating_objects_list = False

        # --- UI ---
        setup_yolo_ae_vae_ui(self)

        # --- Таймеры ---
        self._yolo_reinfer_timer = QTimer()
        self._yolo_reinfer_timer.setSingleShot(True)
        self._yolo_reinfer_timer.setInterval(300)
        self._yolo_reinfer_timer.timeout.connect(self._do_reinfer)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update)

        # --- Пресеты ---
        self.preset_manager = YoloAEBPresetManager(self, preset_file="presets_aeb.json")
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

        # --- Навигация ---
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        # --- Модели ---
        self.btn_load_model.clicked.connect(self.load_yolo_model)
        self.btn_load_ae.clicked.connect(self.load_ae_model)
        self.btn_clear_ae.clicked.connect(self.clear_ae_model)
        self.btn_apply_ae.clicked.connect(lambda: self.apply_ae(silent=False))

        # --- Трансформации ---
        self.btn_rot_cw.clicked.connect(lambda: self.manual_transform("rot_cw"))
        self.btn_rot_ccw.clicked.connect(lambda: self.manual_transform("rot_ccw"))
        self.btn_flip_v.clicked.connect(lambda: self.manual_transform("flip_v"))
        self.btn_flip_h.clicked.connect(lambda: self.manual_transform("flip_h"))
        self.btn_reset.clicked.connect(self.reset_view)

        # --- Сохранение ---
        self.btn_save_image.clicked.connect(self.save_image)
        self.btn_save_errmap.clicked.connect(self.save_error_map)

        # --- Авто-трансформации ---
        self.chk_auto_crop.toggled.connect(self.on_auto_crop_toggled)
        self.chk_auto_crop.toggled.connect(self.schedule_update)
        self.chk_auto_rotate.toggled.connect(self.schedule_update)
        self.crop_padding_spin.valueChanged.connect(self.schedule_update)
        self.crop_offset_x_spin.valueChanged.connect(self.schedule_update)
        self.crop_offset_y_spin.valueChanged.connect(self.schedule_update)
        self.crop_rotation_spin.valueChanged.connect(self.schedule_update)
        self.btn_reset_offsets.clicked.connect(self.reset_offsets)
        self.fill_combo.currentIndexChanged.connect(self.schedule_update)

        self.chk_show_boxes.toggled.connect(self.on_show_boxes_changed)

        # --- Морфология по контуру объекта + инверсия ---
        self.kernel_shape_combo.currentIndexChanged.connect(self.schedule_update)
        self.erosion_slider.valueChanged.connect(self.on_erosion_changed)
        self.dilation_slider.valueChanged.connect(self.on_dilation_changed)
        self.invert_image_checkbox.toggled.connect(self.schedule_update)
        self.chk_contour_only.toggled.connect(self.schedule_update)

        self.conf_spin.valueChanged.connect(self.on_yolo_params_changed)
        self.iou_spin.valueChanged.connect(self.on_yolo_params_changed)
        self.imgsz_spin.valueChanged.connect(self.on_yolo_params_changed)
        self.device_combo.currentTextChanged.connect(self.on_yolo_params_changed)

        self.objects_list.currentRowChanged.connect(self.on_object_selection_changed)

        self.chk_ae_overlay.toggled.connect(self.on_ae_render_params_changed)
        self.chk_ae_show_mask.toggled.connect(self.on_ae_render_params_changed)
        self.ae_alpha_slider.valueChanged.connect(self.on_ae_alpha_changed)
        self.chk_ae_use_ckpt_thr.toggled.connect(self.on_ae_threshold_params_changed)
        self.ae_perc_spin.valueChanged.connect(self.on_ae_threshold_params_changed)

        # --- Прочее ---
        self.toggle_log_btn.clicked.connect(self.toggle_log)

        self.update_navigation_state()

    # --------------------------------------------------------
    # Утилиты
    # --------------------------------------------------------
    def log(self, message):
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        print(message)

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def schedule_update(self, *_):
        self.update_timer.start(50)

    def _do_update(self):
        self._rebuild_displayed()
        self._auto_apply_ae()

    def on_auto_crop_toggled(self, checked):
        self.fill_combo.setEnabled(bool(checked))

    def on_show_boxes_changed(self, *_):
        self._rebuild_yolo_view()

    def on_erosion_changed(self, value):
        self.erosion_label.setText(str(int(value)))
        self.schedule_update()

    def on_dilation_changed(self, value):
        self.dilation_label.setText(str(int(value)))
        self.schedule_update()

    # --------------------------------------------------------
    # Устройства
    # --------------------------------------------------------
    def _resolve_device(self):
        d = self.device_combo.currentText()
        if d == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return d

    def _resolve_torch_device(self):
        d = self.device_combo.currentText()
        if d == "auto":
            return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if d == "cpu":
            return torch.device("cpu")
        if d == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            idx = int(d)
            if torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.device(f"cuda:{idx}")
        except ValueError:
            pass
        return torch.device("cpu")

    # --------------------------------------------------------
    # Навигация
    # --------------------------------------------------------
    def update_navigation_state(self):
        total = len(self.image_paths) if self.image_paths else 0
        if total > 0:
            self.nav_widget.set_prev_enabled(self.current_index > 0)
            self.nav_widget.set_next_enabled(self.current_index < total - 1)
            self.nav_widget.set_current_index(self.current_index, total)
        else:
            self.nav_widget.set_prev_enabled(False)
            self.nav_widget.set_next_enabled(False)
            self.nav_widget.set_current_index(0, 0)

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.display_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.display_current_image()

    def goto_image(self, page_num):
        total = len(self.image_paths) if self.image_paths else 0
        if total == 0:
            return
        page_num = max(1, min(page_num, total))
        self.current_index = page_num - 1
        self.display_current_image()

    # --------------------------------------------------------
    # Загрузка изображений
    # --------------------------------------------------------
    def load_images_from_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self._load_images(file_paths)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self._load_images(folder)

    def _load_images(self, source):
        self.log(f"Loading images from {source}...")

        # --- Собираем список файлов БЕЗ загрузки картинок ---
        if isinstance(source, str) and os.path.isdir(source):
            exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
            all_files = sorted(
                os.path.join(source, f)
                for f in os.listdir(source)
                if os.path.splitext(f)[1].lower() in exts
            )
        elif isinstance(source, list):
            all_files = list(source)
        else:
            self.log("No images loaded.")
            return

        if not all_files:
            self.log("No images loaded.")
            return

        resize_enabled = self.nav_widget.is_resize_enabled()
        max_side = 1024

        # --- Останавливаем предыдущий кэш ---
        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None

        # --- Создаём кэш ---
        loader = lambda p: load_one_image_item(
            p,
            resize_enabled=resize_enabled,
            max_side=max_side,
            safe_max_side=MAX_SAFE_SIDE,
        )
        self._lazy_cache = LazyImageCache(
            paths=all_files,
            loader=loader,
            window=8,
            prefetch_ahead=4,
            max_resident=16,
            max_workers=2,
        )

        # --- Сброс состояния ---
        self.image_paths = all_files
        self.current_index = 0
        self.original_image = None
        self.displayed_image = None
        self.current_detections = []
        self._manual_ops = []
        self._clear_ae_view()
        for view in (self.yolo_view, self.transformed_view, self.error_view):
            view.set_pixmap(numpy_to_qpixmap(None))
        self.objects_list.clear()

        # --- Синхронно грузим первый снимок ---
        first = self._lazy_cache.get(0)
        if first is None:
            self.log("Не удалось загрузить первое изображение.")
            return

        # --- Остальное окно — в фон ---
        total = len(self._lazy_cache)
        self._lazy_cache.prefetch(
            range(1, min(1 + self._lazy_cache.prefetch_ahead + 1, total))
        )

        self.log(f"Загружено {total} изображений (первое — сразу, остальное в фоне).")
        self.display_current_image()

    def on_resize_mode_changed(self, enabled):
        if self.image_paths:
            reply = QMessageBox.question(
                self, "Resize Mode Changed",
                "Resize mode changed. Reload images?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._load_images(self.image_paths)
            else:
                self.clear_images()

    def clear_images(self):
        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None
        self.image_paths = []
        self.current_index = 0
        self.original_image = None
        self.displayed_image = None
        self.current_detections = []
        self._manual_ops = []
        self._clear_ae_view()
        for view in (self.yolo_view, self.transformed_view, self.error_view):
            view.set_pixmap(numpy_to_qpixmap(None))
        self.info_label.setText("No images")
        self.objects_list.clear()
        self.update_navigation_state()

    def reset_all_zooms(self):
        for view in (self.yolo_view, self.transformed_view, self.error_view):
            view.reset_view()

    def closeEvent(self, event):
        # Аккуратно останавливаем фоновый пул, чтобы процесс не висел
        # на незавершённых задачах загрузки.
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        super().closeEvent(event)

    # --------------------------------------------------------
    # Основной цикл отображения
    # --------------------------------------------------------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.clear_images()
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        self.info_label.setText(f"Image {idx + 1} of {total}")
        current_file = os.path.basename(self.image_paths[idx])
        self.log(f"Отображён снимок: {current_file}")

        # --- Синхронно получаем элемент (LRU-кэш вернёт готовый или загрузит) ---
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Не удалось загрузить снимок: {self.image_paths[idx]}")
            self.update_navigation_state()
            return

        original = item["image"]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)

        self.original_image = original
        self.current_detections = []
        self._manual_ops = []
        self._clear_ae_view()

        self._update_suggested_names()

        if self.model is not None:
            self._run_yolo_inference()
        else:
            self._rebuild_yolo_view()
            self._update_objects_list()
            self._rebuild_displayed()

        self._auto_apply_ae()
        self.update_navigation_state()

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # --------------------------------------------------------
    # YOLO-инференс
    # --------------------------------------------------------
    def load_yolo_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель YOLO", "",
            "YOLO weights (*.pt *.onnx *.engine);;Все файлы (*)"
        )
        if not path:
            return
        self._load_yolo_model_from_path(path)

    def _load_yolo_model_from_path(self, path):
        if not path or not os.path.isfile(path):
            self.log(f"[YOLO] Файл не найден: {path!r}")
            return False
        try:
            self.model = YOLO(path)
            self.model_edit.setText(path)
            self.log(f"[YOLO] {path}")
            self.log(f"[YOLO классы] {dict(self.model.names)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            return False
        if self.original_image is not None:
            self._run_yolo_inference()
            self._auto_apply_ae()
        return True

    def on_yolo_params_changed(self, *_):
        if self.model is None or self.original_image is None:
            return
        self._yolo_reinfer_timer.start()

    def _do_reinfer(self):
        if self.model is None or self.original_image is None:
            return
        self._run_yolo_inference()
        self._auto_apply_ae()

    def _run_yolo_inference(self):
        if self.model is None or self.original_image is None:
            return

        conf = float(self.conf_spin.value())
        iou = float(self.iou_spin.value())
        imgsz = int(self.imgsz_spin.value())
        device = self._resolve_device()

        QApplication.processEvents()

        try:
            results = self.model(self.original_image,
                                 conf=conf, iou=iou, imgsz=imgsz, device=device)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка инференса", str(e))
            return

        r = results[0]
        names = self.model.names

        self.current_detections = []
        kind = "detect"

        obb = getattr(r, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None and len(obb) > 0:
            kind = "obb"
            corners_all = obb.xyxyxyxy.cpu().numpy()
            cls_all = obb.cls.cpu().numpy().astype(int)
            conf_all = obb.conf.cpu().numpy()
            for i in range(len(cls_all)):
                corners = corners_all[i].tolist()
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cx = sum(xs) / 4.0
                cy = sum(ys) / 4.0
                cls_id = int(cls_all[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                self.current_detections.append({
                    "kind": "obb", "cls_id": cls_id, "cls_name": cls_name,
                    "conf": float(conf_all[i]),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "corners": corners,
                    "angle_deg": longest_edge_angle_deg(corners),
                    "center": (float(cx), float(cy)),
                })
        elif getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None \
                and len(r.masks.xy) > 0:
            kind = "seg"
            boxes = r.boxes
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()
            for i, poly in enumerate(r.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                pts = poly.astype(np.float32).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(pts)
                box_pts = cv2.boxPoints(rect).tolist()
                angle_deg = longest_edge_angle_deg(box_pts)
                cx, cy = float(rect[0][0]), float(rect[0][1])
                cls_id = int(cls_all[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                xyxy = [float(v) for v in xyxy_all[i].tolist()]
                self.current_detections.append({
                    "kind": "seg", "cls_id": cls_id, "cls_name": cls_name,
                    "conf": float(conf_all[i]), "xyxy": xyxy,
                    "corners": box_pts,
                    "mask_xy": poly.tolist(),
                    "angle_deg": angle_deg,
                    "center": (cx, cy),
                })
        else:
            kind = "detect"
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_all = boxes.xyxy.cpu().numpy()
                cls_all = boxes.cls.cpu().numpy().astype(int)
                conf_all = boxes.conf.cpu().numpy()
                for i in range(len(cls_all)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy_all[i].tolist()]
                    w = x2 - x1
                    h = y2 - y1
                    angle_deg = 0.0 if w >= h else 90.0
                    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    cls_id = int(cls_all[i])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    self.current_detections.append({
                        "kind": "detect", "cls_id": cls_id, "cls_name": cls_name,
                        "conf": float(conf_all[i]),
                        "xyxy": [x1, y1, x2, y2], "corners": corners,
                        "angle_deg": angle_deg,
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    })

        self.current_detections.sort(key=lambda d: -d["conf"])

        self._rebuild_yolo_view()
        self._update_objects_list()
        self._rebuild_displayed()

        self.log_text.clear()
        self.log(f"[ИНФЕРЕНС YOLO] conf={conf} iou={iou} imgsz={imgsz} device={device}")
        self.log(f"[ТИП ВЫВОДА] {kind}")
        self.log(f"[ФАЙЛ] {self.image_paths[self.current_index]}")
        self.log("-" * 60)
        for i, d in enumerate(self.current_detections):
            x1, y1, x2, y2 = d["xyxy"]
            ang = d.get("angle_deg")
            ang_s = f" angle={ang:6.1f}°" if ang is not None else ""
            self.log(f"{i+1:3d}. {d['cls_name']}  conf={d['conf']:.3f}  "
                     f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}){ang_s}")
        self.log("-" * 60)
        self.log(f"Всего объектов: {len(self.current_detections)}")

    def _update_objects_list(self):
        self._updating_objects_list = True
        try:
            self.objects_list.clear()
            for i, d in enumerate(self.current_detections):
                x1, y1, x2, y2 = d["xyxy"]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                ang = d.get("angle_deg")
                ang_s = f"  ∠{ang:5.1f}°" if ang is not None else ""
                text = (f"{i+1}. {d['cls_name']}  conf={d['conf']:.3f}  "
                        f"[{int(w)}×{int(h)}]{ang_s}  ({d['kind']})")
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, i)
                self.objects_list.addItem(item)
            if self.current_detections:
                self.objects_list.setCurrentRow(0)
        finally:
            self._updating_objects_list = False
        self._update_suggested_names()

    def _get_selected_detection(self):
        if not self.current_detections:
            return None
        cur = self.objects_list.currentRow()
        if 0 <= cur < len(self.current_detections):
            return self.current_detections[cur]
        return self.current_detections[0]

    def _selected_index_1based(self):
        """1-based позиция выбранного объекта в current_detections (или None)."""
        if not self.current_detections:
            return None
        cur = self.objects_list.currentRow()
        if 0 <= cur < len(self.current_detections):
            return cur + 1
        return 1

    def on_object_selection_changed(self, _row):
        if self._updating_objects_list:
            return
        if self.original_image is None:
            return
        self._update_suggested_names()
        need_update = (
            self.chk_auto_crop.isChecked()
            or self.chk_auto_rotate.isChecked()
            or self.erosion_slider.value() > 1
            or self.dilation_slider.value() > 1
        )
        if not need_update:
            return
        self.schedule_update()

    # --------------------------------------------------------
    # Имена файлов для сохранения (с учётом класса и номера объекта)
    # --------------------------------------------------------
    def _update_suggested_names(self):
        """Обновляет suggested save-name во всех view с учётом класса и номера объекта."""
        if not (self.image_paths and 0 <= self.current_index < len(self.image_paths)):
            return
        current_file = os.path.basename(self.image_paths[self.current_index])
        base_name = os.path.splitext(current_file)[0]
        suffix = cls_suffix_from_detection(
            self._get_selected_detection(),
            self._selected_index_1based(),
        )
        suf = f"_{suffix}" if suffix else ""
        self.yolo_view.set_suggested_save_name(f"aeb_yolo_{base_name}{suf}")
        self.transformed_view.set_suggested_save_name(f"aeb_transformed_{base_name}{suf}")
        self.error_view.set_suggested_save_name(f"aeb_errmap_{base_name}{suf}")

    # --------------------------------------------------------
    # 2-й снимок: YOLO-вид
    # --------------------------------------------------------
    def _detections_to_unified(self, img_w, img_h):
        unified = []
        for d in self.current_detections:
            kind = d["kind"]
            cls_name = d["cls_name"]
            if kind == "detect":
                x1, y1, x2, y2 = d["xyxy"]
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                unified.append(("detect", cls_name, cx, cy, w, h))
            elif kind == "obb":
                pts = []
                for (x, y) in d["corners"]:
                    pts.append(float(x) / img_w)
                    pts.append(float(y) / img_h)
                unified.append(("obb", cls_name, pts))
            elif kind == "seg":
                mask_xy = d.get("mask_xy") or []
                pts = []
                for p in mask_xy:
                    pts.append(float(p[0]) / img_w)
                    pts.append(float(p[1]) / img_h)
                if pts:
                    unified.append(("segment", cls_name, pts))
        return unified

    def _draw_detections_on_image(self, img):
        if not self.current_detections:
            return img

        H, W = img.shape[:2]
        unified = self._detections_to_unified(W, H)
        if not unified:
            return img

        by_mode = {}
        for i, d in enumerate(self.current_detections):
            if d["kind"] == "obb":
                mode = "OBB (Oriented Box)"
            elif d["kind"] == "seg":
                mode = "Segmentation (Polygon)"
            else:
                mode = "Bounding Box (Detect)"
            by_mode.setdefault(mode, []).append(i)

        out = img
        try:
            thickness, font_scale, font_thickness, _ = get_display_params(out.shape)
            color_rect = settings.get_color("annotation")
            color_label = settings.get_color("label_text")
            for mode, idxs in by_mode.items():
                if not idxs:
                    continue
                out = draw_selected_objects(
                    out, unified, idxs, mode, False,
                    color_rect, color_label,
                    thickness, font_scale, font_thickness,
                )
        except Exception as e:
            self.log(f"[YOLO draw] не удалось нарисовать объекты: {e}")
        return out

    def _rebuild_yolo_view(self):
        if self.original_image is None:
            self.yolo_view.set_pixmap(numpy_to_qpixmap(None))
            return
        if not self.chk_show_boxes.isChecked() or not self.current_detections:
            self.yolo_view.set_pixmap(numpy_to_qpixmap(self.original_image))
            return
        annotated = self._draw_detections_on_image(self.original_image.copy())
        self.yolo_view.set_pixmap(numpy_to_qpixmap(annotated))

    # --------------------------------------------------------
    # 3-й снимок: итоговое изображение без боксов
    # --------------------------------------------------------
    def _rebuild_displayed(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        mode = "original"

        det = self._get_selected_detection()
        img, morph_info = apply_object_morphology(
            img,
            det,
            int(self.erosion_slider.value()),
            int(self.dilation_slider.value()),
            shape_name=self.kernel_shape_combo.currentText(),
            contour_only=self.chk_contour_only.isChecked(),
        )
        if morph_info:
            self.log(
                f"[Морфология] kind={morph_info['kind']} "
                f"cls={morph_info['cls_name']} "
                f"mode={morph_info['contour_mode']} "
                f"erosion={morph_info['erosion']}px "
                f"dilation={morph_info['dilation']}px "
                f"mask_px={morph_info['mask_px']}"
            )

        do_rotate = self.chk_auto_rotate.isChecked()
        do_crop = self.chk_auto_crop.isChecked()
        fill_mode = self._get_fill_mode()

        if (do_rotate or do_crop) and det is not None:
            padding = int(self.crop_padding_spin.value())
            off_x = int(self.crop_offset_x_spin.value())
            off_y = int(self.crop_offset_y_spin.value())
            if do_rotate and det.get("angle_deg") is not None:
                img = rotate_with_optional_crop(
                    img, det, auto_crop=do_crop,
                    padding=padding, off_x=off_x, off_y=off_y,
                    fill_mode=fill_mode,
                )
                mode = "rotated"
            elif do_crop:
                img = crop_to_detection(
                    img, det, padding=padding,
                    off_x=off_x, off_y=off_y, fill_mode=fill_mode,
                )
                mode = "cropped"

        # --- Точный доворот кадра вокруг его центра ---
        fine_angle = float(self.crop_rotation_spin.value())
        if abs(fine_angle) > 1e-6:
            h, w = img.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, fine_angle, 1.0)
            if fill_mode == "white":
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
            elif fill_mode == "black":
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
            else:
                # "none" — не заливать фон, растянуть крайние пиксели
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            if mode == "original":
                mode = "fine_rotated"

        for op in self._manual_ops:
            if op == "rot_cw":
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif op == "rot_ccw":
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif op == "flip_v":
                img = cv2.flip(img, 0)
            elif op == "flip_h":
                img = cv2.flip(img, 1)
            mode = "manual"

        if self.invert_image_checkbox.isChecked():
            img = cv2.bitwise_not(img)

        self.displayed_image = img
        self.display_mode = mode
        self.transformed_view.set_pixmap(numpy_to_qpixmap(img))

    def _get_fill_mode(self):
        data = self.fill_combo.currentData()
        if data in ("none", "black", "white"):
            return data
        return "none"

    # --------------------------------------------------------
    # Ручные трансформации / сброс
    # --------------------------------------------------------
    def manual_transform(self, op):
        if self.displayed_image is None:
            return
        if op not in ("rot_cw", "rot_ccw", "flip_v", "flip_h"):
            return
        self._manual_ops.append(op)
        self._rebuild_displayed()
        self._clear_ae_view()
        names = {
            "rot_cw": "Поворот 90° по часовой",
            "rot_ccw": "Поворот 90° против часовой",
            "flip_v": "Отражение по вертикали",
            "flip_h": "Отражение по горизонтали",
        }
        h, w = self.displayed_image.shape[:2]
        self.log(f"{names[op]}: новое изображение {w}×{h} px")
        self._auto_apply_ae()

    def reset_view(self):
        if self.original_image is None:
            return
        self._manual_ops = []
        self._clear_ae_view()
        self._rebuild_displayed()
        self._auto_apply_ae()

    def reset_offsets(self):
        """Сбрасывает отступ, смещения области обрезки и доворот в 0."""
        self.crop_padding_spin.blockSignals(True)
        self.crop_offset_x_spin.blockSignals(True)
        self.crop_offset_y_spin.blockSignals(True)
        self.crop_rotation_spin.blockSignals(True)
        try:
            self.crop_padding_spin.setValue(0)
            self.crop_offset_x_spin.setValue(0)
            self.crop_offset_y_spin.setValue(0)
            self.crop_rotation_spin.setValue(0.0)
        finally:
            self.crop_padding_spin.blockSignals(False)
            self.crop_offset_x_spin.blockSignals(False)
            self.crop_offset_y_spin.blockSignals(False)
            self.crop_rotation_spin.blockSignals(False)
        self.log("[Смещения] сброшены: отступ=0, X=0, Y=0, поворот=0°")
        self.schedule_update()

    # --------------------------------------------------------
    # Сохранение результатов
    # --------------------------------------------------------
    def _default_save_dir_and_base(self):
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            src = self.image_paths[self.current_index]
            base = os.path.splitext(os.path.basename(src))[0]
            start_dir = os.path.dirname(src)
            return start_dir, base
        return "", "image"

    def save_image(self):
        if self.displayed_image is None:
            QMessageBox.information(self, "Нет изображения", "Нечего сохранять.")
            return
        start_dir, base = self._default_save_dir_and_base()
        suffix = cls_suffix_from_detection(
            self._get_selected_detection(),
            self._selected_index_1based(),
        )
        suf = f"_{suffix}" if suffix else ""
        default_name = f"{base}{suf}_transformed.png"
        default_path = os.path.join(start_dir, default_name) if start_dir else default_name

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить снимок", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;"
            "TIFF (*.tif *.tiff);;Все файлы (*)"
        )
        if not save_path:
            return
        try:
            ok = cv2.imwrite(save_path, self.displayed_image)
            if not ok:
                raise RuntimeError("cv2.imwrite вернул False")
            self.log(f"Сохранён снимок: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить снимок:\n{e}")

    def save_error_map(self):
        if self._error_bgr is None:
            QMessageBox.information(
                self, "Нет карты дефектов",
                "Сначала примените модель AE/VAE к снимку."
            )
            return
        start_dir, base = self._default_save_dir_and_base()
        suffix = cls_suffix_from_detection(
            self._get_selected_detection(),
            self._selected_index_1based(),
        )
        suf = f"_{suffix}" if suffix else ""
        default_name = f"{base}{suf}_errmap.png"
        default_path = os.path.join(start_dir, default_name) if start_dir else default_name

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить карту дефектов", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;"
            "TIFF (*.tif *.tiff);;Все файлы (*)"
        )
        if not save_path:
            return
        try:
            ok = cv2.imwrite(save_path, self._error_bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite вернул False")
            self.log(f"Карта дефектов сохранена: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось сохранить карту дефектов:\n{e}")

    # --------------------------------------------------------
    # Загрузка / очистка модели AE/VAE
    # --------------------------------------------------------
    def load_ae_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите чекпойнт AE/VAE", "",
            "PyTorch checkpoint (*.pt *.pth);;Все файлы (*)"
        )
        if not path:
            return
        self._load_ae_model_from_path(path)

    def _load_ae_model_from_path(self, path):
        if not path or not os.path.isfile(path):
            self.log(f"[AE/VAE] Файл не найден: {path!r}")
            return False
        try:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

            if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
                raise RuntimeError("В чекпойнте нет полей 'model' и 'config'.")

            cfg = dict(ckpt["config"])
            arch = detect_arch(cfg)

            required = ["model_type", "in_channels", "img_h", "img_w"]
            required.append("latent_channels" if arch == "unet" else "latent_dim")
            for k in required:
                if k not in cfg:
                    raise RuntimeError(f"В config нет ключа '{k}' (arch={arch}).")

            device = self._resolve_torch_device()
            model = build_ae_model_from_cfg(cfg)
            model.load_state_dict(ckpt["model"])
            model.eval()
            try:
                model.to(device)
            except Exception as e:
                self.log(f"[AE/VAE] Не удалось перевести на {device}: {e}. Использую CPU.")
                device = torch.device("cpu")
                model.to(device)

            self.ae_model = model
            self.ae_cfg = cfg
            self.ae_device = device
            self.ae_arch = arch
            self.ae_model_edit.setText(path)
            self._ae_ckpt_threshold = ckpt.get("threshold_value", None)
            self._ae_ckpt_percentile = ckpt.get("threshold_percentile", None)

            if arch == "unet":
                latent_info = f"latent_channels={cfg['latent_channels']}"
            else:
                latent_info = f"latent_dim={cfg['latent_dim']}"

            self.log(f"[AE/VAE] {path}")
            self.log(f"[AE/VAE] arch={arch}, type={cfg['model_type']}, {latent_info}, "
                     f"size=({cfg['img_h']},{cfg['img_w']}), device={device}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить AE/VAE:\n{e}")
            return False

        if self.displayed_image is not None:
            self._auto_apply_ae()
        return True

    def apply_preset_models(self, yolo_path, ae_path):
        yolo_path = (yolo_path or "").strip()
        ae_path = (ae_path or "").strip()

        if yolo_path and yolo_path != self.model_edit.text():
            if os.path.isfile(yolo_path):
                self._load_yolo_model_from_path(yolo_path)
            else:
                self.log(f"[Пресет] YOLO-модель не найдена: {yolo_path}")

        if ae_path and ae_path != self.ae_model_edit.text():
            if os.path.isfile(ae_path):
                self._load_ae_model_from_path(ae_path)
            else:
                self.log(f"[Пресет] AE/VAE-модель не найдена: {ae_path}")

    def clear_ae_model(self):
        if self.ae_model is None:
            return
        try:
            self.ae_model.to("cpu")
        except Exception:
            pass
        self.ae_model = None
        self.ae_cfg = None
        self.ae_device = torch.device("cpu")
        self.ae_arch = "conv"
        self._ae_ckpt_threshold = None
        self._ae_ckpt_percentile = None
        self.ae_model_edit.clear()
        self._clear_ae_view()
        self.log("[AE/VAE] Очищено")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------
    # Применение AE/VAE
    # --------------------------------------------------------
    def _auto_apply_ae(self):
        if self.ae_model is None or self.displayed_image is None:
            return
        try:
            self.apply_ae(silent=True)
        except Exception as e:
            self.log(f"[AE/VAE auto] ошибка: {e}")

    def _clear_ae_view(self):
        self._error_bgr = None
        self._ae_err_hw = None
        self._ae_valid_region = None
        self._ae_base_bgr = None
        self._ae_threshold = float("nan")
        self.error_view.set_pixmap(numpy_to_qpixmap(None))

    def apply_ae(self, silent=False):
        if self.ae_model is None:
            if not silent:
                QMessageBox.warning(self, "Нет модели AE/VAE",
                                    "Сначала загрузите чекпойнт AE/VAE.")
            return
        if self.displayed_image is None:
            if not silent:
                QMessageBox.warning(self, "Нет изображения",
                                    "Сначала откройте изображение.")
            return

        target = self._resolve_torch_device()
        if target != self.ae_device:
            try:
                self.ae_model.to(target)
                self.ae_device = target
                self.log(f"[AE/VAE] Модель переведена на {target}")
            except Exception as e:
                self.log(f"[AE/VAE] Не удалось перевести на {target}: {e}")

        device = self.ae_device

        try:
            x, valid_region = preprocess_image_for_ae(self.displayed_image, self.ae_cfg)
            x_t = x.unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _ = self.ae_model(x_t)
            recon = recon.cpu()
            x_cpu = x.unsqueeze(0).cpu()
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Ошибка инференса AE/VAE", str(e))
            else:
                self.log(f"[AE/VAE auto] ошибка инференса: {e}")
            return

        err = (x_cpu - recon).abs().mean(dim=1)[0].numpy()

        self._ae_err_hw = err
        self._ae_valid_region = valid_region
        self._ae_base_bgr = self.displayed_image.copy()
        self._ae_threshold = self._compute_threshold()

        self._render_ae_view()

        x1, y1, x2, y2 = valid_region
        roi = err[y1:y2, x1:x2]
        if roi.size > 0 and not silent:
            mean_err = float(roi.mean())
            max_err = float(roi.max())
            thr = self._ae_threshold
            frac_above = float((roi > thr).mean()) * 100 if np.isfinite(thr) else 0.0
            self.log(f"[AE/VAE] arch={self.ae_arch}  тензор входа: {tuple(x.shape)}  "
                     f"valid ROI: x[{x1}:{x2}] y[{y1}:{y2}]  device: {device}")
            self.log(f"[AE/VAE] mean err={mean_err:.6f}, max={max_err:.6f}")
            if np.isfinite(thr):
                src = "ckpt" if (self.chk_ae_use_ckpt_thr.isChecked()
                                 and self._ae_ckpt_threshold is not None) else "percentile"
                self.log(f"[AE/VAE] threshold={thr:.6f} (src={src}); "
                         f"доля пикселей выше порога: {frac_above:.3f}%")

    def _compute_threshold(self):
        if self._ae_err_hw is None or self._ae_valid_region is None:
            return float("nan")
        if (self.chk_ae_use_ckpt_thr.isChecked()
                and self._ae_ckpt_threshold is not None
                and np.isfinite(self._ae_ckpt_threshold)):
            return float(self._ae_ckpt_threshold)
        x1, y1, x2, y2 = self._ae_valid_region
        roi = self._ae_err_hw[y1:y2, x1:x2]
        if roi.size == 0:
            return float("nan")
        return float(np.percentile(roi, float(self.ae_perc_spin.value())))

    def on_ae_threshold_params_changed(self, *_):
        if self._ae_err_hw is None:
            return
        self._ae_threshold = self._compute_threshold()
        self._render_ae_view()

    def on_ae_render_params_changed(self, *_):
        if self._ae_err_hw is None:
            return
        self._render_ae_view()

    def on_ae_alpha_changed(self, v):
        self.ae_alpha_label.setText(str(v))
        self.on_ae_render_params_changed()

    def _render_ae_view(self):
        if self._ae_err_hw is None or self._ae_base_bgr is None:
            return

        err = self._ae_err_hw
        base = self._ae_base_bgr
        H, W = base.shape[:2]

        if self._ae_valid_region is not None:
            x1, y1, x2, y2 = self._ae_valid_region
            err_crop = err[y1:y2, x1:x2]
            if err_crop.size == 0:
                err_crop = err
        else:
            err_crop = err

        err_full = cv2.resize(err_crop, (W, H), interpolation=cv2.INTER_LINEAR)

        lo = float(np.percentile(err_full, 1))
        hi = float(np.percentile(err_full, 99))
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        norm = np.clip((err_full - lo) / (hi - lo), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

        if self.chk_ae_overlay.isChecked():
            alpha_heat = float(self.ae_alpha_slider.value()) / 100.0
            err_img = cv2.addWeighted(base, 1.0 - alpha_heat, heat, alpha_heat, 0.0)
        else:
            err_img = heat

        if self.chk_ae_show_mask.isChecked() and np.isfinite(self._ae_threshold):
            mask_small = (err_crop > self._ae_threshold).astype(np.uint8)
            mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(err_img, contours, -1, (255, 255, 255), 2)

        self._error_bgr = err_img
        self.error_view.set_pixmap(numpy_to_qpixmap(err_img))

    # --------------------------------------------------------
    # Пресеты
    # --------------------------------------------------------
    def update_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_manager.load_presets()
        for name in self.preset_manager.get_preset_names():
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText("default")
        self.preset_combo.blockSignals(False)

    def on_preset_activated(self, index):
        name = self.preset_combo.itemText(index)
        if name:
            self.preset_manager.apply_preset(name)
            self.log(f"Применён пресет: {name}")
            self.schedule_update()

    def save_as_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите имя пресета:")
        if ok and name:
            if name in self.preset_manager.get_preset_names():
                reply = QMessageBox.question(
                    self, "Перезаписать пресет",
                    f"Пресет '{name}' уже существует. Перезаписать?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return
                self.preset_manager.presets[name] = self.preset_manager.get_current_settings()
                self.preset_manager.save_presets()
            else:
                self.preset_manager.add_preset(name)
            self.update_preset_combo()
            self.preset_combo.setCurrentText(name)
            self.log(f"Сохранён пресет: {name}")

    def delete_current_preset(self):
        current = self.preset_combo.currentText()
        reply = QMessageBox.question(
            self, "Удалить пресет",
            f"Удалить пресет '{current}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.preset_manager.delete_preset(current):
                self.update_preset_combo()
                self.log(f"Удалён пресет: {current}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloInspectWindow()
    window.show()
    sys.exit(app.exec_())