# preparing_dataset_yaml.py
from import_libs_internal import *
from import_libs_methods_ui import setup_preparing_dataset_yaml_ui

from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class DatasetPreparationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Preparation for YOLO")
        self.setMinimumWidth(1100)
        self.setMinimumHeight(750)

        setup_preparing_dataset_yaml_ui(self)

        if ULTRALYTICS_AVAILABLE:
            self.log("Ultralytics YOLO установлен")
        else:
            self.log("Ultralytics YOLO не установлен. Установите: pip install ultralytics")

        self.images_folder = None
        self.labels_folder = None
        self.masks_folder = None
        self.output_folder = None
        self.pairs = []
        self.generator_thread = None
        self.unique_classes = []
        self.original_ids = []
        self.mask_class_remap = {}
        self.included_orig_ids = set()
        self.class_remap = {}
        self.dataset_type = 0
        self.num_classes = 0
        self.annotation_types_stats = {}

        # Статистика по цветности изображений
        self.color_type_stats = {'color': 0, 'gray': 0, 'unknown': 0}

        # Кэш валидации: (img_path, aux_path) -> (is_valid, err, color_type)
        self._validation_cache = {}

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Готово: %p%")

        self.update_split(self.train_slider.value(), 'train')

        # --- Сигналы ---
        self.flip_horizontal.stateChanged.connect(self.update_multiplier_slider_state)
        self.flip_vertical.stateChanged.connect(self.update_multiplier_slider_state)
        self.rotate_90.stateChanged.connect(self.update_multiplier_slider_state)
        self.random_rotate.stateChanged.connect(self.update_multiplier_slider_state)
        self.random_crop.stateChanged.connect(self.update_multiplier_slider_state)
        self.shear.stateChanged.connect(self.update_multiplier_slider_state)
        self.btn_open_output.clicked.connect(self.open_output_folder)
        self.aug_multiplier_slider.valueChanged.connect(self.update_multiplier_label)
        self.aug_multiplier_slider.valueChanged.connect(self.update_split_counts)
        self.train_slider.valueChanged.connect(self.update_split_counts)
        self.val_slider.valueChanged.connect(self.update_split_counts)
        self.test_slider.valueChanged.connect(self.update_split_counts)

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.dataset_type_combo.currentIndexChanged.connect(self._on_dataset_type_changed)
        self.class_table.itemChanged.connect(self.on_class_table_item_changed)

        self.include_empty_annotations.stateChanged.connect(self._on_include_empty_toggled)

        # --- Двойной клик по паре → снимок валидации ---
        self.file_list.itemDoubleClicked.connect(self._on_file_list_double_clicked)

        # --- Ctrl+C: копировать выделенные строки ---
        self._copy_shortcut = QShortcut(QKeySequence.Copy, self.file_list)
        self._copy_shortcut.setContext(Qt.WidgetShortcut)
        self._copy_shortcut.activated.connect(self._copy_selected_rows_to_clipboard)

        # --- Кнопки конвертации цветности (создаются в UI) ---
        if hasattr(self, 'btn_convert_gray'):
            self.btn_convert_gray.clicked.connect(lambda: self.convert_all_images('gray'))
        if hasattr(self, 'btn_convert_rgb'):
            self.btn_convert_rgb.clicked.connect(lambda: self.convert_all_images('color'))

        self.update_multiplier_slider_state()

    # ==================================================================
    # Определение цветности
    # ==================================================================
    def _detect_image_color_type(self, img):
        """
        Возвращает 'color', 'gray' или 'unknown'.
        Проверяем не только число каналов, но и фактическую цветность:
        многие PNG/JPG с 3 каналами могут быть полностью серыми.
        """
        if img is None:
            return 'unknown'
        try:
            if len(img.shape) == 2:
                return 'gray'
            if img.shape[2] == 1:
                return 'gray'
            bgr = img[:, :, :3] if img.shape[2] >= 3 else img
            b = bgr[:, :, 0]
            g = bgr[:, :, 1]
            r = bgr[:, :, 2]
            if np.array_equal(b, g) and np.array_equal(g, r):
                return 'gray'
            return 'color'
        except Exception:
            return 'unknown'

    # ==================================================================
    # Утилиты списка пар
    # ==================================================================
    def _copy_selected_rows_to_clipboard(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        text = "\n".join(it.text() for it in items)
        QApplication.clipboard().setText(text)
        self.log(f"Скопировано строк: {len(items)}")

    def _on_file_list_double_clicked(self, item):
        if item is None:
            return
        pair = item.data(Qt.UserRole + 1)
        if not pair:
            return
        self._open_pair_validation(pair)

    # ==================================================================
    # Валидация пары
    # ==================================================================
    def _validate_pair_full(self, pair):
        """
        Полная валидация: (is_valid, error_message, color_type). Кэшируется.
        """
        key = (pair[0], pair[1] if len(pair) > 1 else None)
        if key in self._validation_cache:
            return self._validation_cache[key]

        result = self._validate_pair_impl(pair)
        self._validation_cache[key] = result
        return result

    def _validate_pair(self, pair):
        """
        Совместимость: возвращает только (is_valid, error_message).
        """
        ok, err, _ = self._validate_pair_full(pair)
        return ok, err

    def _validate_pair_impl(self, pair):
        img_path = pair[0]
        aux_path = pair[1] if len(pair) > 1 else None

        # --- 1. Изображение ---
        if not os.path.exists(img_path):
            return False, "Image file not found", 'unknown'
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False, "Cannot decode image", 'unknown'
            img_h, img_w = img.shape[:2]
            if img_h == 0 or img_w == 0:
                return False, "Zero-size image", 'unknown'
            color_type = self._detect_image_color_type(img)
        except Exception as e:
            return False, f"Image read error: {e}", 'unknown'

        # --- 2. Только изображения (type 2) ---
        if aux_path is None:
            return True, "", color_type

        # --- 3. Маски (type 1) ---
        if self.dataset_type == 1:
            if not os.path.exists(aux_path):
                return False, "Mask file not found", color_type
            try:
                mask = cv2.imread(str(aux_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    return False, "Cannot decode mask", color_type
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                mask = mask.squeeze()
                if mask.dtype != np.uint8:
                    return False, f"Mask dtype {mask.dtype} (expected uint8)", color_type
                if mask.shape[:2] != (img_h, img_w):
                    return False, (f"Mask size {mask.shape[1]}x{mask.shape[0]} != "
                                   f"image {img_w}x{img_h}"), color_type
            except Exception as e:
                return False, f"Mask read error: {e}", color_type
            return True, "", color_type

        # --- 4. Метки (type 0) ---
        if not os.path.exists(aux_path):
            return False, "Label file not found", color_type
        try:
            with open(aux_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
        except Exception as e:
            return False, f"Label read error: {e}", color_type

        lines = []
        for ln, raw in enumerate(raw_lines, 1):
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            lines.append((ln, s))

        if not lines:
            return True, "", color_type  # пустая аннотация — не ошибка формата

        types_in_file = set()
        for ln, line in lines:
            parts = line.split()
            if len(parts) < 5:
                return False, f"Line {ln}: too few values ({len(parts)})", color_type

            try:
                cls = int(parts[0])
                if cls < 0:
                    return False, f"Line {ln}: negative class id {cls}", color_type
            except ValueError:
                return False, f"Line {ln}: invalid class id '{parts[0]}'", color_type

            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError as e:
                return False, f"Line {ln}: invalid coordinate ({e})", color_type

            def in_range(v):
                return 0.0 <= v <= 1.0

            n = len(parts)
            if n == 5:  # detect: cx cy w h
                types_in_file.add('detect')
                cx, cy, bw, bh = coords
                if not in_range(cx) or not in_range(cy):
                    return False, f"Line {ln}: center out of [0,1]", color_type
                if bw <= 0 or bh <= 0:
                    return False, f"Line {ln}: non-positive bbox size", color_type
                if bw > 1 or bh > 1:
                    return False, f"Line {ln}: bbox size > 1", color_type
            elif n == 9:  # obb: 4 точки
                types_in_file.add('obb')
                for i in range(0, 8, 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return False, f"Line {ln}: OBB point {i // 2 + 1} out of [0,1]", color_type
            elif n >= 7 and (n - 1) % 2 == 0:  # segment
                types_in_file.add('segment')
                num_pts = (n - 1) // 2
                if num_pts < 3:
                    return False, f"Line {ln}: polygon needs >=3 points, got {num_pts}", color_type
                for i in range(0, len(coords), 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return False, f"Line {ln}: polygon point out of [0,1]", color_type
            else:
                return False, f"Line {ln}: unknown format ({n} values)", color_type

        if len(types_in_file) > 1:
            return False, f"Mixed types in file: {', '.join(sorted(types_in_file))}", color_type

        return True, "", color_type

    def _validate_all_pairs(self):
        if not self.pairs:
            return 0, 0

        self._validation_cache.clear()
        total = len(self.pairs)
        valid = 0
        invalid = 0

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Проверка пар: 0/{total}...")
        QApplication.processEvents()

        step = max(1, total // 100)

        for idx, pair in enumerate(self.pairs):
            ok, _ = self._validate_pair(pair)
            if ok:
                valid += 1
            else:
                invalid += 1

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Проверка пар: {idx + 1}/{total} ({percent}%)"
                )
                QApplication.processEvents()

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        QApplication.processEvents()

        return valid, invalid

    # ==================================================================
    # Статистика по цветности
    # ==================================================================
    def _recompute_color_stats(self):
        self.color_type_stats = {'color': 0, 'gray': 0, 'unknown': 0}
        for pair in self.pairs:
            _, _, ctype = self._validate_pair_full(pair)
            self.color_type_stats[ctype] = self.color_type_stats.get(ctype, 0) + 1

    def _update_color_stats_label(self):
        if not hasattr(self, 'color_stats_label'):
            return
        c = self.color_type_stats.get('color', 0)
        g = self.color_type_stats.get('gray', 0)
        u = self.color_type_stats.get('unknown', 0)

        if not self.pairs:
            self.color_stats_label.setText("")
            if hasattr(self, 'btn_convert_gray'):
                self.btn_convert_gray.setEnabled(False)
            if hasattr(self, 'btn_convert_rgb'):
                self.btn_convert_rgb.setEnabled(False)
            return

        text = f"🎨 Цветных: {c}   |   ⬛ Ч/Б: {g}"
        if u:
            text += f"   |   ❓ Неизвестно: {u}"
        self.color_stats_label.setText(text)

        mixed = c > 0 and g > 0
        if hasattr(self, 'btn_convert_gray'):
            self.btn_convert_gray.setEnabled(mixed)
        if hasattr(self, 'btn_convert_rgb'):
            self.btn_convert_rgb.setEnabled(mixed)

    # ==================================================================
    # Массовая конвертация цветности
    # ==================================================================
    def convert_all_images(self, target='gray'):
        """
        target: 'gray' — перезаписать только цветные изображения в ЧБ;
                'color' — перезаписать только ЧБ изображения в RGB (3 канала).
        Снимки, которые уже соответствуют целевому типу, НЕ трогаем.
        Работает in-place по путям из self.pairs, с подтверждением.
        """
        if not self.pairs:
            return

        # Определяем, какие файлы нужно конвертировать
        to_convert = []  # список (pair, img_path, current_type)
        for pair in self.pairs:
            img_path = pair[0]
            _, _, ctype = self._validate_pair_full(pair)
            if target == 'gray' and ctype == 'color':
                to_convert.append((pair, img_path, ctype))
            elif target == 'color' and ctype == 'gray':
                to_convert.append((pair, img_path, ctype))

        if not to_convert:
            QMessageBox.information(
                self, "Конвертация не требуется",
                "Все изображения уже соответствуют выбранному типу."
            )
            return

        target_label = "ЧБ" if target == 'gray' else "RGB"
        reply = QMessageBox.question(
            self, "Конвертация изображений",
            f"Будет перезаписано {len(to_convert)} из {len(self.pairs)} изображений "
            f"(в режим {target_label}).\n"
            f"Остальные уже соответствуют целевому типу и останутся без изменений.\n\n"
            f"Действие необратимо. Продолжить?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        converted = 0
        errors = 0
        total = len(to_convert)
        step = max(1, total // 100)

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Конвертация: 0/{total}...")
        QApplication.processEvents()

        for idx, (pair, img_path, _) in enumerate(to_convert):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    errors += 1
                    continue

                # Приводим к 3 каналам для дальнейших преобразований
                if len(img.shape) == 2:
                    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img3 = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img3 = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img3 = img

                if target == 'gray':
                    out = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                else:
                    out = img3

                cv2.imwrite(str(img_path), out)
                converted += 1
            except Exception as e:
                errors += 1
                self.log(f"Ошибка конвертации {img_path}: {e}")

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Конвертация: {idx + 1}/{total} ({percent}%)"
                )
                QApplication.processEvents()

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        self.log(f"Конвертация завершена: перезаписано {converted}, ошибок {errors} "
                 f"(пропущено без изменений: {len(self.pairs) - total})")

        # Сброс кэша и пересканирование
        self._validation_cache.clear()
        self.scan_pairs()

    # ==================================================================
    # Список пар: подсветка и предупреждения
    # ==================================================================
    def _make_list_item(self, text, tag=None, error_msg=""):
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()

        if tag == 'invalid':
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Не корректная пара: {error_msg}\n\n"
                            f"Двойной клик — всё равно открыть снимок.")
        elif tag == 'empty':
            if include_empty:
                item.setForeground(QColor(0, 140, 0))
                item.setToolTip("Пустая аннотация — будет включена как негативный пример.")
            else:
                item.setForeground(QColor(200, 120, 0))
                item.setToolTip("Пустая аннотация — будет пропущена. "
                                "Включите «Include images with empty annotations», чтобы сохранить.")
        elif tag == 'error':
            item.setForeground(QColor(200, 0, 0))
            item.setToolTip("Ошибка чтения файла аннотации.")
        elif tag == 'mixed':
            item.setForeground(QColor(160, 0, 200))
            item.setToolTip("Смешанные типы аннотаций в одном файле.")
        elif tag == 'unknown':
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _rebuild_file_list_display(self):
        if not self.pairs:
            self.validation_warning_label.setVisible(False)
            self.validation_progress_label.setVisible(False)
            self._update_color_stats_label()
            return

        self.file_list.blockSignals(True)
        self.file_list.clear()

        valid_count = 0
        invalid_count = 0
        invalid_names = []

        total = len(self.pairs)
        step = max(1, total // 100)

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Построение списка: 0/{total}...")
        QApplication.processEvents()

        for idx, pair in enumerate(self.pairs):
            img_path = pair[0]
            aux_path = pair[1] if len(pair) > 1 else None
            name = os.path.splitext(os.path.basename(img_path))[0]

            is_valid, err, color_type = self._validate_pair_full(pair)

            if self.dataset_type == 0 and aux_path:
                try:
                    ann_type = self._detect_annotation_type(aux_path)
                except Exception:
                    ann_type = 'error'
            elif self.dataset_type == 1:
                ann_type = 'segment'
            else:
                ann_type = None

            color_marker = ""
            if color_type == 'gray':
                color_marker = " [ч/б]"
            elif color_type == 'color':
                color_marker = " [цветное]"

            if not is_valid:
                invalid_count += 1
                invalid_names.append(name)
                tag = 'invalid'
                text = (f"{name}{color_marker} (image + "
                        f"{'label' if self.dataset_type == 0 else 'mask'}) "
                        f"[INVALID: {err}]")
            elif ann_type == 'empty':
                tag = 'empty'
                marker = "empty → include" if self._is_include_empty() else "empty → skip"
                text = f"{name}{color_marker} (image + label) [{marker}]"
                valid_count += 1
            elif ann_type in ('detect', 'obb', 'segment', 'error', 'mixed', 'unknown'):
                tag = ann_type
                text = f"{name}{color_marker} (image + label) [{ann_type}]"
                valid_count += 1
            else:
                tag = None
                if self.dataset_type == 1:
                    text = f"{name}{color_marker} (image + mask)"
                else:
                    text = f"{name}{color_marker} (image only)"
                valid_count += 1

            item = self._make_list_item(text, tag=tag, error_msg=err)
            item.setData(Qt.UserRole, img_path)
            item.setData(Qt.UserRole + 1, tuple(pair))
            item.setData(Qt.UserRole + 2, 'valid' if is_valid else 'invalid')
            item.setData(Qt.UserRole + 3, color_type)
            self.file_list.addItem(item)

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Построение списка: {idx + 1}/{total} ({percent}%)"
                )
                QApplication.processEvents()

        self.file_list.blockSignals(False)
        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")

        if invalid_count > 0:
            preview = ", ".join(invalid_names[:5])
            more = "" if invalid_count <= 5 else f" и ещё {invalid_count - 5}"
            self.validation_warning_label.setText(
                f"⚠️ Обнаружено {invalid_count} некорректных пар: {preview}{more}. "
                f"Некорректные пары НЕ будут включены в датасет."
            )
            self.validation_warning_label.setVisible(True)
            self.log(f"Валидация: {valid_count} корректных, {invalid_count} некорректных")
        else:
            self.validation_warning_label.setVisible(False)
            self.log(f"Валидация: все {valid_count} пар корректны")

        self._update_color_stats_label()

    # ==================================================================
    # Снимок валидации по двойному клику
    # ==================================================================
    def _draw_annotations_on_image(self, img, annotations):
        out = img.copy()
        h, w = out.shape[:2]
        for ann in annotations:
            typ = ann[0]
            cls = ann[1]
            color = (0, 255, 0)
            if typ == 'detect':
                _, cls, cx, cy, bw, bh = ann
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, f"cls={cls}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif typ == 'obb':
                _, cls, points = ann
                pts = []
                for i in range(0, len(points), 2):
                    pts.append([int(points[i] * w), int(points[i + 1] * h)])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.polylines(out, [pts_np], isClosed=True, color=color, thickness=2)
                    cv2.putText(out, f"cls={cls}", (pts[0][0], max(15, pts[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif typ == 'segment':
                _, cls, points = ann
                pts = []
                for i in range(0, len(points), 2):
                    pts.append([int(points[i] * w), int(points[i + 1] * h)])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.polylines(out, [pts_np], isClosed=True, color=color, thickness=2)
                    cv2.putText(out, f"cls={cls}", (pts[0][0], max(15, pts[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    def _apply_mask_overlay(self, img, mask):
        overlay = img.copy()
        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val == 0:
                continue
            hue = (int(val) * 37) % 180
            bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            overlay[mask == val] = color
        return cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    def _open_pair_validation(self, pair):
        img_path = pair[0]
        aux_path = pair[1] if len(pair) > 1 else None

        img = cv2.imread(str(img_path))
        if img is None:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение:\n{img_path}")
            return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        annotations = []
        if self.dataset_type == 0 and aux_path:
            annotations = load_annotations(aux_path, img.shape[1], img.shape[0])

        if self.dataset_type == 1 and aux_path:
            mask = cv2.imread(str(aux_path), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                img = self._apply_mask_overlay(img, mask)
        elif annotations:
            img = self._draw_annotations_on_image(img, annotations)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Валидация: {os.path.basename(img_path)}")
        dlg.resize(1100, 800)
        layout = QVBoxLayout(dlg)

        is_valid, err, color_type = self._validate_pair_full(pair)
        color_str = {'color': 'Цветное', 'gray': 'Ч/Б', 'unknown': '—'}.get(color_type, '—')
        status = (f"{'✅ Пара корректна' if is_valid else f'❌ Ошибка валидации: {err}'}"
                  f"   |   Тип: {color_str}")
        status_label = QLabel(status)
        status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_label.setStyleSheet(
            "color: #007700; font-weight: bold;" if is_valid
            else "color: #b00000; font-weight: bold;"
        )
        layout.addWidget(status_label)

        view = SmartGraphicsView()
        view.set_pixmap(numpy_to_qpixmap(img))
        view.setMinimumHeight(500)
        layout.addWidget(view, 1)

        path_label = QLabel(f"Image: {img_path}\nAux:   {aux_path or '—'}")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(path_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.exec_()

    # ==================================================================
    # Флаги
    # ==================================================================
    def _on_include_empty_toggled(self, *_):
        if not self.pairs:
            return
        self._rebuild_file_list_display()

    def _is_include_empty(self):
        return bool(self.include_empty_annotations.isChecked())

    # ==================================================================
    # Смена типа датасета
    # ==================================================================
    def _on_dataset_type_changed(self, idx):
        self.dataset_type = idx
        self._validation_cache.clear()
        self.color_type_stats = {'color': 0, 'gray': 0, 'unknown': 0}
        if idx == 0:
            self.btn_labels.setText("Load Labels")
            self.btn_labels.setEnabled(True)
        elif idx == 1:
            self.btn_labels.setText("Load Masks")
            self.btn_labels.setEnabled(True)
        else:
            self.btn_labels.setText("No labels (images only)")
            self.btn_labels.setEnabled(False)
            self.pairs = []
            self.file_list.clear()
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.validation_warning_label.setVisible(False)
            self._update_color_stats_label()
            self.update_generate_button_state()
            self.update_split_counts()
        if self.images_folder:
            self.scan_pairs()

    # ==================================================================
    # Множители и split
    # ==================================================================
    def update_multiplier_slider_state(self):
        any_aug = (self.flip_horizontal.isChecked() or
                   self.flip_vertical.isChecked() or
                   self.rotate_90.isChecked() or
                   self.random_rotate.isChecked() or
                   self.random_crop.isChecked() or
                   self.shear.isChecked())
        self.aug_multiplier_slider.setEnabled(any_aug)
        if not any_aug:
            self.aug_multiplier_slider.setValue(2)
            self.aug_multiplier_label.setText("2x")
        else:
            discrete_count = 0
            if self.flip_horizontal.isChecked():
                discrete_count += 1
            if self.flip_vertical.isChecked():
                discrete_count += 1
            if self.rotate_90.isChecked():
                discrete_count += 2
            if self.random_rotate.isChecked() or self.random_crop.isChecked() or self.shear.isChecked():
                max_unique = "неограничено"
            else:
                max_unique = 2 ** discrete_count if discrete_count > 0 else 1
                if self.aug_multiplier_slider.value() > max_unique:
                    self.log(f"Предупреждение: выбранный множитель {self.aug_multiplier_slider.value()} может привести к дублированию, "
                             f"так как максимальное количество уникальных вариантов при выбранных аугментациях — {max_unique}.")
        self.update_split_counts()

    def update_multiplier_label(self, value):
        self.aug_multiplier_label.setText(f"{value}x")
        self.update_split_counts()

    def update_split(self, value, source):
        self.train_slider.blockSignals(True)
        self.val_slider.blockSignals(True)
        self.test_slider.blockSignals(True)
        try:
            train = self.train_slider.value()
            val = self.val_slider.value()
            test = self.test_slider.value()
            if source == 'train':
                if val + test == 0:
                    new_val = 50
                    new_test = 50
                else:
                    total_remaining = val + test
                    new_val = int(val * (100 - train) / total_remaining)
                    new_test = (100 - train) - new_val
                new_val = max(0, min(100, new_val))
                new_test = max(0, min(100, new_test))
                if new_val + new_test != 100 - train:
                    new_val = (100 - train) - new_test
                self.val_slider.setValue(new_val)
                self.test_slider.setValue(new_test)
            elif source == 'val':
                new_test = 100 - train - value
                if new_test < 0:
                    new_test = 0
                    value = 100 - train
                self.test_slider.setValue(new_test)
                self.val_slider.setValue(value)
            elif source == 'test':
                new_val = 100 - train - value
                if new_val < 0:
                    new_val = 0
                    value = 100 - train
                self.val_slider.setValue(new_val)
                self.test_slider.setValue(value)
        finally:
            self.train_slider.blockSignals(False)
            self.val_slider.blockSignals(False)
            self.test_slider.blockSignals(False)
        self.train_label.setText(f"{self.train_slider.value()}%")
        self.val_label.setText(f"{self.val_slider.value()}%")
        self.test_label.setText(f"{self.test_slider.value()}%")
        self.update_split_counts()

    # ==================================================================
    # Выбор папок
    # ==================================================================
    def select_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self.images_folder = folder
            self.log(f"Images folder: {folder}")
            self._validation_cache.clear()
            self.scan_pairs()

    def select_labels_folder(self):
        if self.dataset_type == 2:
            return
        self._validation_cache.clear()
        if self.dataset_type_combo.currentIndex() == 0:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
            if folder:
                self.labels_folder = folder
                self.masks_folder = None
                self.log(f"Labels folder: {folder}")
                self.scan_pairs()
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with mask images")
            if folder:
                self.masks_folder = folder
                self.labels_folder = folder
                self.log(f"Masks folder: {folder}")
                self.scan_pairs()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save YAML and dataset")
        if folder:
            self.output_folder = folder
            self.output_path_display.setText(folder)
            self.btn_open_output.setEnabled(True)
            self.log(f"Output folder: {folder}")
            self.update_generate_button_state()

    def open_output_folder(self):
        if self.output_folder and os.path.exists(self.output_folder):
            if sys.platform == 'win32':
                os.startfile(self.output_folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{self.output_folder}"')
            else:
                os.system(f'xdg-open "{self.output_folder}"')
        else:
            QMessageBox.warning(self, "Ошибка", "Папка не выбрана или не существует.")

    # ==================================================================
    # Определение типа аннотации (быстро, только по числу токенов)
    # ==================================================================
    def _detect_annotation_type(self, label_path):
        types = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    l = len(parts)
                    if l == 5:
                        types.add('detect')
                    elif l == 9:
                        types.add('obb')
                    elif l >= 7 and (l - 1) % 2 == 0:
                        types.add('segment')
                    else:
                        types.add('unknown')
        except Exception:
            return 'error'
        if len(types) == 0:
            return 'empty'
        if len(types) == 1:
            return next(iter(types))
        return 'mixed'

    def _check_pair_type_consistency(self):
        return None

    # ==================================================================
    # Сканирование пар
    # ==================================================================
    def scan_pairs(self):
        self.pairs.clear()
        self.file_list.clear()
        self.annotation_types_stats = {}
        self.color_type_stats = {'color': 0, 'gray': 0, 'unknown': 0}
        self._validation_cache.clear()

        if not self.images_folder:
            self.log("Папка с изображениями не выбрана")
            self._update_color_stats_label()
            return

        dataset_type_idx = self.dataset_type_combo.currentIndex()
        self.log(f"Начинаем сканирование пар для типа {dataset_type_idx}")

        if dataset_type_idx == 0:  # Метки
            if not self.labels_folder:
                self.log("Папка с метками не выбрана")
                self._update_pair_count_label("", error=True)
                self._update_color_stats_label()
                return
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            img_files = {}
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_files[name] = os.path.join(self.images_folder, f)
            except Exception as e:
                self.log(f"Ошибка при чтении папки изображений: {e}")
                self._update_color_stats_label()
                return
            for name, img_path in img_files.items():
                label_path = os.path.join(self.labels_folder, name + '.txt')
                if os.path.exists(label_path):
                    try:
                        ann_type = self._detect_annotation_type(label_path)
                        self.annotation_types_stats[ann_type] = self.annotation_types_stats.get(ann_type, 0) + 1
                        self.pairs.append((img_path, label_path))
                    except Exception as e:
                        self.log(f"Ошибка обработки пары {name}: {e}")

        elif dataset_type_idx == 1:  # Маски
            if not self.masks_folder:
                self.log("Папка с масками не выбрана")
                self._update_pair_count_label("", error=True)
                self._update_color_stats_label()
                return
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            img_files = {}
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_files[name] = os.path.join(self.images_folder, f)
            except Exception as e:
                self.log(f"Ошибка при чтении папки изображений: {e}")
                self._update_color_stats_label()
                return
            for name, img_path in img_files.items():
                found = False
                for ext in img_extensions:
                    mask_path = os.path.join(self.masks_folder, name + ext)
                    try:
                        if os.path.exists(mask_path):
                            if os.path.abspath(mask_path) == os.path.abspath(img_path):
                                self.log(f"Предупреждение: файл маски совпадает с файлом изображения {img_path}, пара пропущена")
                                break
                            self.pairs.append((img_path, mask_path))
                            found = True
                            break
                    except Exception as e:
                        self.log(f"Ошибка при проверке маски {mask_path}: {e}")
                        continue
                if not found:
                    try:
                        for f in os.listdir(self.masks_folder):
                            mask_name, mask_ext = os.path.splitext(f)
                            if mask_name == name and mask_ext.lower() in img_extensions:
                                mask_path = os.path.join(self.masks_folder, f)
                                if os.path.abspath(mask_path) == os.path.abspath(img_path):
                                    self.log(f"Предупреждение: файл маски совпадает с файлом изображения {img_path}, пара пропущена")
                                    continue
                                self.pairs.append((img_path, mask_path))
                                found = True
                                break
                    except Exception as e:
                        self.log(f"Ошибка при поиске маски для {name}: {e}")

            self.log(f"Проверка валидности {len(self.pairs)} масок...")
            valid_pairs = []
            for img_path, mask_path in self.pairs:
                try:
                    test_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if test_mask is None:
                        self.log(f"Предупреждение: не удалось прочитать маску {mask_path} – файл повреждён или не является изображением")
                        continue
                    valid_pairs.append((img_path, mask_path))
                except Exception as e:
                    self.log(f"Критическая ошибка при проверке маски {mask_path}: {e}")
                    continue
            self.pairs = valid_pairs
            self.log(f"После проверки осталось {len(self.pairs)} валидных пар")

        else:  # Тип 2 – только изображения
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_path = os.path.join(self.images_folder, f)
                        self.pairs.append((img_path, None))
                self.annotation_types_stats = {}
            except Exception as e:
                self.log(f"Ошибка при сканировании изображений: {e}")

        # ---- Валидация ----
        valid_count, invalid_count = self._validate_all_pairs()

        # ---- Статистика по цветности ----
        self._recompute_color_stats()

        # ---- Отображение с подсветкой ----
        self._rebuild_file_list_display()

        error_msg = self._check_pair_type_consistency()
        extra_info = ""
        if self.annotation_types_stats:
            type_str = ", ".join([f"{k}:{v}" for k, v in self.annotation_types_stats.items()])
            extra_info = f"  [{type_str}]"
        if invalid_count > 0:
            extra_info += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra_info, error=bool(error_msg) or invalid_count > 0)
        self.log(f"Найдено пар: {len(self.pairs)} (корректных: {valid_count}, некорректных: {invalid_count})")
        if error_msg:
            self.log(error_msg)

        try:
            self.collect_classes()
            self.update_generate_button_state()
            self.update_split_counts()
        except Exception as e:
            self.log(f"Ошибка при обновлении интерфейса после сканирования: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _update_pair_count_label(self, extra_text, error=False):
        text = f"Всего пар: {len(self.pairs)}"
        if extra_text:
            text += f"  {extra_text}"
        self.pair_count_label.setText(text)
        if error:
            self.pair_count_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pair_count_label.setStyleSheet("")

    # ==================================================================
    # Классы
    # ==================================================================
    def rebuild_class_mapping(self):
        if self.dataset_type != 0:
            return
        self.class_remap = {}
        for row in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(row, 3)
            if cb and cb.isChecked():
                try:
                    original = int(self.class_table.item(row, 1).text())
                    new = int(self.class_table.item(row, 0).text())
                    self.class_remap[original] = new
                except (ValueError, TypeError):
                    continue

    def update_included_classes(self):
        if self.dataset_type == 0:
            self.included_orig_ids = set()
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 3)
                if cb and cb.isChecked():
                    original = int(self.class_table.item(row, 1).text())
                    self.included_orig_ids.add(original)
            self.rebuild_class_mapping()
            self.unique_classes = sorted(set(self.class_remap.values()))
            self.num_classes = len(self.unique_classes)
            self.mask_class_remap = {}
        elif self.dataset_type == 1:
            if self.class_table.rowCount() == 0:
                return
            self.included_orig_ids = set()
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 3)
                if cb and cb.isChecked():
                    try:
                        orig_id = int(self.class_table.item(row, 1).text())
                        self.included_orig_ids.add(orig_id)
                    except (ValueError, TypeError, AttributeError):
                        continue
            sorted_included = sorted(self.included_orig_ids)
            self.mask_class_remap = {old: new for new, old in enumerate(sorted_included)}
            self.unique_classes = list(range(len(sorted_included)))
            self.num_classes = len(self.unique_classes)
            for row in range(self.class_table.rowCount()):
                try:
                    orig_id = int(self.class_table.item(row, 1).text())
                    if orig_id in self.mask_class_remap:
                        self.class_table.item(row, 0).setText(str(self.mask_class_remap[orig_id]))
                    else:
                        self.class_table.item(row, 0).setText('-')
                except Exception:
                    pass
        else:
            self.included_orig_ids = set()
            self.class_remap = {}
            self.mask_class_remap = {}
            self.unique_classes = []
            self.num_classes = 0

    def collect_classes(self):
        if not self.pairs:
            self.class_table.setRowCount(0)
            return
        if self.dataset_type == 0:
            unique_ids = set()
            for _, label_path in self.pairs:
                anns = load_annotations(label_path, 1, 1)
                for ann in anns:
                    cls = ann[1]
                    unique_ids.add(cls)
            self.original_ids = sorted(unique_ids)
            self.class_table.setRowCount(len(self.original_ids))
            for i, cid in enumerate(self.original_ids):
                id_item = QTableWidgetItem(str(cid))
                id_item.setFlags(id_item.flags() | Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, id_item)
                orig_item = QTableWidgetItem(str(cid))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)
                name_item = QTableWidgetItem(str(cid))
                self.class_table.setItem(i, 2, name_item)
                cb = QCheckBox()
                cb.setChecked(True)
                cb.stateChanged.connect(lambda state, row=i: self.update_included_classes())
                self.class_table.setCellWidget(i, 3, cb)
            self.update_included_classes()
        elif self.dataset_type == 1:
            unique_vals = set()
            valid_pairs_for_classes = []
            for img_path, mask_path in self.pairs:
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        self.log(f"Предупреждение: не удалось прочитать маску {mask_path} при сборе классов – пропускаем")
                        continue
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    if mask.size == 0:
                        continue
                    if mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)
                    uniq = np.unique(mask)
                    unique_vals.update(uniq)
                    valid_pairs_for_classes.append((img_path, mask_path))
                except Exception as e:
                    self.log(f"Ошибка при обработке маски {mask_path}: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            if not valid_pairs_for_classes:
                self.log("Нет валидных масок для определения классов")
                self.class_table.setRowCount(0)
                self.original_ids = []
                self.update_included_classes()
                return
            self.pairs = valid_pairs_for_classes
            self.log(f"После фильтрации для классов осталось {len(self.pairs)} пар")
            all_vals = sorted(unique_vals)
            self.original_ids = all_vals
            temp_remap = {old: new for new, old in enumerate(all_vals)}
            self.class_table.setRowCount(0)
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(len(all_vals))
                for i, old_id in enumerate(all_vals):
                    new_id = temp_remap[old_id]
                    new_item = QTableWidgetItem(str(new_id))
                    new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 0, new_item)
                    orig_item = QTableWidgetItem(str(old_id))
                    orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 1, orig_item)
                    name_item = QTableWidgetItem(str(old_id))
                    self.class_table.setItem(i, 2, name_item)
                    cb = QCheckBox()
                    cb.setChecked(True)
                    cb.stateChanged.connect(lambda state, row=i: self.update_included_classes())
                    self.class_table.setCellWidget(i, 3, cb)
            finally:
                self.class_table.blockSignals(False)
            self.update_included_classes()
        else:
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.update_included_classes()

    def get_class_names_from_table(self):
        names = {}
        if self.dataset_type == 2:
            return names
        for i in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(i, 3)
            if cb and cb.isChecked():
                new_id = int(self.class_table.item(i, 0).text())
                name = self.class_table.item(i, 2).text().strip()
                if not name:
                    name = str(new_id)
                names[new_id] = name
        return names

    def on_class_table_item_changed(self, item):
        if item.column() == 0:
            self.update_included_classes()

    def update_generate_button_state(self):
        has_pairs = len(self.pairs) > 0
        has_output = bool(self.output_folder)
        self.generate_btn.setEnabled(has_pairs and has_output)

    # ==================================================================
    # Генерация
    # ==================================================================
    def on_generate(self):
        if not self.images_folder or not self.output_folder:
            QMessageBox.warning(self, "Error", "Please select images and output folders!")
            return
        if self.dataset_type_combo.currentIndex() == 0 and not self.labels_folder:
            QMessageBox.warning(self, "Error", "Please select labels folder for detection!")
            return
        if self.dataset_type_combo.currentIndex() == 1 and not self.masks_folder:
            QMessageBox.warning(self, "Error", "Please select masks folder for segmentation!")
            return
        if len(self.pairs) == 0:
            QMessageBox.warning(self, "Error", "No pairs found!")
            return

        # Предупреждение о некорректных парах
        _, invalid_count = self._validate_all_pairs()
        if invalid_count > 0:
            reply = QMessageBox.question(
                self, "Некорректные пары",
                f"Обнаружено {invalid_count} некорректных пар.\n"
                f"Они будут автоматически исключены из датасета.\n\n"
                f"Продолжить генерацию?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

        error_msg = self._check_pair_type_consistency()
        if error_msg:
            reply = QMessageBox.question(self, "Несоответствие типа",
                                         f"{error_msg}\nВсё равно продолжить?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        train_pct = self.train_slider.value() / 100.0
        val_pct = self.val_slider.value() / 100.0
        test_pct = self.test_slider.value() / 100.0

        if self.resize_keep.isChecked():
            resize_mode = "keep"
            target_size = None
        elif self.resize_stretch.isChecked():
            resize_mode = "stretch"
            target_size = self.resize_size.value()
        else:
            resize_mode = "fixed"
            target_size = self.resize_size.value()

        augmentations = {
            'horizontal_flip': self.flip_horizontal.isChecked(),
            'vertical_flip': self.flip_vertical.isChecked(),
            'rotate_90': self.rotate_90.isChecked(),
            'random_rotate': self.random_rotate.isChecked(),
            'random_rotate_angle': self.random_rotate_angle.value(),
            'random_crop': self.random_crop.isChecked(),
            'shear': self.shear.isChecked(),
            'shear_angle': self.shear_angle.value()
        }
        has_augmentations = any(v for k, v in augmentations.items()
                                if k not in ['random_rotate_angle', 'shear_angle'])
        aug_multiplier = self.aug_multiplier_slider.value() if has_augmentations else 1

        if target_size is None and has_augmentations:
            reply = QMessageBox.question(
                self, "Предупреждение",
                "Аугментации включены, но размер изображений не задан. "
                "Некоторые аугментации могут не работать. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        class_names = self.get_class_names_from_table()
        num_classes = len(class_names)

        included_orig_ids = None
        if self.dataset_type == 1:
            included_orig_ids = self.included_orig_ids.copy()

        include_empty_annotations = self.include_empty_annotations.isChecked()
        delete_source_after_success = self.delete_source_after_success.isChecked()

        # --- Фильтруем некорректные пары ---
        pairs_for_generation = []
        skipped_invalid = 0
        for pair in self.pairs:
            ok, _ = self._validate_pair(pair)
            if ok:
                pairs_for_generation.append(pair)
            else:
                skipped_invalid += 1

        if not pairs_for_generation:
            QMessageBox.critical(self, "Ошибка",
                                 "Все пары оказались некорректными. Генерация невозможна.")
            return

        self.log("=== Dataset generation started ===")
        self.log(f"Train: {train_pct*100:.0f}%, Val: {val_pct*100:.0f}%, Test: {test_pct*100:.0f}%")
        self.log(f"Image size: {'keep original' if target_size is None else target_size}")
        self.log(f"Augmentation: {'enabled' if has_augmentations else 'disabled'}")
        if has_augmentations:
            self.log(f"Augmentation multiplier: {aug_multiplier}x")
        self.log(f"Classes: {num_classes}")
        self.log(f"Include empty annotations: {include_empty_annotations}")
        self.log(f"Delete source files after success: {delete_source_after_success}")
        if skipped_invalid:
            self.log(f"Пропущено некорректных пар: {skipped_invalid}")

        self._set_generate_button_active(True)
        self.cancel_btn.setEnabled(True)
        bg_color = (0, 0, 0) if self.bg_color_combo.currentText() == "Black" else (255, 255, 255)

        if self.dataset_type == 0:
            class_mapping = self.class_remap
        elif self.dataset_type == 1:
            class_mapping = self.mask_class_remap
        else:
            class_mapping = {}

        if self.dataset_type_combo.currentIndex() == 0:
            labels_path = self.labels_folder
        elif self.dataset_type_combo.currentIndex() == 1:
            labels_path = self.masks_folder
        else:
            labels_path = None

        self.generator_thread = DatasetGeneratorThread(
            self.images_folder,
            labels_path,
            self.output_folder, pairs_for_generation,
            train_pct, val_pct, test_pct, target_size,
            augmentations, has_augmentations, aug_multiplier,
            class_names, class_mapping, bg_color,
            self.dataset_type_combo.currentIndex(),
            self.masks_folder,
            num_classes,
            self.mask_class_remap,
            included_orig_ids=included_orig_ids,
            included_class_ids=None,
            total_orig_classes=len(self.original_ids),
            resize_mode=resize_mode,
            include_empty_annotations=include_empty_annotations,
            delete_source_after_success=delete_source_after_success,
        )

        self.generator_thread.log_signal.connect(self.log)
        self.generator_thread.progress_signal.connect(self.update_progress)
        self.generator_thread.finished_signal.connect(self.on_generation_finished)
        self.generator_thread.start()

    def _set_generate_button_active(self, active):
        if active:
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("Генерация...")
            self.generate_btn.setStyleSheet("background-color: orange; color: black; font-weight: bold;")
        else:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate Dataset")
            self.generate_btn.setStyleSheet("")

    def on_cancel_generation(self):
        if self.generator_thread and self.generator_thread.isRunning():
            self.log("Отмена генерации датасета...")
            self.generator_thread.cancel()
            self.cancel_btn.setEnabled(False)

    def update_progress(self, current, total):
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"Готово: {percent}% ({current}/{total})")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Готово: 0% (0/0)")

    def on_generation_finished(self, success, message):
        self._set_generate_button_active(False)
        self.cancel_btn.setEnabled(False)

        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Успех", message)

            if self.delete_source_after_success.isChecked():
                self.pairs = []
                self.file_list.clear()
                self.class_table.setRowCount(0)
                self.original_ids = []
                self.unique_classes = []
                self.num_classes = 0
                self.annotation_types_stats = {}
                self.class_remap = {}
                self.mask_class_remap = {}
                self.included_orig_ids = set()
                self._validation_cache.clear()
                self.color_type_stats = {'color': 0, 'gray': 0, 'unknown': 0}
                self.validation_warning_label.setVisible(False)
                self._update_pair_count_label("")
                self._update_color_stats_label()
                self.update_split_counts()
                self.update_generate_button_state()
                self.log("Исходные файлы удалены, список пар очищен.")
        else:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации датасета:\n{message}")
        self.log(message)

    def log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def update_split_counts(self):
        total_pairs = len(self.pairs)
        if total_pairs == 0:
            self.train_count_label.setText("0")
            self.val_count_label.setText("0")
            self.test_count_label.setText("0")
            self.total_count_with_multiplier.setText("Итоговое кол-во снимков: 0")
            return
        train_pct = self.train_slider.value() / 100.0
        val_pct = self.val_slider.value() / 100.0
        test_pct = self.test_slider.value() / 100.0
        multiplier = self.aug_multiplier_slider.value() if self.aug_multiplier_slider.isEnabled() else 1
        total_generated = total_pairs * multiplier
        train_count = int(total_generated * train_pct)
        val_count = int(total_generated * val_pct)
        test_count = total_generated - train_count - val_count
        self.train_count_label.setText(str(train_count))
        self.val_count_label.setText(str(val_count))
        self.test_count_label.setText(str(test_count))
        self.total_count_with_multiplier.setText(f"Итоговое кол-во снимков: {total_generated}")

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetPreparationWindow()
    window.show()
    sys.exit(app.exec_())