# utils/DatasetGenerator.py
"""
Генератор датасета.

Три пути (input_type):

    INPUT_LABELS:
        Источник — .txt-аннотации.
        Аугментации — annotation-aware (flip, rot90, rotate, shear, crop).
        Выход:
            • YOLO labels (.txt) с возможной конвертацией subtype;
            • U-Net masks (.png), offset применяется на этапе экспорта;
            • masks_data.yaml;
            • COCO annotations_{split}.json.

    INPUT_MASKS:
        Источник — маски (.png) + изображения.
        Аугментации — ТОЛЬКО простые (flip, rot90). Lossless.
        Выход:
            • U-Net masks (.png) как есть (lossless);
            • masks_data.yaml;
            • Опционально YOLO labels / COCO — через findContours (lossy,
              с предупреждением в UI).

    INPUT_IMAGES:
        Источник — только изображения.
        Аугментации — image-only.
        Выход — только images/.

Классовая нумерация — всегда плотная 0..N-1 (после _apply_class_mapping).
"""
from import_libs_internal import *

from utils.annotation_converter import (
    convert_annotations, SUBTYPE_KEEP,
)
from utils.coco_export import (
    export_coco_from_labels, export_coco_from_masks,
)
from utils.unet_export import (
    export_masks_from_annotations, write_unet_yaml,
)
from utils.image_io import (
    load_annotations, save_annotations, read_mask_with_fallback,
    convert_segment_masks_to_yolo_seg_manual, FORMAT_HINT_AUTO,
)


ULTRALYTICS_AVAILABLE = False
try:
    import ultralytics
    print(f"Ultralytics YOLO version: {ultralytics.__version__}")
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    print("Не удалось определить версию YOLO (ultralytics не установлена или повреждена)")


# Значения input_type — синхронизированы с UI
INPUT_LABELS = 0
INPUT_MASKS  = 1
INPUT_IMAGES = 2


class DatasetGeneratorThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self,
                 images_folder, labels_folder, output_folder, pairs,
                 train_pct, val_pct, test_pct, target_size,
                 augmentations, has_augmentations, aug_multiplier,
                 class_names, class_mapping, bg_color,
                 input_type, masks_folder, num_classes,
                 mask_class_remap,
                 included_orig_ids=None, included_class_ids=None,
                 total_orig_classes=1, resize_mode="fixed",
                 include_empty_annotations=False,
                 delete_source_after_success=False,
                 input_format=FORMAT_HINT_AUTO,
                 out_yolo_txt=True,
                 out_yolo_subtype=SUBTYPE_KEEP,
                 out_yolo_obb_format='corners',
                 out_masks=False,
                 out_masks_offset=1,
                 out_unet_yaml=False,
                 out_coco=False,
                 mask_pixel_offset_in=1,
                 min_ann_area_px=0):
        super().__init__()
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.output_folder = output_folder
        self.resize_mode = resize_mode
        self.pairs = pairs
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.target_size = target_size
        self.augmentations = augmentations
        self.has_augmentations = has_augmentations
        self.aug_multiplier = aug_multiplier
        self.class_names = class_names                # плотный 0..N-1
        self.class_mapping = class_mapping            # orig → dense
        self._is_cancelled = False
        self.bg_color = bg_color
        self.input_type = input_type
        self.masks_folder = masks_folder
        self.num_classes = num_classes
        self.total_orig_classes = total_orig_classes
        self.mask_class_remap = mask_class_remap      # orig pixel → dense
        self.included_orig_ids = included_orig_ids
        self.included_class_ids = included_class_ids
        self.include_empty_annotations = include_empty_annotations
        self.delete_source_after_success = delete_source_after_success
        self.input_format = input_format or FORMAT_HINT_AUTO

        # Форматы выхода
        self.out_yolo_txt = bool(out_yolo_txt)
        self.out_yolo_subtype = out_yolo_subtype or SUBTYPE_KEEP
        self.out_yolo_obb_format = out_yolo_obb_format or 'corners'
        self.out_masks = bool(out_masks)
        self.out_masks_offset = int(out_masks_offset)
        self.out_unet_yaml = bool(out_unet_yaml)
        self.out_coco = bool(out_coco)
        self.mask_pixel_offset_in = int(mask_pixel_offset_in)
        # NEW: минимальная площадь аннотации в пикселях²
        self.min_ann_area_px = int(min_ann_area_px or 0)
        # Счётчик отброшенных мелких аннотаций (для итогового лога)
        self._dropped_ann_count = 0

        # Обработанные исходные файлы: (img_path, label_or_none, mask_or_none)
        self._processed_source_files = []

        self._create_transforms()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def cancel(self):
        self._is_cancelled = True

    def log(self, message):
        self.log_signal.emit(message)

    def _create_transforms(self):
        # Простые — доступны всегда
        self.simple_aug_needed = any([
            self.augmentations.get('horizontal_flip', False),
            self.augmentations.get('vertical_flip', False),
            self.augmentations.get('rotate_90', False),
        ])

        # Сложные — только для labels/images (mask-path использует только simple)
        if self.input_type == INPUT_MASKS:
            self.complex_aug_needed = False
            self.complex_mask_aug_needed = False
        else:
            self.complex_aug_needed = any([
                self.augmentations.get('random_rotate', False),
                self.augmentations.get('shear', False),
                self.augmentations.get('random_crop', False),
            ])
            self.complex_mask_aug_needed = False  # не используется

    def _compute_image_hash(self, img):
        return hashlib.md5(img.tobytes()).hexdigest()

    def _annotation_area_px(self, ann, w, h):
        """
        Площадь аннотации в пикселях².
        • detect  — площадь bbox;
        • obb/segment — площадь полигона (шнуровка).
        """
        typ = ann[0]
        if typ == 'detect':
            try:
                _, _, cx, cy, bw, bh = ann
                return float(bw) * float(w) * float(bh) * float(h)
            except Exception:
                return 0.0

        # obb / segment
        try:
            pts = ann[2]
        except Exception:
            return 0.0
        n = len(pts) // 2
        if n < 3:
            return 0.0
        xs = [pts[i] * w for i in range(0, len(pts), 2)]
        ys = [pts[i] * h for i in range(1, len(pts), 2)]
        area = 0.0
        for i in range(n):
            area += xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
        return abs(area) * 0.5

    def _filter_small_annotations(self, anns, w, h):
        """
        Убирает аннотации с площадью < min_ann_area_px.
        Обновляет self._dropped_ann_count.
        """
        if self.min_ann_area_px <= 0 or not anns:
            return anns
        keep = []
        for ann in anns:
            area = self._annotation_area_px(ann, w, h)
            if area >= self.min_ann_area_px:
                keep.append(ann)
            else:
                self._dropped_ann_count += 1
        return keep

    def _load_mask(self, path):
        """Загружает маску, сохраняя dtype (uint8/uint16)."""
        try:
            mask = read_mask_with_fallback(str(path))
            if mask is None:
                self.log(f"Ошибка: не удалось прочитать маску {path}")
                return None
            if mask.ndim != 2:
                self.log(f"Ошибка: маска {path} имеет форму {mask.shape}, "
                         f"ожидалась 2D")
                return None
            if np.max(mask) == 0:
                if not self.include_empty_annotations:
                    self.log(f"Предупреждение: маска {path} — только фон, пропускаем")
                    return None
                self.log(f"Маска {path} без объектов — сохранена как "
                         f"негативный пример")
            return mask
        except Exception as e:
            self.log(f"Критическая ошибка при загрузке маски {path}: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _resize_mask(self, mask, target_size, resize_mode):
        """Ресайз маски с INTER_NEAREST и паддингом нулями (фон)."""
        if target_size is None:
            return mask
        h, w = mask.shape[:2]
        if resize_mode == "stretch":
            return cv2.resize(mask, (target_size, target_size),
                              interpolation=cv2.INTER_NEAREST)
        scale = target_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(mask, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom,
                                   pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=0)

    # ------------------------------------------------------------------
    # Плотный ремап class_id
    # ------------------------------------------------------------------
    def _dense_class_mapping(self):
        """
        Возвращает {orig: dense} для labels-пути.
        Гарантирует, что значения — плотный диапазон 0..N-1.
        """
        if not self.class_mapping:
            return {}
        values_sorted = sorted(set(int(v) for v in self.class_mapping.values()))
        dense = {old: i for i, old in enumerate(values_sorted)}
        return {int(k): dense[int(v)] for k, v in self.class_mapping.items()}

    def _dense_mask_class_mapping(self):
        """
        Возвращает {orig_pixel: dense} для masks-пути.
        orig_pixel — старое значение пикселя в маске.
        dense — плотный 0..N-1.
        """
        if not self.mask_class_remap:
            return {}
        values_sorted = sorted(set(int(v) for v in self.mask_class_remap.values()))
        dense = {old: i for i, old in enumerate(values_sorted)}
        return {int(k): dense[int(v)] for k, v in self.mask_class_remap.items()}

    def _apply_class_mapping(self, annotations):
        """Перекладывает class_id по плотному ремапу, отбрасывает unmapped."""
        dense_map = self._dense_class_mapping()
        if not dense_map:
            return annotations
        mapped = []
        for ann in annotations:
            typ, cls = ann[0], int(ann[1])
            new_cls = dense_map.get(cls)
            if new_cls is None:
                continue
            if typ == 'detect':
                mapped.append(('detect', new_cls, *ann[2:]))
            else:  # obb / segment
                mapped.append((typ, new_cls, ann[2]))
        return mapped

    def _remap_mask_pixels(self, mask):
        """
        Перекладывает значения пикселей маски по плотному ремапу
        И применяет out_masks_offset.

        В выходной маске U-Net:
            0                       — фон
            dense_id + offset       — класс (dense_id 0..N-1)

        Без этой поправки класс с dense_id=0 записывался бы пикселем 0
        и сливался с фоном — объекты исчезали бы из выходной маски.
        """
        dense_map = self._dense_mask_class_mapping()
        # Принудительно offset ≥ 1. offset=0 бессмысленен для U-Net:
        # он ставит первый класс на пиксель 0, что конфликтует с фоном.
        out_offset = int(self.out_masks_offset)
        if out_offset < 1:
            out_offset = 1

        max_new = max(dense_map.values()) if dense_map else 0
        max_pixel = max_new + out_offset
        dtype = np.uint16 if max_pixel > 255 else np.uint8
        out = np.zeros(mask.shape, dtype=dtype)

        if not dense_map:
            return out
        for old_val, new_val in dense_map.items():
            out[mask == old_val] = new_val + out_offset
        return out

    # ------------------------------------------------------------------
    # Запись пары (общая для всех путей)
    # ------------------------------------------------------------------
    def _write_label(self, anns, out_path, w, h):
        """Пишет .txt с учётом subtype и obb-формата."""
        converted = convert_annotations(anns, self.out_yolo_subtype)
        save_annotations(converted, str(out_path), w, h,
                         obb_as_xywhr=(self.out_yolo_obb_format == 'xywhr'))

    # ==================================================================
    # ГЕНЕРАЦИЯ ВСЕХ ПАР
    # ==================================================================
    def _generate_all_pairs(self, output_path):
        temp_images = output_path / 'all' / 'images'
        temp_labels = output_path / 'all' / 'labels'
        temp_masks = output_path / 'all' / 'masks'
        temp_images.mkdir(parents=True, exist_ok=True)

        # labels/ нужны для:
        #   input=labels → всегда
        #   input=masks + (out_yolo_txt или out_coco) → lossy-экспорт
        need_labels_dir = (self.input_type == INPUT_LABELS or
                           (self.input_type == INPUT_MASKS and
                            (self.out_yolo_txt or self.out_coco)))
        if need_labels_dir:
            temp_labels.mkdir(parents=True, exist_ok=True)

        # masks/ нужны для input=masks
        if self.input_type == INPUT_MASKS:
            temp_masks.mkdir(parents=True, exist_ok=True)

        all_pairs = []   # (img_path, label_path_or_None, mask_path_or_None)
        total_pairs = len(self.pairs)
        total_to_generate = total_pairs * max(1, self.aug_multiplier)
        generated = 0

        for pair in self.pairs:
            if self._is_cancelled:
                return None

            if self.input_type == INPUT_LABELS:
                self._process_pair_labels(pair, temp_images, temp_labels,
                                            all_pairs,
                                            total_to_generate)
                generated = len(all_pairs)
                self.progress_signal.emit(generated, total_to_generate)

            elif self.input_type == INPUT_MASKS:
                self._process_pair_masks(pair, temp_images, temp_masks,
                                          all_pairs,
                                          total_to_generate)
                generated = len(all_pairs)
                self.progress_signal.emit(generated, total_to_generate)

            else:  # INPUT_IMAGES
                self._process_pair_images(pair, temp_images,
                                            all_pairs,
                                            total_to_generate)
                generated = len(all_pairs)
                self.progress_signal.emit(generated, total_to_generate)

        # --- Для input=masks и запросе labels/COCO: конвертация масок в .txt ---
        if (self.input_type == INPUT_MASKS
                and (self.out_yolo_txt or self.out_coco)):
            self._convert_masks_to_yolo_labels(temp_masks, temp_labels,
                                                all_pairs)

        return all_pairs

    # ------------------------------------------------------------------
    # Обработка пары — LABELS
    # ------------------------------------------------------------------
    def _process_pair_labels(self, pair, temp_images, temp_labels,
                              all_pairs, total_to_generate):
        img_path = pair[0]
        label_path = pair[1] if len(pair) > 1 else None

        img = cv2.imread(str(img_path))
        if img is None:
            self.log(f"Предупреждение: не удалось прочитать {img_path}")
            return
        img_h, img_w = img.shape[:2]

        annotations = load_annotations(label_path, img_w, img_h,
                                        format_hint=self.input_format) \
            if label_path else []

        originally_empty = (len(annotations) == 0)
        if not originally_empty:
            annotations = self._apply_class_mapping(annotations)

        if not annotations:
            if originally_empty and self.include_empty_annotations:
                self.log(f"Пустая аннотация {label_path} — "
                         f"негативный пример")
            else:
                self.log(f"Нет аннотаций после маппинга классов в {label_path}")
                return

        img_orig, ann_orig = self._resize_with_annotations(img, annotations)

        # NEW: фильтр мелких аннотаций (после ресайза, чтобы площадь
        # считалась в тех же пикселях, что реально записываются)
        if self.min_ann_area_px > 0:
            before = len(ann_orig)
            ann_orig = self._filter_small_annotations(
                ann_orig, img_orig.shape[1], img_orig.shape[0])
            dropped = before - len(ann_orig)
            if dropped and not ann_orig:
                if self.include_empty_annotations:
                    self.log(f"[фильтр] все аннотации отброшены как мелкие: "
                             f"{os.path.basename(img_path)} — сохранён как негатив")
                else:
                    self.log(f"[фильтр] все аннотации отброшены как мелкие: "
                             f"{os.path.basename(img_path)} — пропуск")
                    return

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1] or '.png'

        out_img_path = temp_images / f"{base_name}{ext}"
        cv2.imwrite(str(out_img_path), img_orig)

        out_label_path = temp_labels / f"{base_name}.txt"
        self._write_label(ann_orig, out_label_path,
                           img_orig.shape[1], img_orig.shape[0])

        all_pairs.append((str(out_img_path), str(out_label_path), None))
        self._processed_source_files.append(
            (str(img_path), str(label_path) if label_path else None, None)
        )

        if not self.has_augmentations:
            return

        # --- Аугментации ---
        saved_hashes = {self._compute_image_hash(img_orig)}
        copy_idx = 1
        while copy_idx < self.aug_multiplier:
            if self._is_cancelled:
                return
            max_attempts = 10
            success = False
            for attempt in range(max_attempts):
                try:
                    aug_img = img.copy()
                    aug_anns = annotations.copy()
                    if self.simple_aug_needed:
                        aug_img, aug_anns = apply_simple_augmentations(
                            aug_img, aug_anns, self.augmentations)
                    if self.complex_aug_needed:
                        aug_img, aug_anns = apply_complex_augmentations(
                            aug_img, aug_anns, self.augmentations,
                            border_color=self.bg_color)

                    if not aug_anns and not originally_empty:
                        continue

                    aug_img, aug_anns = self._resize_with_annotations(
                        aug_img, aug_anns)

                    # NEW: фильтр мелких после ресайза аугментации
                    if self.min_ann_area_px > 0:
                        aug_anns = self._filter_small_annotations(
                            aug_anns, aug_img.shape[1], aug_img.shape[0])
                        if not aug_anns and not originally_empty:
                            continue

                    h = self._compute_image_hash(aug_img)
                    if h in saved_hashes:
                        continue

                    suffix = f"_aug{copy_idx}"
                    out_img2 = temp_images / f"{base_name}{suffix}{ext}"
                    cv2.imwrite(str(out_img2), aug_img)
                    out_label2 = temp_labels / f"{base_name}{suffix}.txt"
                    self._write_label(aug_anns, out_label2,
                                       aug_img.shape[1], aug_img.shape[0])
                    all_pairs.append((str(out_img2), str(out_label2), None))
                    saved_hashes.add(h)
                    self.progress_signal.emit(len(all_pairs), total_to_generate)
                    success = True
                    break
                except Exception as e:
                    self.log(f"Ошибка аугментации для {base_name} "
                             f"(попытка {attempt + 1}): {e}")
                    continue
            if not success:
                self.log(f"Не удалось создать уникальную копию "
                         f"{base_name}_aug{copy_idx} после {max_attempts} попыток")
            copy_idx += 1

    # ------------------------------------------------------------------
    # Обработка пары — MASKS (только простые аугментации, lossless)
    # ------------------------------------------------------------------
    def _process_pair_masks(self, pair, temp_images, temp_masks,
                             all_pairs, total_to_generate):
        img_path = pair[0]
        mask_path = pair[2] if len(pair) > 2 else (
            pair[1] if len(pair) > 1 else None
        )

        img = cv2.imread(str(img_path))
        if img is None:
            self.log(f"Предупреждение: не удалось прочитать {img_path}")
            return

        mask = self._load_mask(mask_path)
        if mask is None or mask.size == 0:
            return

        # --- Применяем плотный ремап пикселей маски ---
        mask = self._remap_mask_pixels(mask)

        if self.target_size is not None:
            if self.resize_mode == "fixed":
                img_orig, _ = resize_image_and_annotations(
                    img, [], self.target_size, self.bg_color)
            elif self.resize_mode == "stretch":
                img_orig = cv2.resize(img, (self.target_size, self.target_size),
                                       interpolation=cv2.INTER_LINEAR)
            else:
                img_orig = img.copy()
            mask_orig = self._resize_mask(mask, self.target_size,
                                            self.resize_mode)
        else:
            img_orig = img.copy()
            mask_orig = mask.copy()

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1] or '.png'

        out_img_path = temp_images / f"{base_name}{ext}"
        out_mask_path = temp_masks / f"{base_name}.png"
        cv2.imwrite(str(out_img_path), img_orig)
        cv2.imwrite(str(out_mask_path), mask_orig)

        all_pairs.append((str(out_img_path), None, str(out_mask_path)))
        self._processed_source_files.append(
            (str(img_path), None, str(mask_path) if mask_path else None)
        )

        if not self.has_augmentations or not self.simple_aug_needed:
            return

        # --- Только простые аугментации ---
        saved_hashes = {self._compute_image_hash(img_orig)}
        copy_idx = 1
        while copy_idx < self.aug_multiplier:
            if self._is_cancelled:
                return
            try:
                aug_img, aug_mask = img.copy(), mask.copy()
                aug_img, aug_mask = apply_simple_augmentations_mask(
                    aug_img, aug_mask, self.augmentations)

                if self.target_size is not None:
                    if self.resize_mode == "fixed":
                        aug_img, _ = resize_image_and_annotations(
                            aug_img, [], self.target_size, self.bg_color)
                    elif self.resize_mode == "stretch":
                        aug_img = cv2.resize(
                            aug_img, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_LINEAR)
                    aug_mask = self._resize_mask(aug_mask, self.target_size,
                                                   self.resize_mode)

                h = self._compute_image_hash(aug_img)
                if h in saved_hashes:
                    # Простые аугментации дают ограниченное число вариантов.
                    # Пропускаем дубликат.
                    copy_idx += 1
                    continue

                suffix = f"_aug{copy_idx}"
                out_img2 = temp_images / f"{base_name}{suffix}{ext}"
                out_mask2 = temp_masks / f"{base_name}{suffix}.png"
                cv2.imwrite(str(out_img2), aug_img)
                cv2.imwrite(str(out_mask2), aug_mask)
                all_pairs.append((str(out_img2), None, str(out_mask2)))
                saved_hashes.add(h)
                self.progress_signal.emit(len(all_pairs), total_to_generate)
            except Exception as e:
                self.log(f"Ошибка аугментации маски {base_name} "
                         f"(копия {copy_idx}): {e}")
            copy_idx += 1

    # ------------------------------------------------------------------
    # Обработка пары — IMAGES
    # ------------------------------------------------------------------
    def _process_pair_images(self, pair, temp_images,
                              all_pairs, total_to_generate):
        img_path = pair[0]
        img = cv2.imread(str(img_path))
        if img is None:
            self.log(f"Предупреждение: не удалось прочитать {img_path}")
            return

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1] or '.png'

        if self.target_size is not None:
            if self.resize_mode == "fixed":
                img_orig, _ = resize_image_and_annotations(
                    img, [], self.target_size, self.bg_color)
            elif self.resize_mode == "stretch":
                img_orig = cv2.resize(img, (self.target_size, self.target_size),
                                       interpolation=cv2.INTER_LINEAR)
            else:
                img_orig = img.copy()
        else:
            img_orig = img.copy()

        out_img_path = temp_images / f"{base_name}{ext}"
        cv2.imwrite(str(out_img_path), img_orig)
        all_pairs.append((str(out_img_path), None, None))
        self._processed_source_files.append((str(img_path), None, None))

        if not self.has_augmentations:
            return

        saved_hashes = {self._compute_image_hash(img_orig)}
        copy_idx = 1
        while copy_idx < self.aug_multiplier:
            if self._is_cancelled:
                return
            max_attempts = 10
            success = False
            for attempt in range(max_attempts):
                try:
                    aug_img = img.copy()
                    if self.simple_aug_needed:
                        aug_img, _ = apply_simple_augmentations(
                            aug_img, [], self.augmentations)
                    if self.complex_aug_needed:
                        aug_img, _ = apply_complex_augmentations(
                            aug_img, [], self.augmentations,
                            border_color=self.bg_color)

                    if self.target_size is not None:
                        if self.resize_mode == "fixed":
                            aug_img, _ = resize_image_and_annotations(
                                aug_img, [], self.target_size, self.bg_color)
                        elif self.resize_mode == "stretch":
                            aug_img = cv2.resize(
                                aug_img, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_LINEAR)

                    h = self._compute_image_hash(aug_img)
                    if h in saved_hashes:
                        continue

                    suffix = f"_aug{copy_idx}"
                    out_img2 = temp_images / f"{base_name}{suffix}{ext}"
                    cv2.imwrite(str(out_img2), aug_img)
                    all_pairs.append((str(out_img2), None, None))
                    saved_hashes.add(h)
                    self.progress_signal.emit(len(all_pairs), total_to_generate)
                    success = True
                    break
                except Exception as e:
                    self.log(f"Ошибка аугментации для {base_name} "
                             f"(попытка {attempt + 1}): {e}")
                    continue
            if not success:
                self.log(f"Не удалось создать уникальную копию "
                         f"{base_name}_aug{copy_idx} после {max_attempts} попыток")
            copy_idx += 1

    # ------------------------------------------------------------------
    # Ресайз пары (img, anns) — единая точка
    # ------------------------------------------------------------------
    def _resize_with_annotations(self, img, anns):
        if self.target_size is None:
            return img.copy(), list(anns)
        if self.resize_mode == "fixed":
            return resize_image_and_annotations(img, anns,
                                                 self.target_size, self.bg_color)
        if self.resize_mode == "stretch":
            return resize_image_and_annotations_stretch(img, anns,
                                                         self.target_size)
        return img.copy(), list(anns)

    # ------------------------------------------------------------------
    # Конвертация масок в .txt (для input=masks + out_yolo_txt/coco)
    # ------------------------------------------------------------------
    def _convert_masks_to_yolo_labels(self, temp_masks, temp_labels, all_pairs):
        """
        Lossy: findContours на масках → .txt с полигонами.
        Использует mask_class_remap (orig pixel → dense) и mask_class_offset_in.
        """
        if not ULTRALYTICS_AVAILABLE:
            self.log("⚠️ Ultralytics не установлен — конвертация масок "
                     "в .txt пропущена.")
            return

        masks_dir = str(temp_masks)
        labels_dir = str(temp_labels)
        labels_dir_p = Path(labels_dir)
        labels_dir_p.mkdir(parents=True, exist_ok=True)

        # pixel_value в маске = orig_pixel, соответствует dense yolo_cls
        dense_map = self._dense_mask_class_mapping()
        if not dense_map:
            self.log("⚠️ Нет маппинга масок → YOLO, конвертация пропущена.")
            return

        # Маски в temp_masks уже прошли _remap_mask_pixels,
        # поэтому значения пикселей в них — dense_id + out_masks_offset.
        out_offset = int(self.out_masks_offset)
        if out_offset < 1:
            out_offset = 1

        pixel_to_class = {}
        for dense_id in sorted(set(dense_map.values())):
            pixel_val = dense_id + out_offset
            pixel_to_class[pixel_val] = dense_id

        self.log(f"[masks → txt] pixel_to_class={pixel_to_class} "
                 f"(out_offset={out_offset})")

        try:
            convert_segment_masks_to_yolo_seg_manual(
                masks_dir, labels_dir, pixel_to_class,
                mask_offset=0,  # ключи уже содержат смещение
            )
        except TypeError:
            # старая сигнатура без mask_offset
            convert_segment_masks_to_yolo_seg_manual(
                masks_dir, labels_dir, pixel_to_class)

        # Обновляем all_pairs: связываем img с новым txt
        new_pairs = []
        for img_p, lbl_p, mask_p in all_pairs:
            if mask_p and not lbl_p:
                stem = Path(mask_p).stem
                cand = labels_dir_p / (stem + '.txt')
                if cand.exists():
                    new_pairs.append((img_p, str(cand), mask_p))
                else:
                    new_pairs.append((img_p, None, mask_p))
            else:
                new_pairs.append((img_p, lbl_p, mask_p))
        all_pairs.clear()
        all_pairs.extend(new_pairs)

    # ==================================================================
    # SPLIT + MOVE
    # ==================================================================
    def _split_and_move(self, output_path, all_pairs):
        random.shuffle(all_pairs)
        total = len(all_pairs)
        train_count = int(total * self.train_pct)
        val_count = int(total * self.val_pct)
        test_count = total - train_count - val_count

        train_pairs = all_pairs[:train_count]
        val_pairs = all_pairs[train_count:train_count + val_count]
        test_pairs = all_pairs[train_count + val_count:]

        self.log(f"Распределение: Train={len(train_pairs)}, "
                 f"Val={len(val_pairs)}, Test={len(test_pairs)}")

        def move_pairs(pairs, split):
            img_dst = output_path / 'images' / split
            lbl_dst = output_path / 'labels' / split
            msk_dst = output_path / 'masks' / split
            img_dst.mkdir(parents=True, exist_ok=True)
            if any(p[1] for p in pairs):
                lbl_dst.mkdir(parents=True, exist_ok=True)
            if any(p[2] for p in pairs):
                msk_dst.mkdir(parents=True, exist_ok=True)

            for img_p, lbl_p, msk_p in pairs:
                try:
                    shutil.move(img_p, img_dst / os.path.basename(img_p))
                except Exception as e:
                    self.log(f"[move] не удалось перенести {img_p}: {e}")
                if lbl_p:
                    try:
                        shutil.move(lbl_p, lbl_dst / os.path.basename(lbl_p))
                    except Exception as e:
                        self.log(f"[move] не удалось перенести {lbl_p}: {e}")
                if msk_p:
                    try:
                        shutil.move(msk_p, msk_dst / os.path.basename(msk_p))
                    except Exception as e:
                        self.log(f"[move] не удалось перенести {msk_p}: {e}")

        move_pairs(train_pairs, 'train')
        move_pairs(val_pairs, 'val')
        move_pairs(test_pairs, 'test')

        # Удаляем временную all/
        shutil.rmtree(output_path / 'all', ignore_errors=True)

    # ==================================================================
    # ЭКСПОРТ ФОРМАТОВ
    # ==================================================================
    def _write_classes_txt(self, output_path):
        if self.input_type == INPUT_IMAGES or not self.class_names:
            return
        sorted_cids = sorted(self.class_names.keys())
        with open(output_path / 'classes.txt', 'w', encoding='utf-8') as f:
            for cid in sorted_cids:
                f.write(str(self.class_names[cid]) + '\n')
        self.log(f"[classes.txt] {len(sorted_cids)} классов")

    def _write_data_yaml(self, output_path):
        if not self.out_yolo_txt or not self.class_names:
            return
        sorted_cids = sorted(self.class_names.keys())
        data_yaml = {
            'path': str(output_path.absolute()).replace('\\', '/'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(sorted_cids),
            'names': [str(self.class_names[cid]) for cid in sorted_cids],
        }
        with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False,
                      allow_unicode=True)
        self.log(f"[data.yaml] nc={len(sorted_cids)}")

    def _export_masks(self, output_path):
        """input=labels → out_masks: генерируем PNG из .txt."""
        if not self.out_masks:
            return
        if self.input_type == INPUT_LABELS:
            total = 0
            for split in ('train', 'val', 'test'):
                n = export_masks_from_annotations(
                    output_path, split, self.class_names,
                    mask_class_offset=self.out_masks_offset,
                    log_cb=self.log,
                )
                total += n
            self.log(f"[U-Net] всего масок: {total}")
        elif self.input_type == INPUT_MASKS:
            # Маски уже лежат в masks/{split} после split_and_move.
            self.log("[U-Net] маски сохранены в masks/{split} "
                     "(lossless, без пересборки).")

    def _export_unet_yaml(self, output_path):
        if not self.out_unet_yaml:
            return
        if not self.class_names:
            self.log("⚠️ masks_data.yaml не записан: нет классов.")
            return
        write_unet_yaml(output_path, self.class_names,
                         mask_class_offset=self.out_masks_offset,
                         log_cb=self.log)

    def _export_coco(self, output_path):
        if not self.out_coco:
            return
        if not self.class_names:
            self.log("⚠️ COCO не записан: нет классов.")
            return

        for split in ('train', 'val', 'test'):
            if self.input_type == INPUT_LABELS:
                export_coco_from_labels(output_path, split,
                                          self.class_names,
                                          log_cb=self.log)
            elif self.input_type == INPUT_MASKS:
                # Lossy — из масок.
                export_coco_from_masks(
                    output_path, split, self.class_names,
                    mask_pixel_offset=self.out_masks_offset,
                    log_cb=self.log,
                )

    # ==================================================================
    # УДАЛЕНИЕ ИСХОДНИКОВ
    # ==================================================================
    def _delete_processed_sources(self):
        if not self._processed_source_files:
            self.log("Удаление исходников: нет обработанных пар.")
            return

        output_dir = os.path.abspath(self.output_folder)
        deleted = 0
        missing = 0
        failed = 0
        seen = set()

        for img_path, label_path, mask_path in self._processed_source_files:
            for path in (img_path, label_path, mask_path):
                if not path:
                    continue
                norm = os.path.abspath(path)
                if norm in seen:
                    continue
                seen.add(norm)
                if norm.startswith(output_dir + os.sep) or norm == output_dir:
                    self.log(f"Удаление: пропущен файл внутри output_folder: {path}")
                    continue
                if not os.path.exists(path):
                    missing += 1
                    continue
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    failed += 1
                    self.log(f"Не удалось удалить {path}: {e}")

        self.log(f"Удаление исходников: удалено={deleted}, "
                 f"отсутствовало={missing}, ошибок={failed}")

    # ==================================================================
    # RUN
    # ==================================================================
    def run(self):
        try:
            self.log("=== Начало генерации датасета ===")
            output_path = Path(self.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            # --- Логирование конфигурации ---
            self.log(f"input_type={self.input_type}, "
                     f"input_format={self.input_format}")
            self.log(f"outputs: yolo_txt={self.out_yolo_txt} "
                     f"(subtype={self.out_yolo_subtype}, "
                     f"obb={self.out_yolo_obb_format}), "
                     f"masks={self.out_masks} (offset={self.out_masks_offset}), "
                     f"unet_yaml={self.out_unet_yaml}, coco={self.out_coco}")

            if not any([self.out_yolo_txt, self.out_masks, self.out_coco]):
                if self.input_type != INPUT_IMAGES:
                    self.finished_signal.emit(
                        False, "Не выбран ни один формат вывода "
                                "(YOLO txt / masks / COCO).")
                    return

            # --- 1. Генерация ---
            all_pairs = self._generate_all_pairs(output_path)
            if all_pairs is None:
                self.finished_signal.emit(False, "Генерация прервана пользователем.")
                return
            if not all_pairs:
                self.log("Ошибка: не удалось сгенерировать ни одной пары.")
                self.finished_signal.emit(False, "Нет данных для датасета.")
                return

            # --- 2. Split ---
            self._split_and_move(output_path, all_pairs)

            # --- 3. classes.txt (общий для всех форматов) ---
            self._write_classes_txt(output_path)

            # --- 4. Экспорт целевых форматов ---
            self._export_masks(output_path)
            self._export_unet_yaml(output_path)
            self._export_coco(output_path)
            self._write_data_yaml(output_path)

            # --- 5. Удаление исходников ---
            if self.delete_source_after_success:
                self.log("=== Удаление исходных файлов ===")
                self._delete_processed_sources()
            else:
                self.log("Удаление исходных файлов отключено.")

            if self.min_ann_area_px > 0:
                self.log(f"[фильтр] Min area: {self.min_ann_area_px} px², "
                         f"отброшено мелких аннотаций: {self._dropped_ann_count}")

            self.log("Генерация датасета завершена успешно!")
            self.finished_signal.emit(True, "Датасет успешно создан")
        except Exception as e:
            import traceback
            self.log(f"Критическая ошибка: {e}")
            self.log(traceback.format_exc())
            self.finished_signal.emit(False, str(e))