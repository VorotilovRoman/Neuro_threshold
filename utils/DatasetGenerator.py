# utils/DatasetGenerator.py
from import_libs_internal import *

ULTRALYTICS_AVAILABLE = False
try:
    import ultralytics
    print(f"Ultralytics YOLO version: {ultralytics.__version__}")
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    print("Не удалось определить версию YOLO (ultralytics не установлена или повреждена)")


class DatasetGeneratorThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, images_folder, labels_folder, output_folder, pairs,
                 train_pct, val_pct, test_pct, target_size, augmentations,
                 has_augmentations, aug_multiplier, class_names, class_mapping,
                 bg_color, dataset_type, masks_folder, num_classes,
                 mask_class_remap, included_orig_ids=None, included_class_ids=None,
                 total_orig_classes=1, resize_mode="fixed"):
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
        self.class_names = class_names          # словарь new_id -> name
        self.class_mapping = class_mapping      # original_id -> new_id (только для детекции)
        self._is_cancelled = False
        self.bg_color = bg_color
        self.dataset_type = dataset_type          # 0=detect, 1=segment, 2=images only
        self.masks_folder = masks_folder
        self.num_classes = num_classes
        self.total_orig_classes = total_orig_classes
        self.mask_class_remap = mask_class_remap
        self.included_orig_ids = included_orig_ids   # для масок
        self.included_class_ids = included_class_ids # для детекции (устаревший, не используется)
        self._create_transforms()

    def cancel(self):
        self._is_cancelled = True

    def log(self, message):
        self.log_signal.emit(message)

    def _create_transforms(self):
        # Определяем типы аугментаций для детекции
        self.simple_aug_needed = any([
            self.augmentations.get('horizontal_flip', False),
            self.augmentations.get('vertical_flip', False),
            self.augmentations.get('rotate_90', False)
        ])
        self.complex_aug_needed = any([
            self.augmentations.get('random_rotate', False),
            self.augmentations.get('shear', False)
        ])

        # Определяем типы аугментаций для масок
        self.simple_mask_aug_needed = any([
            self.augmentations.get('horizontal_flip', False),
            self.augmentations.get('vertical_flip', False),
            self.augmentations.get('rotate_90', False)
        ])
        self.complex_mask_aug_needed = any([
            self.augmentations.get('random_rotate', False),
            self.augmentations.get('shear', False)
        ])

    def _compute_image_hash(self, img):
        return hashlib.md5(img.tobytes()).hexdigest()

    def _load_mask(self, path):
        """Загружает маску с обработкой ошибок."""
        try:
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                self.log(
                    f"Ошибка: не удалось прочитать файл маски (формат не поддерживается или файл повреждён): {path}")
                return None
            # Если маска цветная (3 или 4 канала) – берём первый канал
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            # Убираем лишние измерения
            mask = mask.squeeze()
            # Приводим к uint8 (если вдруг другой тип)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            # Проверка на пустую маску (все нули)
            if np.max(mask) == 0:
                self.log(f"Предупреждение: маска {path} состоит только из нулей (фон) – пропускаем")
                return None

            return mask
        except Exception as e:
            self.log(f"Критическая ошибка при загрузке маски {path}: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _resize_mask(self, mask, target_size, resize_mode):
        """Ресайз маски для сегментации."""
        if target_size is None:
            return mask
        h, w = mask.shape[:2]
        if resize_mode == "stretch":
            return cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        else:  # "fixed"
            scale = target_size / max(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pad_top = (target_size - new_h) // 2
            pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            pad_right = target_size - new_w - pad_left
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=0)
            return padded

    def _apply_class_mapping(self, annotations):
        """Применяет маппинг original_id -> new_id к аннотациям."""
        if not self.class_mapping:
            return annotations
        mapped = []
        for ann in annotations:
            typ, cls = ann[0], ann[1]
            new_cls = self.class_mapping.get(cls)
            if new_cls is None:
                continue  # класс исключён из датасета (не отмечен галочкой)
            if typ == 'detect':
                mapped.append(('detect', new_cls, *ann[2:]))
            elif typ == 'obb':
                mapped.append(('obb', new_cls, ann[2]))
            else:  # segment
                mapped.append(('segment', new_cls, ann[2]))
        return mapped

    def _generate_all_pairs(self, output_path):
        temp_images = output_path / 'all' / 'images'
        temp_labels = output_path / 'all' / 'labels'
        temp_masks = output_path / 'all' / 'masks'
        temp_images.mkdir(parents=True, exist_ok=True)
        if self.dataset_type == 0:      # detection
            temp_labels.mkdir(parents=True, exist_ok=True)
        elif self.dataset_type == 1:    # segmentation
            temp_masks.mkdir(parents=True, exist_ok=True)
        # для типа 2 (images only) создаём только temp_images

        all_pairs = []
        total_pairs = len(self.pairs)
        total_to_generate = total_pairs * self.aug_multiplier
        generated = 0

        for pair in self.pairs:
            if self._is_cancelled:
                return None

            if self.dataset_type == 1:  # Segmentation (маски)
                img_path, mask_path = pair
                img = cv2.imread(str(img_path))
                if img is None:
                    self.log(f"Предупреждение: не удалось прочитать {img_path}")
                    continue
                mask = self._load_mask(mask_path)
                if mask is None or mask.size == 0:
                    self.log(f"Маска {mask_path} не загружена или пуста, пропускаем пару")
                    continue
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1]

                # --- Оригинал: только ресайз ---
                if self.target_size is not None:
                    if self.resize_mode == "fixed":
                        img_orig, _ = resize_image_and_annotations(img, [], self.target_size, self.bg_color)
                    elif self.resize_mode == "stretch":
                        img_orig = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_orig = img.copy()
                    mask_orig = self._resize_mask(mask, self.target_size, self.resize_mode)
                else:
                    img_orig = img.copy()
                    mask_orig = mask.copy()

                out_img_name = f"{base_name}{ext}"
                out_img_path = temp_images / out_img_name
                out_mask_name = f"{base_name}.png"
                out_mask_path = temp_masks / out_mask_name
                cv2.imwrite(str(out_img_path), img_orig)
                cv2.imwrite(str(out_mask_path), mask_orig)
                all_pairs.append((str(out_img_path), None, str(out_mask_path)))
                generated += 1
                self.progress_signal.emit(generated, total_to_generate)

                # --- Аугментированные копии ---
                if self.has_augmentations:
                    saved_hashes = {self._compute_image_hash(img_orig)}
                    copy_idx = 1
                    while copy_idx < self.aug_multiplier:
                        if self._is_cancelled:
                            return None
                        max_attempts = 10
                        success = False
                        for attempt in range(max_attempts):
                            try:
                                aug_img = img.copy()
                                aug_mask = mask.copy()
                                if self.simple_mask_aug_needed:
                                    aug_img, aug_mask = apply_simple_augmentations_mask(aug_img, aug_mask,
                                                                                        self.augmentations)
                                if self.complex_mask_aug_needed:
                                    aug_img, aug_mask = apply_complex_augmentations_mask(aug_img, aug_mask,
                                                                                         self.augmentations,
                                                                                         border_color=self.bg_color)

                                if self.target_size is not None:
                                    if self.resize_mode == "fixed":
                                        aug_img, _ = resize_image_and_annotations(aug_img, [], self.target_size, self.bg_color)
                                        aug_mask = self._resize_mask(aug_mask, self.target_size, self.resize_mode)
                                    elif self.resize_mode == "stretch":
                                        aug_img = cv2.resize(aug_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                                        aug_mask = self._resize_mask(aug_mask, self.target_size, self.resize_mode)

                                img_hash = self._compute_image_hash(aug_img)
                                if img_hash not in saved_hashes:
                                    suffix = f"_aug{copy_idx}"
                                    out_img_name2 = f"{base_name}{suffix}{ext}"
                                    out_img_path2 = temp_images / out_img_name2
                                    out_mask_name2 = f"{base_name}{suffix}.png"
                                    out_mask_path2 = temp_masks / out_mask_name2
                                    cv2.imwrite(str(out_img_path2), aug_img)
                                    cv2.imwrite(str(out_mask_path2), aug_mask)
                                    all_pairs.append((str(out_img_path2), None, str(out_mask_path2)))
                                    saved_hashes.add(img_hash)
                                    generated += 1
                                    self.progress_signal.emit(generated, total_to_generate)
                                    success = True
                                    break
                            except Exception as e:
                                self.log(f"Ошибка аугментации для {base_name} (попытка {attempt+1}): {e}")
                                continue
                        if not success:
                            self.log(f"Не удалось создать уникальную копию {base_name}_aug{copy_idx} после {max_attempts} попыток")
                        copy_idx += 1

            elif self.dataset_type == 0:  # Detection – используем аннотации
                img_path, label_path = pair
                img = cv2.imread(str(img_path))
                if img is None:
                    self.log(f"Предупреждение: не удалось прочитать {img_path}")
                    continue
                img_h, img_w = img.shape[:2]
                annotations = load_annotations(label_path, img_w, img_h)

                # Применяем маппинг классов (original -> new) и отбрасываем ненужные
                annotations = self._apply_class_mapping(annotations)
                if not annotations:
                    self.log(f"Нет аннотаций после маппинга классов в {label_path}")
                    continue

                # --- Оригинал (без аугментаций) ---
                if self.target_size is not None:
                    if self.resize_mode == "fixed":
                        img_orig, ann_orig = resize_image_and_annotations(img, annotations, self.target_size, self.bg_color)
                    elif self.resize_mode == "stretch":
                        img_orig, ann_orig = resize_image_and_annotations_stretch(img, annotations, self.target_size)
                    else:
                        img_orig, ann_orig = img.copy(), annotations.copy()
                else:
                    img_orig, ann_orig = img.copy(), annotations.copy()

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1]

                out_img_name = f"{base_name}{ext}"
                out_img_path = temp_images / out_img_name
                cv2.imwrite(str(out_img_path), img_orig)
                out_label_name = f"{base_name}.txt"
                out_label_path = temp_labels / out_label_name
                save_annotations(ann_orig, str(out_label_path), img_orig.shape[1], img_orig.shape[0])
                all_pairs.append((str(out_img_path), str(out_label_path), None))
                generated += 1
                self.progress_signal.emit(generated, total_to_generate)

                # --- Аугментированные копии ---
                if self.has_augmentations:
                    saved_hashes = {self._compute_image_hash(img_orig)}
                    copy_idx = 1
                    while copy_idx < self.aug_multiplier:
                        if self._is_cancelled:
                            return None
                        max_attempts = 10
                        success = False
                        for attempt in range(max_attempts):
                            try:
                                # Начинаем с копии оригиналов
                                aug_img = img.copy()
                                aug_anns = annotations.copy()
                                # Применяем простые аугментации (если включены)
                                if self.simple_aug_needed:
                                    aug_img, aug_anns = apply_simple_augmentations(aug_img, aug_anns,
                                                                                   self.augmentations)
                                # Применяем сложные аугментации (если включены)
                                if self.complex_aug_needed:
                                    aug_img, aug_anns = apply_complex_augmentations(aug_img, aug_anns,
                                                                                    self.augmentations,
                                                                                    border_color=self.bg_color)

                                if not aug_anns:
                                    continue

                                if self.target_size is not None:
                                    if self.resize_mode == "fixed":
                                        aug_img, aug_anns = resize_image_and_annotations(aug_img, aug_anns, self.target_size, self.bg_color)
                                    elif self.resize_mode == "stretch":
                                        aug_img, aug_anns = resize_image_and_annotations_stretch(aug_img, aug_anns, self.target_size)

                                img_hash = self._compute_image_hash(aug_img)
                                if img_hash not in saved_hashes:
                                    suffix = f"_aug{copy_idx}"
                                    out_img_name2 = f"{base_name}{suffix}{ext}"
                                    out_img_path2 = temp_images / out_img_name2
                                    cv2.imwrite(str(out_img_path2), aug_img)
                                    out_label_name2 = f"{base_name}{suffix}.txt"
                                    out_label_path2 = temp_labels / out_label_name2
                                    save_annotations(aug_anns, str(out_label_path2), aug_img.shape[1], aug_img.shape[0])
                                    all_pairs.append((str(out_img_path2), str(out_label_path2), None))
                                    saved_hashes.add(img_hash)
                                    generated += 1
                                    self.progress_signal.emit(generated, total_to_generate)
                                    success = True
                                    break
                            except Exception as e:
                                self.log(f"Ошибка аугментации для {base_name} (попытка {attempt+1}): {e}")
                                continue
                        if not success:
                            self.log(f"Не удалось создать уникальную копию {base_name}_aug{copy_idx} после {max_attempts} попыток")
                        copy_idx += 1

            else:  # self.dataset_type == 2: Только изображения
                img_path = pair[0]  # pair = (img_path, None)
                img = cv2.imread(str(img_path))
                if img is None:
                    self.log(f"Предупреждение: не удалось прочитать {img_path}")
                    continue

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1]

                # --- Оригинал (только ресайз, без аннотаций) ---
                if self.target_size is not None:
                    if self.resize_mode == "fixed":
                        img_orig, _ = resize_image_and_annotations(img, [], self.target_size, self.bg_color)
                    elif self.resize_mode == "stretch":
                        img_orig = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_orig = img.copy()
                else:
                    img_orig = img.copy()

                out_img_name = f"{base_name}{ext}"
                out_img_path = temp_images / out_img_name
                cv2.imwrite(str(out_img_path), img_orig)
                all_pairs.append((str(out_img_path), None, None))
                generated += 1
                self.progress_signal.emit(generated, total_to_generate)

                # --- Аугментированные копии ---
                if self.has_augmentations:
                    saved_hashes = {self._compute_image_hash(img_orig)}
                    copy_idx = 1
                    while copy_idx < self.aug_multiplier:
                        if self._is_cancelled:
                            return None
                        max_attempts = 10
                        success = False
                        for attempt in range(max_attempts):
                            try:
                                aug_img = img.copy()
                                # Применяем простые аугментации (если включены) – с пустым списком аннотаций
                                if self.simple_aug_needed:
                                    # Используем apply_simple_augmentations с пустыми аннотациями
                                    aug_img, _ = apply_simple_augmentations(aug_img, [], self.augmentations)
                                # Применяем сложные аугментации – тоже с пустыми аннотациями
                                if self.complex_aug_needed:
                                    aug_img, _ = apply_complex_augmentations(aug_img, [], self.augmentations,
                                                                              border_color=self.bg_color)

                                if self.target_size is not None:
                                    if self.resize_mode == "fixed":
                                        aug_img, _ = resize_image_and_annotations(aug_img, [], self.target_size, self.bg_color)
                                    elif self.resize_mode == "stretch":
                                        aug_img = cv2.resize(aug_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

                                img_hash = self._compute_image_hash(aug_img)
                                if img_hash not in saved_hashes:
                                    suffix = f"_aug{copy_idx}"
                                    out_img_name2 = f"{base_name}{suffix}{ext}"
                                    out_img_path2 = temp_images / out_img_name2
                                    cv2.imwrite(str(out_img_path2), aug_img)
                                    all_pairs.append((str(out_img_path2), None, None))
                                    saved_hashes.add(img_hash)
                                    generated += 1
                                    self.progress_signal.emit(generated, total_to_generate)
                                    success = True
                                    break
                            except Exception as e:
                                self.log(f"Ошибка аугментации для {base_name} (попытка {attempt+1}): {e}")
                                continue
                        if not success:
                            self.log(f"Не удалось создать уникальную копию {base_name}_aug{copy_idx} после {max_attempts} попыток")
                        copy_idx += 1

        # Для сегментации: конвертируем маски в YOLO-полигоны
        if self.dataset_type == 1 and all_pairs:
            masks_dir = str(temp_masks)
            labels_dir = str(temp_labels)
            self.log("[1/5] Проверка ULTRALYTICS_AVAILABLE = " + str(ULTRALYTICS_AVAILABLE))
            if not ULTRALYTICS_AVAILABLE:
                self.log("ОШИБКА: Ultralytics не установлен. Конвертация масок невозможна.")
                return []
            self.log("[2/5] Конвертация масок в YOLO-полигоны (manual mapping)...")
            try:
                pixel_to_class = {}
                for old_id in self.included_orig_ids:
                    if old_id == 0:
                        continue
                    new_id = self.mask_class_remap.get(old_id)
                    if new_id is not None:
                        pixel_to_class[old_id] = new_id
                    else:
                        self.log(f"Предупреждение: для old_id {old_id} нет remap, пропускаем")
                if not pixel_to_class:
                    self.log("ОШИБКА: нет отображения для объектов (фон исключён). Конвертация прервана.")
                    return []
                self.log(f"  Отображение пикселей: {pixel_to_class}")
                convert_segment_masks_to_yolo_seg_manual(masks_dir, labels_dir, pixel_to_class)
                self.log("  Конвертация завершена.")
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                self.log(f"  Создано {len(label_files)} .txt файлов в {labels_dir}")
            except Exception as e:
                self.log(f"  КРИТИЧЕСКАЯ ОШИБКА при конвертации:\n{traceback.format_exc()}")
                return []
            # Связываем метки с парами
            new_pairs = []
            for img_p, _, mask_p in all_pairs:
                mask_stem = os.path.splitext(os.path.basename(mask_p))[0]
                label_p = os.path.join(labels_dir, mask_stem + '.txt')
                if os.path.exists(label_p):
                    new_pairs.append((img_p, label_p, mask_p))
                else:
                    self.log(f"  Предупреждение: не найден label для {mask_p}")
                    new_pairs.append((img_p, None, mask_p))
            all_pairs = new_pairs
            self.log(f"Итоговое количество пар после конвертации: {len(all_pairs)}")

        return all_pairs

    def _split_and_move(self, output_path, all_pairs):
        random.shuffle(all_pairs)
        total = len(all_pairs)
        train_count = int(total * self.train_pct)
        val_count = int(total * self.val_pct)
        test_count = total - train_count - val_count

        train_pairs = all_pairs[:train_count]
        val_pairs = all_pairs[train_count:train_count + val_count]
        test_pairs = all_pairs[train_count + val_count:]

        self.log(f"Распределение: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

        def move_pairs(pairs, split):
            img_dst_dir = output_path / 'images' / split
            lbl_dst_dir = output_path / 'labels' / split
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            if self.dataset_type != 2:
                lbl_dst_dir.mkdir(parents=True, exist_ok=True)
            if self.dataset_type == 1:      # segmentation
                mask_dst_dir = output_path / 'masks' / split
                mask_dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path, lbl_path, mask_path in pairs:
                img_name = os.path.basename(img_path)
                shutil.move(img_path, img_dst_dir / img_name)
                if lbl_path:
                    lbl_name = os.path.basename(lbl_path)
                    shutil.move(lbl_path, lbl_dst_dir / lbl_name)
                if mask_path:
                    mask_name = os.path.basename(mask_path)
                    shutil.move(mask_path, mask_dst_dir / mask_name)

        move_pairs(train_pairs, 'train')
        move_pairs(val_pairs, 'val')
        move_pairs(test_pairs, 'test')
        shutil.rmtree(output_path / 'all', ignore_errors=True)

    def run(self):
        try:
            self.log("=== Начало генерации датасета ===")
            output_path = Path(self.output_folder)

            # Создаём файлы классов, только если есть классы (тип 0 или 1)
            if self.dataset_type != 2 and self.class_names:
                sorted_cids = sorted(self.class_names.keys())
                with open(output_path / 'classes.txt', 'w') as f:
                    for cid in sorted_cids:
                        f.write(self.class_names[cid] + '\n')
                self.log(f"Создан classes.txt с {len(self.class_names)} классами")

                categories = [{"id": new_id, "name": self.class_names[cid]}
                              for new_id, cid in enumerate(sorted_cids)]
                notes_data = {
                    "categories": categories,
                    "info": {"year": 2026, "version": "1.0", "contributor": "Dataset Preparation Tool"}
                }
                with open(output_path / 'notes.json', 'w') as f:
                    json.dump(notes_data, f, indent=2, ensure_ascii=False)

                data_yaml = {
                    'path': str(output_path.absolute()).replace('\\', '/'),
                    'train': 'images/train',
                    'val': 'images/val',
                    'test': 'images/test',
                    'nc': len(self.class_names),
                    'names': [self.class_names[cid] for cid in sorted_cids]
                }
                with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
                self.log("Создан data.yaml")
            else:
                self.log("Датасет без аннотаций – файлы классов не создаются.")

            all_pairs = self._generate_all_pairs(output_path)
            if all_pairs is None:
                self.finished_signal.emit(False, "Генерация прервана пользователем.")
                return
            if not all_pairs:
                self.log("Ошибка: не удалось сгенерировать ни одного изображения.")
                self.finished_signal.emit(False, "Нет данных для датасета.")
                return

            self._split_and_move(output_path, all_pairs)
            self.log("Генерация датасета завершена успешно!")
            self.finished_signal.emit(True, "Датасет успешно создан")
        except Exception as e:
            self.log(f"Критическая ошибка: {e}")
            self.finished_signal.emit(False, str(e))