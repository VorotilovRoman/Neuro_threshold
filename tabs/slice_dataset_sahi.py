# slice_dataset_sahi.py
from import_libs_internal import *
from import_libs_methods_ui import setup_sahi_slicer_ui

import io
import json
import os
import random
import shutil
import sys
import traceback
from contextlib import redirect_stdout

try:
    from sahi.slicing import slice_coco
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


# ==================================================================
# Автодетект датасета в папке
# ==================================================================
def detect_dataset(folder: str):
    """Ищет COCO JSON и папку с изображениями.

    Возвращает (coco_json_path, images_dir) или (None, None).
    """
    coco_path = None
    images_dir = None

    # 1. COCO JSON (рекурсивно)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if not f.lower().endswith('.json'):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                if (isinstance(data, dict)
                        and 'images' in data
                        and 'annotations' in data
                        and 'categories' in data):
                    coco_path = p
                    break
            except Exception:
                continue
        if coco_path:
            break

    if not coco_path:
        return None, None

    # 2. Папка с изображениями
    coco_dir = os.path.dirname(coco_path)
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for candidate in [
        os.path.join(coco_dir, 'images'),
        os.path.join(folder, 'images'),
        coco_dir,
    ]:
        if not os.path.isdir(candidate):
            continue
        try:
            for entry in os.listdir(candidate):
                if entry.lower().endswith(img_exts):
                    images_dir = candidate
                    break
        except Exception:
            pass
        if images_dir:
            break

    # 3. Fallback через file_name в COCO
    if not images_dir:
        try:
            with open(coco_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if data.get('images'):
                fname = data['images'][0].get('file_name', '')
                if fname:
                    for base in [coco_dir, folder]:
                        candidate = os.path.join(base, fname)
                        if os.path.exists(candidate):
                            images_dir = os.path.dirname(candidate)
                            break
        except Exception:
            pass

    return coco_path, images_dir


# ==================================================================
# Нарезка
# ==================================================================
def slice_dataset(
    coco_json_path: str,
    image_dir: str,
    output_dir: str,
    slice_size: int = 640,
    overlap: float = 0.2,
    min_area_ratio: float = 0.3,
):
    if not SAHI_AVAILABLE:
        raise RuntimeError(
            "SAHI не установлен. Установите: pip install sahi")

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔪 Нарезка {os.path.basename(coco_json_path)}")
    print(f"   тайл: {slice_size}×{slice_size}, "
          f"overlap={overlap:.0%}, "
          f"min_area_ratio={min_area_ratio}")

    coco_dict, output_coco_path = slice_coco(
        coco_annotation_file_path=coco_json_path,
        image_dir=image_dir,
        output_dir=output_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        min_area_ratio=min_area_ratio,
        verbose=False,
    )

    n_images = len(coco_dict.get("images", []))
    n_anns = len(coco_dict.get("annotations", []))
    print(f"   ✅ Получено тайлов: {n_images}, аннотаций: {n_anns}")
    return coco_dict, output_coco_path


# ==================================================================
# Субдискретизация фоновых тайлов
# ==================================================================
def subsample_negative_tiles(
    coco_dict: dict,
    coco_json_path: str,
    image_dir: str,
    keep_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    if not (0 < keep_ratio <= 1):
        return coco_dict

    ann_by_img = {}
    for ann in coco_dict.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], 0)
        ann_by_img[ann["image_id"]] += 1

    with_objs = [img for img in coco_dict["images"]
                 if ann_by_img.get(img["id"], 0) > 0]
    without_objs = [img for img in coco_dict["images"]
                    if ann_by_img.get(img["id"], 0) == 0]

    if not without_objs:
        print("   ⚠️ Фоновых тайлов нет — субдискретизация не требуется.")
        return coco_dict

    random.seed(seed)
    keep_count = max(1, int(len(without_objs) * keep_ratio))
    kept_negative = random.sample(without_objs, keep_count)
    dropped_negative = [img for img in without_objs
                        if img not in kept_negative]

    for img in dropped_negative:
        fname = img.get("file_name")
        if fname:
            path = os.path.join(image_dir, fname)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"   ⚠️ Не удалось удалить {path}: {e}")

    kept_ids = {img["id"] for img in with_objs + kept_negative}
    coco_dict["images"] = [img for img in coco_dict["images"]
                           if img["id"] in kept_ids]
    coco_dict["annotations"] = [ann for ann in coco_dict["annotations"]
                                if ann["image_id"] in kept_ids]

    print(f"   📉 Фоновых тайлов: {len(without_objs)} → {keep_count} "
          f"({keep_ratio:.0%}), удалено файлов: {len(dropped_negative)}")
    print(f"   Итого тайлов: {len(coco_dict['images'])} "
          f"(с объектами: {len(with_objs)}, фоновых: {keep_count})")

    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False)
    return coco_dict


# ==================================================================
# Train / val split
# ==================================================================
def split_train_val(
    coco_dict: dict,
    coco_json_path: str,
    image_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    if not (0 < train_ratio < 1):
        return (coco_dict,
                {"images": [], "annotations": [],
                 "categories": coco_dict.get("categories", [])},
                coco_json_path, None)

    random.seed(seed)
    images = list(coco_dict["images"])
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    anns = coco_dict.get("annotations", [])
    cats = coco_dict.get("categories", [])

    train_dict = {
        "images": train_images,
        "annotations": [a for a in anns if a["image_id"] in train_ids],
        "categories": cats,
    }
    val_dict = {
        "images": val_images,
        "annotations": [a for a in anns if a["image_id"] in val_ids],
        "categories": cats,
    }

    base_out = os.path.dirname(os.path.dirname(coco_json_path))
    for split_name, split_dict in [("train", train_dict), ("val", val_dict)]:
        split_img_dir = os.path.join(base_out, "images", split_name)
        os.makedirs(split_img_dir, exist_ok=True)
        for img in split_dict["images"]:
            fname = img.get("file_name")
            if not fname:
                continue
            src = os.path.join(image_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(split_img_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                img["file_name"] = os.path.join(split_name, fname)

    os.makedirs(os.path.join(base_out, "annotations"), exist_ok=True)
    train_json = os.path.join(base_out, "annotations", "train.json")
    val_json = os.path.join(base_out, "annotations", "val.json")

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_dict, f, ensure_ascii=False)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_dict, f, ensure_ascii=False)

    print(f"   ✂️ Разделение: train={len(train_images)}, "
          f"val={len(val_images)}")
    return train_dict, val_dict, train_json, val_json


# ==================================================================
# COCO → YOLO
# ==================================================================
def coco_to_yolo(
    coco_json_path: str,
    image_dir: str,
    labels_dir: str,
    class_names: list = None,
    verbose: bool = True,
):
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    if class_names is None:
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    else:
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        cat_id_to_idx = {}
        for c in categories:
            cname = c.get("name", str(c["id"]))
            cat_id_to_idx[c["id"]] = name_to_idx.get(cname, 0)

    img_meta = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_written = 0
    for img_id, img in img_meta.items():
        w, h = float(img["width"]), float(img["height"])
        if w <= 0 or h <= 0:
            continue

        fname = img.get("file_name", "")
        base = os.path.splitext(os.path.basename(fname))[0]
        txt_path = os.path.join(labels_dir, base + ".txt")

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cls_idx = cat_id_to_idx.get(ann["category_id"])
            if cls_idx is None:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} "
                         f"{nw:.6f} {nh:.6f}")
            n_written += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if verbose:
        print(f"   📝 YOLO labels: {n_written} аннотаций в {labels_dir}")


# ==================================================================
# data.yaml
# ==================================================================
def write_yolo_yaml(
    output_dir: str,
    class_names: list,
    channels: int = 3,
):
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"channels: {channels}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    print(f"   📄 data.yaml → {yaml_path}")


# ==================================================================
# Визуализация
# ==================================================================
def visualize_samples(
    coco_json_path: str,
    image_dir: str,
    n: int = 3,
):
    try:
        import cv2
    except ImportError:
        print("   ⚠️ Для визуализации нужен opencv-python.")
        return

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    vis_dir = os.path.join(os.path.dirname(coco_json_path),
                           "..", "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    random.seed(42)
    samples = random.sample(coco["images"],
                            min(n, len(coco["images"])))

    for img in samples:
        fname = img.get("file_name")
        src = os.path.join(image_dir, fname)
        if not os.path.exists(src):
            continue
        canvas = cv2.imread(src)
        if canvas is None:
            continue
        for ann in anns_by_img.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(canvas, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        dst = os.path.join(vis_dir, os.path.basename(fname))
        cv2.imwrite(dst, canvas)

    print(f"   🖼️ Визуализация: {len(samples)} тайлов → {vis_dir}")


# ==================================================================
# Worker-поток
# ==================================================================
class SliceWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._pipeline()
            self.finished_signal.emit(True, "Готово")
        except Exception as e:
            self.log_signal.emit(f"❌ Ошибка: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

    def _pipeline(self):
        p = self.params

        class _Emitter:
            def __init__(self, sig):
                self.sig = sig

            def write(self, s):
                if not s:
                    return
                for line in s.split('\n'):
                    if line.strip():
                        self.sig.emit(line)

            def flush(self):
                pass

        with redirect_stdout(_Emitter(self.log_signal)):
            self.progress_signal.emit(5, "Нарезка...")
            coco_dict, sliced_json = slice_dataset(
                coco_json_path=p['coco'],
                image_dir=p['images'],
                output_dir=p['out'],
                slice_size=p['slice_size'],
                overlap=p['overlap'],
                min_area_ratio=p['min_area_ratio'],
            )
            if not coco_dict.get('images'):
                raise RuntimeError(
                    "После нарезки не осталось изображений.")

            if p['keep_negative'] < 1.0:
                self.progress_signal.emit(40, "Субдискретизация фона...")
                coco_dict = subsample_negative_tiles(
                    coco_dict=coco_dict,
                    coco_json_path=sliced_json,
                    image_dir=os.path.dirname(sliced_json),
                    keep_ratio=p['keep_negative'],
                    seed=p['seed'],
                )

            train_json = val_json = None
            final_train_dir = final_val_dir = None
            if 0 < p['split'] < 1:
                self.progress_signal.emit(50, "Разделение train/val...")
                _, _, train_json, val_json = split_train_val(
                    coco_dict=coco_dict,
                    coco_json_path=sliced_json,
                    image_dir=os.path.dirname(sliced_json),
                    train_ratio=p['split'],
                    seed=p['seed'],
                )
                final_train_dir = os.path.join(p['out'],
                                               'images', 'train')
                final_val_dir = os.path.join(p['out'],
                                             'images', 'val')

            if p['format'] in ('yolo', 'both'):
                self.progress_signal.emit(60, "Конвертация в YOLO...")
                cats = sorted(coco_dict.get('categories', []),
                              key=lambda c: c['id'])
                class_names = [c.get('name', str(c['id']))
                               for c in cats]

                if train_json is not None:
                    coco_to_yolo(
                        coco_json_path=train_json,
                        image_dir=final_train_dir,
                        labels_dir=os.path.join(p['out'],
                                                'labels', 'train'),
                        class_names=class_names,
                    )
                    coco_to_yolo(
                        coco_json_path=val_json,
                        image_dir=final_val_dir,
                        labels_dir=os.path.join(p['out'],
                                                'labels', 'val'),
                        class_names=class_names,
                    )
                else:
                    coco_to_yolo(
                        coco_json_path=sliced_json,
                        image_dir=os.path.dirname(sliced_json),
                        labels_dir=os.path.join(p['out'], 'labels'),
                        class_names=class_names,
                    )

                write_yolo_yaml(
                    output_dir=p['out'],
                    class_names=class_names,
                    channels=p['channels'],
                )

            if p['visualize'] > 0:
                self.progress_signal.emit(85, "Визуализация...")
                if train_json is not None:
                    visualize_samples(train_json, final_train_dir,
                                      n=p['visualize'])
                else:
                    visualize_samples(sliced_json,
                                      os.path.dirname(sliced_json),
                                      n=p['visualize'])

        self.progress_signal.emit(100, "Готово")


# ==================================================================
# Главное окно
# ==================================================================
class SahiSlicerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAHI Dataset Slicer")
        self.setMinimumWidth(800)
        self.setMinimumHeight(720)

        setup_sahi_slicer_ui(self)

        # --- Состояние ---
        self.coco_path = None
        self.images_path = None
        self.output_path = None
        self.worker = None

        # --- Инициализация UI ---
        self.log("SAHI Slicer готов к работе.")
        if not SAHI_AVAILABLE:
            self.log("⚠️ SAHI не установлен! Установите: pip install sahi")
            self.btn_generate.setEnabled(False)

        self._update_generate_state()

    # ------------------------------------------------------------------
    # Логирование
    # ------------------------------------------------------------------
    def log(self, message):
        try:
            self.log_view.append(message)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
            QApplication.processEvents()
        except Exception:
            print(f"[log] {message}")

    def toggle_log(self, checked):
        self.log_view.setVisible(bool(checked))
        self.btn_toggle_log.setText(
            "Скрыть лог" if checked else "Показать лог")

    # ------------------------------------------------------------------
    # Выбор папок
    # ------------------------------------------------------------------
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с датасетом (COCO JSON + images)")
        if not folder:
            return
        self.dataset_edit.setText(folder)
        coco, images = detect_dataset(folder)
        self.coco_path = coco
        self.images_path = images

        if coco and images:
            self.coco_label.setText(coco)
            self.images_label.setText(images)
            self.coco_label.setStyleSheet("color: #2e7d32;")
            self.images_label.setStyleSheet("color: #2e7d32;")
            self.log(f"✅ COCO JSON: {coco}")
            self.log(f"✅ Images:    {images}")
        else:
            self.coco_label.setText(
                coco or "❌ COCO JSON не найден")
            self.images_label.setText(
                images or "❌ папка с изображениями не найдена")
            self.coco_label.setStyleSheet(
                "color: #2e7d32;" if coco else "color: #c62828;")
            self.images_label.setStyleSheet(
                "color: #2e7d32;" if images else "color: #c62828;")
            if not coco:
                self.log("⚠️ COCO JSON не найден в выбранной папке.")
            if not images:
                self.log("⚠️ Папка с изображениями не найдена.")

        self._update_generate_state()

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку для нового датасета")
        if not folder:
            return
        self.output_edit.setText(folder)
        self.output_path = folder
        self.btn_open_output.setEnabled(True)
        self.log(f"📁 Output: {folder}")
        self._update_generate_state()

    def open_output_folder(self):
        if not self.output_path or not os.path.isdir(self.output_path):
            return
        if sys.platform == 'win32':
            os.startfile(self.output_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{self.output_path}"')
        else:
            os.system(f'xdg-open "{self.output_path}"')

    def _update_generate_state(self):
        can_run = bool(
            SAHI_AVAILABLE
            and self.coco_path
            and self.images_path
            and self.output_path)
        self.btn_generate.setEnabled(can_run and not self._is_running())

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------
    def on_generate(self):
        if not (self.coco_path and self.images_path and self.output_path):
            return

        if not SAHI_AVAILABLE:
            QMessageBox.critical(
                self, "SAHI не установлен",
                "Установите SAHI:\n\n    pip install sahi")
            return

        if os.path.exists(self.output_path) and os.listdir(self.output_path):
            reply = QMessageBox.question(
                self, "Папка не пуста",
                f"Папка:\n{self.output_path}\n\nне пуста. "
                f"Продолжить (существующие файлы могут быть "
                f"перезаписаны)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.log("")
        self.log("═" * 60)
        self.log(f"📁 Input:   {self.coco_path}")
        self.log(f"📁 Images:  {self.images_path}")
        self.log(f"📁 Output:  {self.output_path}")

        params = {
            'coco': self.coco_path,
            'images': self.images_path,
            'out': self.output_path,
            'slice_size': self.slice_size.value(),
            'overlap': self.overlap.value(),
            'min_area_ratio': self.min_area_ratio.value(),
            'keep_negative': self.keep_negative.value(),
            'split': self.split_ratio.value(),
            'format': self.format_combo.currentData(),
            'channels': self.channels_combo.currentData(),
            'seed': self.seed_spin.value(),
            'visualize': self.visualize_spin.value(),
        }

        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск...")

        self.worker = SliceWorker(params)
        self.worker.log_signal.connect(self._on_log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    # ------------------------------------------------------------------
    # Сигналы worker
    # ------------------------------------------------------------------
    def _on_log(self, line):
        self.log_view.append(line)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, percent, msg):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    def _on_finished(self, ok, msg):
        self.progress_bar.setVisible(False)
        self._update_generate_state()

        if ok:
            self.status_label.setText("✅ Готово")
            reply = QMessageBox.information(
                self, "Успех",
                f"Датасет создан в:\n{self.output_path}\n\n"
                f"Открыть папку?",
                QMessageBox.Open | QMessageBox.Close,
                QMessageBox.Open)
            if reply == QMessageBox.Open:
                self.open_output_folder()
        else:
            self.status_label.setText("❌ Ошибка")
            QMessageBox.critical(
                self, "Ошибка",
                f"Не удалось создать датасет:\n{msg}\n\n"
                f"Подробности в логе.")


# ==================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SahiSlicerWindow()
    window.show()
    sys.exit(app.exec_())