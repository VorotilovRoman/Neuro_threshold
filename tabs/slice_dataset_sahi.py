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
import tempfile
import copy
from contextlib import redirect_stdout

try:
    from sahi.slicing import slice_coco

    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


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

    try:
        coco_dict, output_coco_path = slice_coco(
            coco_annotation_file_path=coco_json_path,
            image_dir=image_dir,
            output_coco_annotation_file_name="sliced_coco.json",
            output_dir=output_dir,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            min_area_ratio=min_area_ratio,
            ignore_negative_samples=False,
            verbose=False,
        )
    except TypeError:
        coco_dict, output_coco_path = slice_coco(
            coco_json_path,
            image_dir,
            "sliced_coco.json",
            output_dir,
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
# Определение формата исходного датасета
# ==================================================================
def detect_dataset_format(coco_dict: dict) -> str:
    """
    Возвращает 'segmentation' если в аннотациях присутствует segmentation,
    иначе 'bbox'.
    """
    for ann in coco_dict.get("annotations", []):
        if ann.get("segmentation"):
            return "segmentation"
    return "bbox"


def describe_segmentation(coco_dict: dict) -> str:
    """Возвращает человекочитаемую подсказку о сегментах."""
    seg_total = 0
    seg_4pt = 0
    for ann in coco_dict.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg:
            continue
        try:
            flat = seg[0] if isinstance(seg[0], (list, tuple)) else seg
            npts = len(flat) // 2
        except Exception:
            continue
        seg_total += 1
        if npts == 4:
            seg_4pt += 1

    if seg_total == 0:
        return ""
    if seg_4pt == seg_total:
        return f"OBB ({seg_total} сегментов, все 4-точечные)"
    return f"Segmentation ({seg_4pt}/{seg_total} сегментов 4-точечные)"


# ==================================================================
# Субдискретизация фоновых тайлов
# ==================================================================
def subsample_negative_tiles(
        coco_dict: dict,
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
            path = fname if os.path.isabs(fname) else os.path.join(image_dir, fname)
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

    return coco_dict


# ==================================================================
# Разбиение на Train / Val / Test и перемещение файлов
# ==================================================================
def split_and_move(coco_dict, output_dir, train_ratio=0.8, seed=42):
    """Разбивает картинки по папкам, обновляет пути в едином COCO JSON.
    Сами аннотации не трогаются — в sliced_coco.json остаётся
    исходная разметка от SAHI (bbox + segmentation, как SAHI вернул)."""
    images = list(coco_dict["images"])
    has_val = 0 < train_ratio < 1

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    if has_val:
        random.seed(seed)
        random.shuffle(images)
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        val_images = images[n_train:]
    else:
        train_images = images
        val_images = []

    test_images = []

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    for img in images:
        old_fname = img["file_name"]
        base_name = os.path.basename(old_fname)

        if img["id"] in train_ids:
            split_folder = "train"
        elif img["id"] in val_ids:
            split_folder = "val"
        else:
            split_folder = "test"

        src = os.path.join(output_dir, base_name)
        if not os.path.exists(src) and os.path.exists(old_fname):
            src = old_fname

        dst = os.path.join(output_dir, 'images', split_folder, base_name)
        if os.path.exists(src):
            shutil.move(src, dst)

        img["file_name"] = f"images/{split_folder}/{base_name}"

    final_json_path = os.path.join(output_dir, "sliced_coco.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    print(f"   ✂️ Разделение и перемещение: train={len(train_images)}, "
          f"val={len(val_images)}, test=0")
    return coco_dict, final_json_path


# ==================================================================
# COCO → YOLO (bbox / segmentation)
# ==================================================================
def coco_to_yolo(
        coco_json_path: str,
        output_dir: str,
        class_names: list = None,
        annotation_format: str = "bbox",
        verbose: bool = True,
):
    """
    Конвертирует COCO JSON в YOLO.
    annotation_format:
        • 'bbox'         — HBB:      class cx cy w h
        • 'segmentation' — Полигон:  class x1 y1 x2 y2 ... xn yn

    Каждая аннотация пишется в выбранном формате, если поле есть.
    Если нужного поля нет, но есть второе — используется оно.
    Если нет ни того, ни другого — аннотация пропускается.
    """
    if annotation_format not in ("bbox", "segmentation"):
        raise ValueError(
            f"annotation_format должен быть 'bbox' или 'segmentation', "
            f"получено: {annotation_format}")

    np = None
    if annotation_format == "segmentation":
        try:
            import numpy as np  # noqa: F811
        except ImportError:
            raise RuntimeError(
                "Для формата 'segmentation' нужен numpy "
                "(pip install numpy)")

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

    n_bbox = 0
    n_seg = 0
    n_skipped = 0
    seg_dist = {}  # n_points -> count

    prefer_seg = (annotation_format == "segmentation")

    for img_id, img in img_meta.items():
        w, h = float(img["width"]), float(img["height"])
        if w <= 0 or h <= 0:
            continue

        fname = img.get("file_name", "").replace("\\", "/")
        parts = fname.split("/")
        split_name = "train"
        if len(parts) >= 2 and parts[0] == "images":
            split_name = parts[1]

        base = os.path.splitext(os.path.basename(fname))[0]
        labels_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(labels_dir, exist_ok=True)
        txt_path = os.path.join(labels_dir, base + ".txt")

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cls_idx = cat_id_to_idx.get(ann["category_id"])
            if cls_idx is None:
                n_skipped += 1
                continue

            bbox = ann.get("bbox")
            seg = ann.get("segmentation")
            has_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4
            has_seg = bool(seg)

            write_bbox = False
            write_seg = False

            if prefer_seg:
                if has_seg:
                    write_seg = True
                elif has_bbox:
                    write_bbox = True
            else:
                if has_bbox:
                    write_bbox = True
                elif has_seg:
                    write_seg = True

            if write_bbox:
                x, y, bw, bh = bbox
                cx = max(0.0, min(1.0, (x + bw / 2) / w))
                cy = max(0.0, min(1.0, (y + bh / 2) / h))
                nw = max(0.0, min(1.0, bw / w))
                nh = max(0.0, min(1.0, bh / h))
                if nw <= 0 or nh <= 0:
                    n_skipped += 1
                    continue
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_bbox += 1
                continue

            if write_seg:
                try:
                    flat = seg[0] if isinstance(seg[0], (list, tuple)) else seg
                    pts = np.array(flat, dtype=np.float32).reshape(-1, 2)
                except Exception:
                    n_skipped += 1
                    continue
                if len(pts) < 3:
                    n_skipped += 1
                    continue

                npts = len(pts)
                seg_dist[npts] = seg_dist.get(npts, 0) + 1

                pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
                pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
                coords = " ".join(f"{v:.6f}" for v in pts.flatten())
                lines.append(f"{cls_idx} {coords}")
                n_seg += 1
                continue

            n_skipped += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if verbose:
        print(f"   📝 YOLO [{annotation_format}]: "
              f"bbox={n_bbox}, seg={n_seg}, пропущено={n_skipped}")
        if seg_dist:
            dist_str = ", ".join(
                f"{k} точек: {v}" for k, v in sorted(seg_dist.items()))
            print(f"      распределение сегментов: {dist_str}")
            non4 = sum(v for k, v in seg_dist.items() if k != 4)
            if non4 > 0:
                print(f"      ⚠️ {non4} сегментов имеют не 4 точки. "
                      f"Это ОК для YOLO-seg, но YOLO-OBB требует ровно "
                      f"4 точки — такие объекты не подойдут для OBB-модели.")


# ==================================================================
# data.yaml
# ==================================================================
def write_yolo_yaml(output_dir, class_names, channels=3):
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("names:\n")
        for name in class_names:
            f.write(f"- '{name}'\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"channels: {channels}\n")
    print(f"   📄 data.yaml → {yaml_path}")


# ==================================================================
# Визуализация
# ==================================================================
def visualize_samples(coco_json_path, output_dir, n=3):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("   ⚠️ Для визуализации нужны opencv-python и numpy.")
        return

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    random.seed(42)
    samples = random.sample(coco["images"], min(n, len(coco["images"])))

    for img in samples:
        fname = img.get("file_name", "").replace("\\", "/")
        src = os.path.join(output_dir, fname)
        if not os.path.exists(src):
            continue
        canvas = cv2.imread(src)
        if canvas is None:
            continue
        for ann in anns_by_img.get(img["id"], []):
            seg = ann.get("segmentation")
            if seg:
                try:
                    flat = seg[0] if isinstance(seg[0], (list, tuple)) else seg
                    pts = np.array(flat, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(canvas, [pts], isClosed=True,
                                  color=(0, 255, 0), thickness=2)
                    continue
                except Exception:
                    pass
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh),
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
                raise RuntimeError("После нарезки не осталось изображений.")

            if p['keep_negative'] < 1.0:
                self.progress_signal.emit(30, "Субдискретизация фона...")
                coco_dict = subsample_negative_tiles(
                    coco_dict=coco_dict,
                    image_dir=p['out'],
                    keep_ratio=p['keep_negative'],
                    seed=p['seed'],
                )

            self.progress_signal.emit(50, "Разделение и перемещение файлов...")
            final_dict, final_json = split_and_move(
                coco_dict=coco_dict,
                output_dir=p['out'],
                train_ratio=p['split'],
                seed=p['seed'],
            )

            # --- ОЧИСТКА: удаляем промежуточный COCO-JSON, созданный SAHI ---
            if sliced_json and os.path.abspath(sliced_json) != os.path.abspath(final_json):
                if os.path.exists(sliced_json):
                    try:
                        os.remove(sliced_json)
                        self.log_signal.emit(
                            f"🧹 Удалён промежуточный файл SAHI: "
                            f"{os.path.basename(sliced_json)}")
                    except OSError as e:
                        self.log_signal.emit(
                            f"⚠️ Не удалось удалить {sliced_json}: {e}")

            if p['format'] in ('yolo', 'both'):
                self.progress_signal.emit(70, "Конвертация в YOLO...")
                cats = sorted(final_dict.get('categories', []),
                              key=lambda c: c['id'])
                class_names = [c.get('name', str(c['id'])) for c in cats]

                ann_format = p.get('ann_format', 'auto')
                if ann_format == 'auto':
                    ann_format = p.get('detected_format', 'bbox')
                if ann_format not in ('bbox', 'segmentation'):
                    ann_format = 'bbox'

                # Диагностика по sliced_coco.json
                n_anns = len(final_dict.get('annotations', []))
                n_with_seg = sum(1 for a in final_dict.get('annotations', [])
                                 if a.get('segmentation'))
                n_with_bbox = sum(1 for a in final_dict.get('annotations', [])
                                  if isinstance(a.get('bbox'), (list, tuple))
                                  and len(a['bbox']) == 4)
                self.log_signal.emit(
                    f"   📊 В sliced_coco.json: аннотаций={n_anns}, "
                    f"с bbox={n_with_bbox}, с segmentation={n_with_seg}")
                self.log_signal.emit(
                    f"   🏷️ Формат YOLO: {ann_format}")

                if ann_format == 'segmentation' and n_with_seg == 0:
                    self.log_signal.emit(
                        f"   ⚠️ Выбран формат 'segmentation', но в аннотациях "
                        f"нет segmentation — будет использован bbox.")

                coco_to_yolo(
                    coco_json_path=final_json,
                    output_dir=p['out'],
                    class_names=class_names,
                    annotation_format=ann_format,
                )

                write_yolo_yaml(
                    output_dir=p['out'],
                    class_names=class_names,
                    channels=p['channels'],
                )

            if p['visualize'] > 0:
                self.progress_signal.emit(85, "Визуализация...")
                visualize_samples(
                    final_json,
                    p['out'],
                    n=p['visualize']
                )

        self.progress_signal.emit(100, "Готово")


# ==================================================================
# Главное окно
# ==================================================================
class SahiSlicerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAHI Dataset Slicer")
        self.setMinimumWidth(800)
        self.setMinimumHeight(780)

        setup_sahi_slicer_ui(self)

        # --- Состояние ---
        self.coco_path = None
        self.images_path = None
        self.output_path = None
        self.worker = None

        self.coco_dict = None
        self._temp_coco_path = None
        self._loaded_json_paths = []
        self.detected_format = 'bbox'      # 'bbox' | 'segmentation'
        self.seg_hint = ""                 # человекочитаемая подсказка

        # --- Инициализация UI ---
        self.log("SAHI Slicer готов к работе.")
        if not SAHI_AVAILABLE:
            self.log("⚠️ SAHI не установлен! Установите: pip install sahi")
            self.btn_generate.setEnabled(False)

        self._update_generate_state()

    def closeEvent(self, event):
        if self._temp_coco_path and os.path.exists(self._temp_coco_path):
            try:
                os.remove(self._temp_coco_path)
            except Exception:
                pass
        super().closeEvent(event)

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
    # Загрузка COCO и Валидация
    # ------------------------------------------------------------------
    def load_coco(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите COCO JSON", "", "JSON (*.json)")
        if not path:
            return
        self._process_coco(path, is_add=False)

    def add_coco(self):
        if not self.coco_dict:
            self.load_coco()
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите COCO JSON для добавления", "", "JSON (*.json)")
        if not path:
            return
        self._process_coco(path, is_add=True)

    def _process_coco(self, json_path, is_add=False):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                new_dict = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось прочитать JSON:\n{e}")
            return

        if (not isinstance(new_dict, dict)
                or 'images' not in new_dict
                or 'annotations' not in new_dict):
            QMessageBox.warning(self, "Ошибка", "Файл не похож на COCO JSON.")
            return

        base_dir = os.path.dirname(json_path)
        img_dir = self._find_images_dir(new_dict, base_dir)

        if not img_dir:
            img_dir = QFileDialog.getExistingDirectory(
                self,
                "Не удалось автоопределить папку с изображениями. Выберите её:")
            if not img_dir:
                return

        valid_imgs = 0
        missing_imgs = 0
        for img in new_dict.get('images', []):
            fname = img.get('file_name', '')
            abs_path = os.path.join(img_dir, fname)
            if not os.path.exists(abs_path):
                abs_path = os.path.join(img_dir, os.path.basename(fname))

            if os.path.exists(abs_path):
                img['file_name'] = os.path.abspath(abs_path)
                valid_imgs += 1
            else:
                missing_imgs += 1

        if valid_imgs == 0:
            QMessageBox.warning(
                self, "Ошибка",
                f"В папке {img_dir} не найдено ни одного изображения "
                f"из JSON.")
            return

        dropped, fixed = self._sanitize_coco_dict(new_dict)

        if is_add and self.coco_dict:
            self.coco_dict = self._merge_coco(self.coco_dict, new_dict)
        else:
            self.coco_dict = new_dict

        self.images_path = img_dir
        self._loaded_json_paths.append(os.path.basename(json_path))

        # Автоопределение формата
        self.detected_format = detect_dataset_format(self.coco_dict)
        self.seg_hint = describe_segmentation(self.coco_dict)
        if self.detected_format == 'segmentation':
            self.log(f"   🏷️ Автоопределение: Segmentation — {self.seg_hint}")
        else:
            self.log(f"   🏷️ Автоопределение: HBB (bbox)")

        self._save_temp_coco()

        total_imgs = len(self.coco_dict.get('images', []))
        total_anns = len(self.coco_dict.get('annotations', []))

        self.coco_label.setText(
            f"Файлов JSON: {len(self._loaded_json_paths)} | "
            f"Изображений: {total_imgs} | Аннотаций: {total_anns}")
        self.images_label.setText(
            f"{img_dir} (Успешно: {valid_imgs}, Потеряно: {missing_imgs})")

        self.log(f"✅ Загружен: {json_path}")
        if missing_imgs > 0:
            self.log(f"   ⚠️ Внимание: {missing_imgs} изображений "
                     f"не найдены на диске и будут пропущены SAHI.")
        if dropped > 0 or fixed > 0:
            self.log(f"   ⚙️ Очистка аннотаций: удалено битых: {dropped}, "
                     f"исправлено нулевых area: {fixed}")

        self._update_generate_state()
        self._update_validation_ui()

    def _sanitize_coco_dict(self, coco_dict):
        valid_annotations = []
        dropped_count = 0
        fixed_area_count = 0

        for ann in coco_dict.get('annotations', []):
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                w, h = bbox[2], bbox[3]

                if w < 1.0 or h < 1.0:
                    dropped_count += 1
                    continue

                if ann.get('area', 0) <= 0:
                    ann['area'] = w * h
                    fixed_area_count += 1
            else:
                dropped_count += 1
                continue

            valid_annotations.append(ann)

        coco_dict['annotations'] = valid_annotations
        return dropped_count, fixed_area_count

    def _update_validation_ui(self):
        if not self.coco_dict:
            if hasattr(self, 'validation_summary_label'):
                self.validation_summary_label.setText("")
            if hasattr(self, 'color_stats_label'):
                self.color_stats_label.setText("")
            return

        images = self.coco_dict.get('images', [])
        total = len(images)

        import cv2
        import numpy as np

        real_gray = 0
        visual_gray = 0
        real_color = 0
        bit8_png = 0
        bit8_other = 0

        for img in images:
            p = img.get('file_name')
            if not p or not os.path.exists(p):
                continue

            ext = os.path.splitext(p)[1].lower()
            try:
                cv_img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if cv_img is None:
                    continue

                is_png = (ext == '.png')
                is_8bit = (cv_img.dtype == np.uint8)

                if is_8bit:
                    if is_png:
                        bit8_png += 1
                    else:
                        bit8_other += 1

                if len(cv_img.shape) == 2:
                    real_gray += 1
                    visual_gray += 1
                elif len(cv_img.shape) == 3:
                    if cv_img.shape[2] == 1:
                        real_gray += 1
                        visual_gray += 1
                    elif cv_img.shape[2] in (3, 4):
                        b, g, r = cv2.split(cv_img[:, :, :3])
                        if np.array_equal(b, g) and np.array_equal(g, r):
                            visual_gray += 1
                        else:
                            real_color += 1
            except Exception:
                pass

        if self.detected_format == 'segmentation':
            fmt_display = f"Segmentation — {self.seg_hint}"
        else:
            fmt_display = "HBB (bbox)"

        self.validation_summary_label.setText(
            f"Формат разметки: <b>{fmt_display}</b> | "
            f"Изображений: {total}")
        self.color_stats_label.setText(
            f"⚫ Реально ЧБ: {real_gray} &nbsp;|&nbsp; "
            f"🦇 Визуально ЧБ: {visual_gray} &nbsp;|&nbsp; "
            f"🤔 Реально цветных: {real_color}<br>"
            f"🔵 8-bit PNG: {bit8_png} &nbsp;|&nbsp; "
            f"🟡 8-bit non-PNG: {bit8_other}"
        )

    def _find_images_dir(self, coco_dict, base_dir):
        candidates = [
            os.path.join(base_dir, 'images'),
            os.path.join(os.path.dirname(base_dir), 'images'),
            os.path.join(base_dir, 'train', 'images'),
            os.path.join(base_dir, 'val', 'images'),
            base_dir
        ]

        test_imgs = [img.get('file_name')
                     for img in coco_dict.get('images', [])[:5]
                     if img.get('file_name')]
        if not test_imgs:
            return None

        for d in candidates:
            if not os.path.isdir(d):
                continue
            matches = 0
            for fname in test_imgs:
                if (os.path.exists(os.path.join(d, fname))
                        or os.path.exists(os.path.join(d, os.path.basename(fname)))):
                    matches += 1
            if matches > 0:
                return d
        return None

    def _save_temp_coco(self):
        if self._temp_coco_path and os.path.exists(self._temp_coco_path):
            try:
                os.remove(self._temp_coco_path)
            except Exception:
                pass

        fd, temp_path = tempfile.mkstemp(suffix='.json',
                                         prefix='sahi_coco_')
        os.close(fd)

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_dict, f, ensure_ascii=False)

        self.coco_path = temp_path
        self._temp_coco_path = temp_path

    def _merge_coco(self, base, new):
        base = copy.deepcopy(base)

        cat_map = {}
        next_cat_id = max([c['id'] for c in base.get('categories', [])] + [0]) + 1
        base_cats_by_name = {c['name']: c['id']
                             for c in base.get('categories', [])}

        for c in new.get('categories', []):
            name = c.get('name', str(c['id']))
            if name in base_cats_by_name:
                cat_map[c['id']] = base_cats_by_name[name]
            else:
                cat_map[c['id']] = next_cat_id
                base.setdefault('categories', []).append({
                    'id': next_cat_id,
                    'name': name,
                    'supercategory': c.get('supercategory', '')
                })
                base_cats_by_name[name] = next_cat_id
                next_cat_id += 1

        img_map = {}
        next_img_id = max([img['id'] for img in base.get('images', [])] + [0]) + 1
        base_imgs_by_fname = {img['file_name']: img['id']
                              for img in base.get('images', [])}

        for img in new.get('images', []):
            fname = img['file_name']
            if fname in base_imgs_by_fname:
                img_map[img['id']] = base_imgs_by_fname[fname]
            else:
                img_map[img['id']] = next_img_id
                new_img = dict(img)
                new_img['id'] = next_img_id
                base.setdefault('images', []).append(new_img)
                base_imgs_by_fname[fname] = next_img_id
                next_img_id += 1

        next_ann_id = max([a['id'] for a in base.get('annotations', [])] + [0]) + 1
        for ann in new.get('annotations', []):
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = img_map.get(ann['image_id'])
            new_ann['category_id'] = cat_map.get(ann['category_id'])

            if (new_ann['image_id'] is not None
                    and new_ann['category_id'] is not None):
                base.setdefault('annotations', []).append(new_ann)
                next_ann_id += 1

        return base

    # ------------------------------------------------------------------
    # Выбор выходной папки и запуск
    # ------------------------------------------------------------------
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

    def on_generate(self):
        if not (self.coco_path and self.images_path and self.output_path):
            return

        if not SAHI_AVAILABLE:
            QMessageBox.critical(self, "SAHI не установлен",
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
            'ann_format': self.annotation_format_combo.currentData(),
            'detected_format': self.detected_format,
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