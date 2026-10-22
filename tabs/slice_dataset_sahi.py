# slice_dataset_sahi.py
from import_libs_internal import *
from import_libs_methods_ui import setup_sahi_slicer_ui

import json
import os
import random
import shutil
import sys
import traceback
import tempfile
import math
from contextlib import redirect_stdout

try:
    from sahi.slicing import slice_coco
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


# ==================================================================
# YOLO .txt → COCO-аннотации
# ==================================================================
def yolo_labels_to_coco_annotations(label_path, img_w, img_h,
                                     input_format='auto'):
    results = []
    if not label_path or not os.path.exists(label_path):
        return results

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            n = len(coords)
            bbox = None
            seg = None
            fmt = input_format

            if fmt in ('auto', 'mixed'):
                if n == 4:
                    fmt = 'detect'
                elif n == 5:
                    fmt = 'obb_xywhr'
                elif n == 8:
                    fmt = 'obb_corners'
                elif n >= 6 and n % 2 == 0:
                    fmt = 'segment'
                else:
                    continue

            if fmt == 'detect' and n == 4:
                cx, cy, w, h = coords
                bbox = ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                        w * img_w, h * img_h)

            elif fmt == 'obb_corners' and n == 8:
                pts = [(coords[i] * img_w, coords[i + 1] * img_h)
                       for i in range(0, 8, 2)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = (min(xs), min(ys),
                        max(xs) - min(xs), max(ys) - min(ys))
                seg = [c for p in pts for c in p]

            elif fmt == 'obb_xywhr' and n == 5:
                cx, cy, w, h, angle = coords
                if abs(angle) > 3.2:
                    angle = math.radians(angle)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                dx, dy = w / 2, h / 2
                local = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
                pts = []
                for lx, ly in local:
                    rx = cx + lx * cos_a - ly * sin_a
                    ry = cy + lx * sin_a + ly * cos_a
                    pts.append((rx * img_w, ry * img_h))
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = (min(xs), min(ys),
                        max(xs) - min(xs), max(ys) - min(ys))
                seg = [c for p in pts for c in p]

            elif fmt == 'segment' and n >= 6 and n % 2 == 0:
                pts = [(coords[i] * img_w, coords[i + 1] * img_h)
                       for i in range(0, n, 2)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = (min(xs), min(ys),
                        max(xs) - min(xs), max(ys) - min(ys))
                seg = [c for p in pts for c in p]

            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                continue

            item = {'cls_id': cls_id, 'bbox': bbox}
            if seg is not None:
                item['segmentation'] = [seg]
            results.append(item)

    return results


# ==================================================================
# Нарезка
# ==================================================================
def slice_dataset(coco_json_path, image_dir, output_dir,
                  slice_size=640, overlap=0.2, min_area_ratio=0.3):
    if not SAHI_AVAILABLE:
        raise RuntimeError("SAHI не установлен. pip install sahi")

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔪 Нарезка {os.path.basename(coco_json_path)}")
    print(f"   тайл: {slice_size}×{slice_size}, "
          f"overlap={overlap:.0%}, min_area_ratio={min_area_ratio}")

    try:
        coco_dict, output_coco_path = slice_coco(
            coco_annotation_file_path=coco_json_path,
            image_dir=image_dir or '',
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
            coco_json_path, image_dir or '', "sliced_coco.json", output_dir,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            min_area_ratio=min_area_ratio, verbose=False,
        )

    print(f"   ✅ тайлов: {len(coco_dict.get('images', []))}, "
          f"аннотаций: {len(coco_dict.get('annotations', []))}")
    return coco_dict, output_coco_path


def detect_dataset_format(coco_dict):
    for ann in coco_dict.get("annotations", []):
        if ann.get("segmentation"):
            return "segmentation"
    return "bbox"


def describe_segmentation(coco_dict):
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


def subsample_negative_tiles(coco_dict, image_dir, keep_ratio=0.1, seed=42):
    if not (0 < keep_ratio <= 1):
        return coco_dict

    ann_by_img = {}
    for ann in coco_dict.get("annotations", []):
        ann_by_img[ann["image_id"]] = ann_by_img.get(ann["image_id"], 0) + 1

    with_objs = [i for i in coco_dict["images"]
                 if ann_by_img.get(i["id"], 0) > 0]
    without_objs = [i for i in coco_dict["images"]
                    if ann_by_img.get(i["id"], 0) == 0]

    if not without_objs:
        print("   ⚠️ Фоновых тайлов нет.")
        return coco_dict

    random.seed(seed)
    keep_count = max(1, int(len(without_objs) * keep_ratio))
    kept_negative = random.sample(without_objs, keep_count)
    dropped = [i for i in without_objs if i not in kept_negative]

    for img in dropped:
        fname = img.get("file_name")
        if fname:
            path = fname if os.path.isabs(fname) else os.path.join(image_dir or '', fname)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"   ⚠️ Не удалось удалить {path}: {e}")

    kept_ids = {i["id"] for i in with_objs + kept_negative}
    coco_dict["images"] = [i for i in coco_dict["images"] if i["id"] in kept_ids]
    coco_dict["annotations"] = [a for a in coco_dict["annotations"]
                                if a["image_id"] in kept_ids]
    print(f"   📉 Фоновых: {len(without_objs)} → {keep_count} "
          f"({keep_ratio:.0%}), удалено: {len(dropped)}")
    return coco_dict


def split_and_move(coco_dict, output_dir, train_ratio=0.8, seed=42,
                   strip_segmentation=False):
    """
    strip_segmentation: если True — удалить поле 'segmentation' из всех
    аннотаций перед сохранением sliced_coco.json. Используется, когда
    пользователь явно выбрал HBB (bbox) как формат разметки.
    """
    images = list(coco_dict["images"])
    has_val = 0 < train_ratio < 1

    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    if has_val:
        random.seed(seed)
        random.shuffle(images)
        n_train = int(len(images) * train_ratio)
        train_images, val_images = images[:n_train], images[n_train:]
    else:
        train_images, val_images = images, []

    train_ids = {i["id"] for i in train_images}
    val_ids = {i["id"] for i in val_images}

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

    # --- Удаление segmentation, если пользователь выбрал HBB ---
    n_stripped = 0
    if strip_segmentation:
        for ann in coco_dict.get("annotations", []):
            if 'segmentation' in ann:
                del ann['segmentation']
                n_stripped += 1
        if n_stripped:
            print(f"   🧹 Удалён segmentation из {n_stripped} аннотаций "
                  f"(формат: bbox)")
        else:
            print(f"   ℹ️ segmentation не найден — очистка не требуется")

    final_json = os.path.join(output_dir, "sliced_coco.json")
    with open(final_json, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)
    print(f"   ✂️ train={len(train_images)}, val={len(val_images)}, test=0")
    return coco_dict, final_json


# ==================================================================
# COCO → YOLO (bbox / segmentation) — без minAreaRect
# ==================================================================
def coco_to_yolo(coco_json_path, output_dir, class_names=None,
                 annotation_format="bbox", verbose=True):
    if annotation_format not in ("bbox", "segmentation"):
        raise ValueError(f"annotation_format: {annotation_format}")

    np = None
    if annotation_format == "segmentation":
        try:
            import numpy as np  # noqa
        except ImportError:
            raise RuntimeError("Для segmentation нужен numpy")

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

    img_meta = {i["id"]: i for i in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_bbox = n_seg = n_skipped = 0
    seg_dist = {}
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

            if prefer_seg and has_seg:
                write_bbox, write_seg = False, True
            elif prefer_seg and has_bbox:
                write_bbox, write_seg = True, False
            elif not prefer_seg and has_bbox:
                write_bbox, write_seg = True, False
            elif not prefer_seg and has_seg:
                write_bbox, write_seg = False, True
            else:
                n_skipped += 1
                continue

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

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if verbose:
        print(f"   📝 YOLO [{annotation_format}]: bbox={n_bbox}, "
              f"seg={n_seg}, пропущено={n_skipped}")
        if seg_dist:
            dist_str = ", ".join(f"{k}pt:{v}" for k, v in sorted(seg_dist.items()))
            print(f"      сегменты: {dist_str}")
            non4 = sum(v for k, v in seg_dist.items() if k != 4)
            if non4 > 0:
                print(f"      ⚠️ {non4} сегментов не 4-точечные — "
                      f"YOLO-OBB их не примет (нужен YOLO-seg).")


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
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(fname)), canvas)
    print(f"   🖼️ Визуализация: {len(samples)} → {vis_dir}")


# ==================================================================
# Worker
# ==================================================================
class SliceWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

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
            # Определяем формат заранее: он влияет и на sliced_coco.json
            ann_format_requested = p.get('ann_format', 'auto')
            strip_seg = (ann_format_requested == 'bbox')
            self.log_signal.emit(
                f"   🏷️ Формат разметки (выбор): {ann_format_requested}"
                + (f" → сегментация будет удалена из sliced_coco.json"
                   if strip_seg else ""))

            self.progress_signal.emit(
                -1, "Нарезка SAHI (может занять время)...")
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
                    coco_dict=coco_dict, image_dir=p['out'],
                    keep_ratio=p['keep_negative'], seed=p['seed'])

            self.progress_signal.emit(50, "Разделение и перемещение...")
            final_dict, final_json = split_and_move(
                coco_dict=coco_dict, output_dir=p['out'],
                train_ratio=p['split'], seed=p['seed'],
                strip_segmentation=strip_seg,
            )

            if sliced_json and os.path.abspath(sliced_json) != os.path.abspath(final_json):
                if os.path.exists(sliced_json):
                    try:
                        os.remove(sliced_json)
                        self.log_signal.emit(
                            f"🧹 Удалён промежуточный: "
                            f"{os.path.basename(sliced_json)}")
                    except OSError as e:
                        self.log_signal.emit(f"⚠️ {e}")

            if p['format'] in ('yolo', 'both'):
                self.progress_signal.emit(70, "Конвертация в YOLO...")
                cats = sorted(final_dict.get('categories', []),
                              key=lambda c: c['id'])
                class_names = [c.get('name', str(c['id'])) for c in cats]

                ann_format = ann_format_requested
                if ann_format == 'auto':
                    ann_format = p.get('detected_format', 'bbox')
                if ann_format not in ('bbox', 'segmentation'):
                    ann_format = 'bbox'

                n_anns = len(final_dict.get('annotations', []))
                n_seg = sum(1 for a in final_dict.get('annotations', [])
                            if a.get('segmentation'))
                n_bbox = sum(1 for a in final_dict.get('annotations', [])
                             if isinstance(a.get('bbox'), (list, tuple))
                             and len(a['bbox']) == 4)
                self.log_signal.emit(
                    f"   📊 аннотаций={n_anns}, bbox={n_bbox}, seg={n_seg}")
                self.log_signal.emit(f"   🏷️ Формат YOLO: {ann_format}")

                if ann_format == 'segmentation' and n_seg == 0:
                    self.log_signal.emit(
                        "   ⚠️ Нет segmentation — будет bbox.")

                coco_to_yolo(
                    coco_json_path=final_json, output_dir=p['out'],
                    class_names=class_names,
                    annotation_format=ann_format)
                write_yolo_yaml(
                    output_dir=p['out'], class_names=class_names,
                    channels=p['channels'])

            if p['visualize'] > 0:
                self.progress_signal.emit(85, "Визуализация...")
                visualize_samples(final_json, p['out'], n=p['visualize'])

        self.progress_signal.emit(100, "Готово")


# ==================================================================
# Главное окно
# ==================================================================
class SahiSlicerWindow(DatasetLoaderMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAHI Dataset Slicer")
        self.setMinimumWidth(1400)
        self.setMinimumHeight(900)

        setup_sahi_slicer_ui(self)

        self.images_folder = None
        self.labels_folder = None
        self.masks_folder = None
        self.output_folder = None
        self.pairs = []
        self.worker = None

        self.unique_classes = []
        self.original_ids = []
        self.mask_class_remap = {}
        self.included_orig_ids = set()
        self.class_remap = {}
        self.num_classes = 0
        self.annotation_types_stats = {}

        self.dataset_type = 0
        self.input_format = 'auto'
        self.mask_pixel_offset_in = 1

        self.validator = PairValidator(dataset_type=0, input_format='auto')
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        self._last_patched_yaml = None
        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None
        self._coco_tmp_labels_dir = None
        self._coco_json_counter = 0
        self._updating_class_table = False

        self.coco_path = None
        self.images_path = None
        self._coco_dict = None
        self._temp_coco_path = None
        self.detected_format = 'bbox'
        self.seg_hint = ""
        self._img_size_cache = {}

        try:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
        except Exception:
            pass

        # --- Подключения (как в preparation) ---
        self.toggle_log_btn = self.btn_toggle_log
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.dataset_type_combo.currentIndexChanged.connect(
            self._on_dataset_type_changed)
        self.input_format_combo.currentIndexChanged.connect(
            self._on_input_format_changed)
        self.class_table.itemChanged.connect(
            self.on_class_table_item_changed)
        self.include_empty_annotations.stateChanged.connect(
            self._on_include_empty_toggled)
        self.file_list.itemDoubleClicked.connect(
            self._on_file_list_double_clicked)

        self._copy_shortcut = QShortcut(QKeySequence.Copy, self.file_list)
        self._copy_shortcut.setContext(Qt.WidgetShortcut)
        self._copy_shortcut.activated.connect(
            self._copy_selected_rows_to_clipboard)

        try:
            if hasattr(self, '_sync_all_states'):
                self._sync_all_states()
        except Exception as e:
            self._log_exception("Ошибка первичной синхронизации UI", e)

        self.log("SAHI Slicer готов к работе.")
        if not SAHI_AVAILABLE:
            self.log("⚠️ SAHI не установлен! pip install sahi")

        self.update_generate_button_state()

    def closeEvent(self, event):
        if self._temp_coco_path and os.path.exists(self._temp_coco_path):
            try:
                os.remove(self._temp_coco_path)
            except Exception:
                pass
        super().closeEvent(event)

    def _show_indeterminate(self, msg):
        try:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText(msg)
            QApplication.processEvents()
        except Exception:
            pass

    def _hide_indeterminate(self, final_msg="Готов к работе"):
        try:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText(final_msg)
            QApplication.processEvents()
        except Exception:
            pass

    # ==================================================================
    # Обёртки для долгих операций (без *args — PyQt отбросит checked)
    # ==================================================================
    def scan_pairs(self):
        self._show_indeterminate("Сканирование и валидация пар...")
        try:
            return super().scan_pairs()
        except Exception as e:
            self._log_exception("Ошибка в scan_pairs", e)
            QMessageBox.critical(self, "Ошибка сканирования",
                                 f"{type(e).__name__}: {e}")
        finally:
            self._hide_indeterminate()

    def _load_yolo_yaml_from_path(self, file_path):
        self._show_indeterminate(
            f"Загрузка YAML: {os.path.basename(file_path)}...")
        try:
            return super()._load_yolo_yaml_from_path(file_path)
        except Exception as e:
            self._log_exception("Ошибка в _load_yolo_yaml_from_path", e)
            QMessageBox.critical(self, "Ошибка YAML",
                                 f"{type(e).__name__}: {e}")
        finally:
            self._hide_indeterminate()

    def _load_coco_from_paths(self, json_paths, replace=True):
        self._show_indeterminate("Загрузка COCO JSON...")
        try:
            return super()._load_coco_from_paths(
                json_paths, replace=replace)
        except Exception as e:
            self._log_exception("Ошибка в _load_coco_from_paths", e)
            QMessageBox.critical(self, "Ошибка COCO",
                                 f"{type(e).__name__}: {e}")
        finally:
            self._hide_indeterminate()

    # ==================================================================
    # Mixin hooks
    # ==================================================================
    def _after_load_hook(self):
        if self._temp_coco_path and os.path.exists(self._temp_coco_path):
            try:
                os.remove(self._temp_coco_path)
            except Exception:
                pass
        self._coco_dict = None
        self._temp_coco_path = None
        self.coco_path = None

        n_pairs = len(self.pairs)
        n_classes = len(self.get_class_names_from_table())
        fmt = self.input_format

        self.coco_label.setText("(соберётся при генерации)")
        self.images_label.setText(self.images_folder or "—")
        self.validation_summary_label.setText(
            f"Пар: {n_pairs} | Классов: {n_classes} | Формат: {fmt}")

        self.log(f"📋 Готово к генерации: pairs={n_pairs}, "
                 f"classes={n_classes}, format={fmt}")

        self.update_generate_button_state()

    def _after_type_change_hook(self, new_type):
        pass

    def log(self, message):
        try:
            self.log_view.append(message)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
            QApplication.processEvents()
        except Exception:
            print(f"[log] {message}")

    def toggle_log(self, checked=False):
        self.log_view.setVisible(bool(checked))
        self.btn_toggle_log.setText(
            "Скрыть лог" if checked else "Показать лог")

    def get_class_names_from_table(self):
        names = {}
        if not hasattr(self, 'class_table'):
            return names
        for i in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(i, 4)
            if cb and cb.isChecked():
                try:
                    new_id = int(self.class_table.item(i, 0).text())
                except (ValueError, TypeError, AttributeError):
                    continue
                name_item = self.class_table.item(i, 2)
                name = name_item.text().strip() if name_item else str(new_id)
                names[new_id] = name or str(new_id)
        return names

    def _pairs_to_coco(self, progress_cb=None):
        if not self.pairs:
            return None
        import cv2

        input_format = self.input_format or 'auto'

        orig_to_name = {}
        if hasattr(self, 'class_table'):
            for i in range(self.class_table.rowCount()):
                orig_id_item = self.class_table.item(i, 1)
                name_item = self.class_table.item(i, 2)
                cb = self.class_table.cellWidget(i, 4)
                if orig_id_item and cb and cb.isChecked():
                    try:
                        orig_id = int(orig_id_item.text())
                        name = name_item.text() if name_item else str(orig_id)
                        orig_to_name[orig_id] = name or str(orig_id)
                    except (ValueError, TypeError):
                        pass

        images, annotations = [], []
        cat_name_to_id = {}
        next_img_id = next_ann_id = 1
        total = len(self.pairs)

        for i, pair in enumerate(self.pairs):
            img_path = pair[0]
            aux_path = pair[1] if len(pair) > 1 else None
            if not img_path or not os.path.exists(img_path):
                if progress_cb:
                    progress_cb(i + 1, total)
                continue

            if img_path in self._img_size_cache:
                img_w, img_h = self._img_size_cache[img_path]
            else:
                img = cv2.imread(img_path)
                if img is None:
                    if progress_cb:
                        progress_cb(i + 1, total)
                    continue
                img_h, img_w = img.shape[:2]
                self._img_size_cache[img_path] = (img_w, img_h)

            img_id = next_img_id
            next_img_id += 1
            images.append({
                'id': img_id,
                'file_name': os.path.abspath(img_path),
                'width': img_w,
                'height': img_h,
            })

            label_path = aux_path
            if not label_path and getattr(self, 'labels_folder', None):
                base = os.path.splitext(os.path.basename(img_path))[0]
                cand = os.path.join(self.labels_folder, base + '.txt')
                if os.path.exists(cand):
                    label_path = cand

            raw_anns = []
            if label_path and os.path.exists(label_path):
                raw_anns = yolo_labels_to_coco_annotations(
                    label_path, img_w, img_h, input_format)

            for ad in raw_anns:
                cls_name = orig_to_name.get(ad['cls_id'])
                if cls_name is None:
                    continue
                if cls_name not in cat_name_to_id:
                    cat_name_to_id[cls_name] = len(cat_name_to_id)
                cat_id = cat_name_to_id[cls_name]

                bbox = ad['bbox']
                ann = {
                    'id': next_ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': list(bbox),
                    'area': float(bbox[2] * bbox[3]),
                    'iscrowd': 0,
                }
                if 'segmentation' in ad:
                    ann['segmentation'] = ad['segmentation']
                annotations.append(ann)
                next_ann_id += 1

            if progress_cb:
                progress_cb(i + 1, total)

        if not images:
            return None

        cats = [{'id': cid, 'name': name, 'supercategory': 'none'}
                for name, cid in cat_name_to_id.items()]
        return {'images': images, 'annotations': annotations, 'categories': cats}

    def _save_temp_coco(self):
        if self._coco_dict is None:
            return
        if self._temp_coco_path and os.path.exists(self._temp_coco_path):
            try:
                os.remove(self._temp_coco_path)
            except Exception:
                pass
        fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='sahi_coco_')
        os.close(fd)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self._coco_dict, f, ensure_ascii=False)
        self.coco_path = temp_path
        self._temp_coco_path = temp_path

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку для нового датасета")
        if not folder:
            return
        self.output_folder = folder
        self.output_edit.setText(folder)
        self.btn_open_output.setEnabled(True)
        self.log(f"📁 Output: {folder}")
        self.update_generate_button_state()

    def open_output_folder(self):
        folder = self.output_folder
        if not folder or not os.path.isdir(folder):
            return
        if sys.platform == 'win32':
            os.startfile(folder)
        elif sys.platform == 'darwin':
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    def update_generate_button_state(self):
        has_pairs = bool(self.pairs)
        has_output = bool(self.output_folder)
        can_run = bool(SAHI_AVAILABLE and has_pairs and has_output)
        self.btn_generate.setEnabled(can_run and not self._is_running())

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    # ==================================================================
    # Генерация
    # ==================================================================
    def on_generate(self):
        if not SAHI_AVAILABLE:
            QMessageBox.critical(self, "SAHI не установлен",
                                 "Установите: pip install sahi")
            return
        if not self.output_folder:
            QMessageBox.warning(self, "Ошибка",
                                "Не выбрана папка для нового датасета.")
            return
        if not self.pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Сначала загрузите датасет.")
            return

        if os.path.exists(self.output_folder) and os.listdir(self.output_folder):
            reply = QMessageBox.question(
                self, "Папка не пуста",
                f"Папка:\n{self.output_folder}\nне пуста. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.log("")
        self.log("═" * 60)
        self.log(f"🔄 Сборка COCO из {len(self.pairs)} пар...")
        self.status_label.setText("Сборка COCO из пар...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        def on_build_progress(cur, tot):
            pct = int(cur / tot * 100) if tot else 0
            self.progress_bar.setValue(pct)
            self.status_label.setText(
                f"Сборка COCO: {cur}/{tot} ({pct}%)")
            QApplication.processEvents()

        try:
            self._coco_dict = self._pairs_to_coco(
                progress_cb=on_build_progress)
        except Exception as e:
            self._hide_indeterminate("Ошибка")
            QMessageBox.critical(self, "Ошибка сборки COCO",
                                 f"{type(e).__name__}: {e}")
            return

        if not self._coco_dict or not self._coco_dict.get('images'):
            self._hide_indeterminate("Ошибка")
            QMessageBox.warning(
                self, "Нет данных",
                "COCO пуст (возможно, все классы исключены или "
                "ни одна пара не валидна).")
            return

        self._save_temp_coco()
        self.detected_format = detect_dataset_format(self._coco_dict)
        self.seg_hint = describe_segmentation(self._coco_dict)

        n_imgs = len(self._coco_dict.get('images', []))
        n_anns = len(self._coco_dict.get('annotations', []))
        self.log(f"✅ COCO собран: {n_imgs} imgs, {n_anns} anns, "
                 f"format={self.detected_format}"
                 + (f" — {self.seg_hint}" if self.seg_hint else ""))
        self.coco_label.setText(
            f"{os.path.basename(self._temp_coco_path or '')} | "
            f"изображений: {n_imgs}, аннотаций: {n_anns}")

        self.log(f"📁 COCO:   {self._temp_coco_path}")
        self.log(f"📁 Output: {self.output_folder}")

        params = {
            'coco': self.coco_path,
            'images': self.images_path or '',
            'out': self.output_folder,
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
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск нарезки...")

        self.worker = SliceWorker(params)
        self.worker.log_signal.connect(self._on_log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _on_log(self, line):
        self.log_view.append(line)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, percent, msg):
        try:
            if percent < 0:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(percent)
            self.status_label.setText(msg)
        except Exception:
            pass

    def _on_finished(self, ok, msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if ok else 0)
        self.update_generate_button_state()
        if ok:
            self.status_label.setText("✅ Готово")
            reply = QMessageBox.information(
                self, "Успех",
                f"Датасет создан в:\n{self.output_folder}\n\nОткрыть папку?",
                QMessageBox.Open | QMessageBox.Close, QMessageBox.Open)
            if reply == QMessageBox.Open:
                self.open_output_folder()
        else:
            self.status_label.setText("❌ Ошибка")
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка:\n{msg}\n\nПодробности в логе.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SahiSlicerWindow()
    window.show()
    sys.exit(app.exec_())