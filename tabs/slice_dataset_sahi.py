
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# slice_dataset_sahi.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import json
import os
import random
import shutil
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QShortcut

from ui_tabs.slice_dataset_sahi_ui import setup_sahi_slicer_ui
from ui.dataset_loader import DatasetLoaderMixin
from utils_new.images.processing import ImageStats
from utils_new.dataset.validation import PairValidator
from utils_new.annotations import ImageSize
from utils_new.import_export.yolo_labels import read_yolo_labels
from utils_new.import_export.dataset_output import _coco_record, write_classes_and_yaml
from utils_new.import_export.coco import export_coco_yolo
from utils_new.import_export.json_objects import write_json_atomic
from utils_new.dataset.preparation import allocate_split_counts

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
    if label_path is None:
        return []
    annotations = read_yolo_labels(label_path, ImageSize(img_w, img_h), strict=True,
                                   format_hint='auto' if input_format == 'mixed' else input_format)
    result = []
    for index, ann in enumerate(annotations):
        record = _coco_record(ann, 0, index)
        record['cls_id'] = ann.class_id
        result.append(record)
    return result


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

    with open(coco_json_path, encoding='utf-8-sig') as stream:
        source_images = json.load(stream)['images']
    # SAHI names tiles <source stem>_<source index>_<crop coordinates>.
    # Retain source membership: overlapping tiles must not leak across splits.
    prefixes = [(f"{Path(source['file_name']).stem}_{index}_", source)
                for index, source in enumerate(source_images)]
    for tile in coco_dict.get('images', []):
        name = Path(tile['file_name']).stem
        parent_prefix, *coordinates = name.rsplit('_', 4)
        matches = [source for prefix, source in prefixes if parent_prefix + '_' == prefix
                   and len(coordinates) == 4 and all(c.isdigit() for c in coordinates)]
        if len(matches) != 1:
            raise ValueError(f"SAHI: cannot identify source of tile {tile['file_name']}")
        source = matches[0]
        tile['source_image'] = str(source['id'])
        tile['annotation_status'] = source.get('annotation_status', 'labelled')
    write_json_atomic(output_coco_path, coco_dict)

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
    if not 0 <= keep_ratio <= 1:
        raise ValueError('Background ratio must be in [0, 1]')
    result = deepcopy(coco_dict)
    annotated = {a['image_id'] for a in result.get('annotations', [])}
    unknown = {i['id'] for i in result['images'] if i.get('annotation_status') == 'unlabelled'}
    negative = [i for i in result['images'] if i['id'] not in annotated | unknown]
    count = int(len(negative) * keep_ratio)
    kept = {i['id'] for i in random.Random(seed).sample(negative, count)} | annotated | unknown
    dropped = [i for i in negative if i['id'] not in kept]
    # Preflight ALL paths before deleting anything: source datasets are read-only.
    paths = [_owned_tile(image_dir, image['file_name']) for image in dropped]
    for path in paths:
        path.unlink()
    result['images'] = [i for i in result['images'] if i['id'] in kept]
    result['annotations'] = [a for a in result.get('annotations', []) if a['image_id'] in kept]
    print(f"   Фоновых тайлов: {len(negative)} → {count}")
    return result


def _owned_tile(output_dir, name):
    root = Path(output_dir).resolve()
    path = Path(name)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Нет тайла внутри выходной папки: {path}')
    return path


def split_and_move(coco_dict, output_dir, train_ratio=0.8, seed=42,
                   strip_segmentation=False):
    result = deepcopy(coco_dict)
    images = list(result['images'])
    random.Random(seed).shuffle(images)
    n_train, _ = allocate_split_counts(len(images), (train_ratio, 1 - train_ratio))
    groups = {}
    for entry in images:
        groups.setdefault(entry.get('source_image', entry['id']), []).append(entry)
    # Greedy allocation keeps all overlapping tiles of a source in one split.
    train_ids, assigned = set(), dict(train=0, val=0)
    targets = dict(train=len(images) * train_ratio, val=len(images) * (1 - train_ratio))
    for group in groups.values():
        split = max((key for key in targets if targets[key] > 0),
                    key=lambda key: targets[key] - assigned[key])
        assigned[split] += len(group)
        if split == 'train':
            train_ids.update(entry['id'] for entry in group)
    n_train = len(train_ids)
    planned, used = [], set()
    for image in images:
        source = _owned_tile(output_dir, image['file_name'])
        split = 'train' if image['id'] in train_ids else 'val'
        relative = Path('images') / split / source.name
        key = relative.as_posix().casefold()
        target = Path(output_dir) / relative
        if key in used or target.exists():
            raise ValueError(f'Коллизия тайлов: {target}')
        used.add(key)
        planned.append((source, target))
        image['file_name'] = relative.as_posix()
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            (Path(output_dir) / folder / split).mkdir(parents=True, exist_ok=True)
    for source, target in planned:
        shutil.move(str(source), str(target))
    if strip_segmentation:
        for ann in result.get('annotations', []):
            ann.pop('segmentation', None)
            if len(ann.get('bbox', [])) == 4:
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
    final_json = Path(output_dir) / 'sliced_coco.json'
    write_json_atomic(final_json, result)
    print(f'   train={n_train}, val={len(images) - n_train}, test=0')
    return result, str(final_json)


# ==================================================================
# COCO → YOLO (bbox / segmentation) — без minAreaRect
# ==================================================================
def coco_to_yolo(coco_json_path, output_dir, class_names=None,
                 annotation_format="bbox", verbose=True):
    if annotation_format not in ('bbox', 'segmentation'):
        raise ValueError(f'annotation_format: {annotation_format}')
    return export_coco_yolo(coco_json_path, output_dir, class_names=class_names,
                            subtype='detect' if annotation_format == 'bbox' else 'segment',
                            log=print if verbose else None)


def write_yolo_yaml(output_dir, class_names, channels=3):
    write_classes_and_yaml(output_dir, dict(enumerate(class_names)), yolo_channels=int(channels))


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
        destination = Path(self.params['out']).resolve()
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError('Для экспорта SAHI выберите пустую папку')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.sahi-') as temporary:
            stage = Path(temporary) / 'dataset'
            stage.mkdir()
            original = self.params
            self.params = dict(original, out=str(stage))
            try:
                self._pipeline_staged()
                if destination.exists():
                    destination.rmdir()
                os.replace(stage, destination)
            finally:
                self.params = original

    def _pipeline_staged(self):
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
                + (" → сегментация будет удалена из sliced_coco.json"
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

            if not coco_dict.get('images'):
                raise ValueError('После отбора фоновых тайлов датасет пуст')

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
                        "   Нет segmentation — bbox будут представлены полигонами.")

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
        self.setMinimumSize(900, 600)

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

        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None
        self._updating_class_table = False

        self.coco_path = None
        self.images_path = None
        self._coco_dict = None
        self._temp_coco_path = None
        self.detected_format = 'bbox'
        self.seg_hint = ""

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
        except Exception:
            pass

    def _hide_indeterminate(self, final_msg="Готов к работе"):
        try:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText(final_msg)
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
        can_run = bool(SAHI_AVAILABLE and has_pairs and has_output and self._dataset_ready())
        self.btn_generate.setEnabled(can_run and not self._is_running())

    def _is_running(self):
        return (self.worker is not None and self.worker.isRunning()) or (hasattr(self, "_dataset_tasks") and self._dataset_tasks.busy("sahi-source"))

    # ==================================================================
    # Генерация
    # ==================================================================
    def on_generate(self):
        if not self._dataset_ready() or self._is_running():
            self.log("Дождитесь завершения проверки датасета или текущей нарезки.")
            return
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

        params = {
            'coco': None,
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

        from ui.dataset_loading import dataset_tasks
        from utils_new.import_export.coco import CocoPairsJob
        names = {int(self.class_table.item(row, 1).text()): self.class_table.item(row, 2).text()
                 for row in range(self.class_table.rowCount()) if self.class_table.cellWidget(row, 4).isChecked()}
        def prepared(result):
            self._coco_dict = result['coco']
            self._temp_coco_path = self.coco_path = result['path']
            params['coco'] = result['path']
            params['detected_format'] = detect_dataset_format(result['coco'])
            self.worker = SliceWorker(params)
            self.worker.log_signal.connect(self._on_log)
            self.worker.progress_signal.connect(self._on_progress)
            self.worker.finished_signal.connect(self._on_finished)
            self.worker.start()
            self.update_generate_button_state()
        dataset_tasks(self).start('sahi-source', CocoPairsJob, list(self.pairs), names, self.input_format,
            label='Подготовка COCO для нарезки…', finished=prepared)
        self.update_generate_button_state()

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
    from ui.application import exec_window
    sys.exit(exec_window(window, app))