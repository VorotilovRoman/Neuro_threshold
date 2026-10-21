# dataset_analysis.py
from import_libs_internal import *
from import_libs_methods_ui import setup_dataset_analysis_ui
from utils.dataset_analysis_logic import (
    DatasetAnalyzer, FilterParams,
    ClassImbalanceAnalyzer, ResolutionAnalyzer, BrightnessAnalyzer,
    TrainingResultsLoader,
)
from utils.dataset_audit_logic import detect_device

from collections import Counter
import traceback
import os
import shutil
import subprocess
import io
import sys as _sys
import numpy as np



# ============================================================
# Валидация YOLO в отдельном потоке
# ============================================================
class YoloValThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, model_path, data_path, split='val',
                 imgsz=640, conf=0.001, iou=0.6, device=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.data_path = data_path
        self.split = split
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device or detect_device()
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _extract_metrics(self, metrics):
        """Извлекает summary + per-class из объекта DetMetrics."""
        box = getattr(metrics, 'box', None)
        names = getattr(metrics, 'names', {}) or {}
        summary = {}
        per_class = []

        if box is None:
            return {'summary': summary, 'per_class': per_class}

        def _safe(obj, attr, default=0.0):
            try:
                v = getattr(obj, attr, None)
                if v is None:
                    return default
                if hasattr(v, '__len__') and not isinstance(v, str):
                    return float(np.mean(v)) if len(v) > 0 else default
                return float(v)
            except Exception:
                return default

        summary['Precision'] = _safe(box, 'mp')
        summary['Recall'] = _safe(box, 'mr')
        summary['mAP50'] = _safe(box, 'map50')
        summary['mAP50-95'] = _safe(box, 'map')
        summary['mAP75'] = _safe(box, 'map75')

        try:
            f1_arr = getattr(box, 'f1', None)
            if f1_arr is not None and len(f1_arr) > 0:
                summary['F1'] = float(np.mean(f1_arr))
        except Exception:
            pass
        if 'F1' not in summary:
            p, r = summary['Precision'], summary['Recall']
            summary['F1'] = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        try:
            ap_class_index = getattr(box, 'ap_class_index', None)
            if ap_class_index is None:
                return {'summary': summary, 'per_class': per_class}

            p_arr = getattr(box, 'p', None)
            r_arr = getattr(box, 'r', None)
            f1_arr = getattr(box, 'f1', None)
            ap50_arr = getattr(box, 'ap50', None)
            ap_arr = getattr(box, 'ap', None)

            def _v(arr, i, fmt='{:.4f}'):
                if arr is None or i >= len(arr):
                    return ''
                try:
                    return fmt.format(float(arr[i]))
                except Exception:
                    return ''

            for i, ci in enumerate(ap_class_index):
                ci = int(ci)
                if isinstance(names, dict):
                    name = names.get(ci, str(ci))
                elif isinstance(names, (list, tuple)) and ci < len(names):
                    name = names[ci]
                else:
                    name = str(ci)
                per_class.append({
                    'class': f"{ci}: {name}",
                    'p': _v(p_arr, i),
                    'r': _v(r_arr, i),
                    'f1': _v(f1_arr, i),
                    'ap50': _v(ap50_arr, i),
                    'ap': _v(ap_arr, i),
                })
        except Exception as e:
            self.log_signal.emit(f"[VAL] Ошибка per-class: {e}")

        return {'summary': summary, 'per_class': per_class}


    def run(self):
        try:
            from ultralytics import YOLO
            from ultralytics.utils import LOGGER
        except ImportError:
            self.finished_signal.emit({
                'error': 'ultralytics не установлен',
                'metrics': None})
            return

        import logging

        class _QtLoggerHandler(logging.Handler):
            """Пересылает записи Ultralytics-LOGGER в Qt-сигнал."""
            def __init__(self, emit_fn):
                super().__init__()
                self._emit = emit_fn
            def emit(self, record):
                try:
                    msg = self.format(record).strip()
                    if msg:
                        self._emit(msg)
                except Exception:
                    pass

        handler = _QtLoggerHandler(
            lambda s: self.log_signal.emit(f"[VAL] {s}"))
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)

        try:
            self.log_signal.emit(
                f"[VAL] Загрузка модели: {self.model_path}")
            model = YOLO(self.model_path)

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"data.yaml не найден: {self.data_path}")

            self.log_signal.emit(
                f"[VAL] Модель загружена. Старт валидации: "
                f"data={self.data_path}, split={self.split}, "
                f"imgsz={self.imgsz}, conf={self.conf}, iou={self.iou}, "
                f"device={self.device}")

            # Callback отмены — срабатывает между батчами
            try:
                def _stop_cb(*args, **kwargs):
                    if self._stop:
                        raise SystemExit("__VAL_CANCEL__")
                model.add_callback("on_val_batch_end", _stop_cb)
            except Exception as e:
                self.log_signal.emit(
                    f"[VAL] Не удалось добавить callback отмены: {e}")

            try:
                metrics = model.val(
                    data=self.data_path,
                    split=self.split,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=True,
                )
            except Exception as e:
                # Один автоматический fallback на CPU при сбое GPU
                msg = str(e).lower()
                if (self.device != 'cpu'
                        and any(k in msg for k in
                                ('cuda', 'out of memory',
                                 'device', 'cublas', 'cudnn'))):
                    self.log_signal.emit(
                        f"[VAL] Ошибка на {self.device} ({e}), "
                        f"повтор на cpu")
                    try:
                        model.to('cpu')
                    except Exception:
                        pass
                    self.device = 'cpu'
                    metrics = model.val(
                        data=self.data_path,
                        split=self.split,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device='cpu',
                        verbose=True,
                    )
                else:
                    raise
        except SystemExit as e:
            if str(e) == "__VAL_CANCEL__":
                self.finished_signal.emit({
                    'error': 'Валидация отменена пользователем',
                    'metrics': None})
            else:
                self.finished_signal.emit({
                    'error': f'Валидация прервана: {e}',
                    'metrics': None})
            return
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"[VAL][ERROR] {e}")
            self.log_signal.emit(tb)
            self.finished_signal.emit({'error': str(e), 'metrics': None})
            return
        finally:
            try:
                LOGGER.removeHandler(handler)
            except Exception:
                pass

        if self._stop:
            self.finished_signal.emit({
                'error': 'Валидация отменена пользователем',
                'metrics': None})
            return

        try:
            data = self._extract_metrics(metrics)
            self.finished_signal.emit({'error': None, 'metrics': data})
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"[VAL][ERROR] extract: {e}")
            self.log_signal.emit(tb)
            self.finished_signal.emit({'error': str(e), 'metrics': None})





class DatasetAnalysisWindow(DatasetLoaderMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ и фильтрация датасета")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)
        setup_dataset_analysis_ui(self)

        # Автоопределение устройства + лог
        self._device = detect_device()
        try:
            import torch
            if self._device.startswith('cuda'):
                gpu_count = torch.cuda.device_count()
                try:
                    gpu_name = (torch.cuda.get_device_name(0)
                                if gpu_count > 0 else '?')
                except Exception:
                    gpu_name = '?'
                self._audit_log(
                    f"[DEVICE] Обнаружен GPU: {gpu_name} "
                    f"(CUDA devices: {gpu_count}). "
                    f"Модели будут выполняться на {self._device}.",
                    tag="INIT", level='OK')
            else:
                self._audit_log(
                    "[DEVICE] GPU не обнаружен (CUDA недоступна). "
                    "Модели будут выполняться на CPU.",
                    tag="INIT", level='WARN')
        except ImportError:
            self._audit_log(
                "[DEVICE] PyTorch не установлен. "
                "Модели будут выполняться на CPU.",
                tag="INIT", level='WARN')

        self.images_folder = self.labels_folder = self.masks_folder = None
        self.pairs, self.filtered_pairs = [], []

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
        self.analyzer = DatasetAnalyzer(dataset_type=0, input_format='auto')
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None
        self._coco_tmp_labels_dir = None
        self._coco_json_counter = 0
        self._updating_class_table = False

        # Результаты / состояние
        self._stop_requested = False
        self._last_quality = None
        self._last_dup = None
        self._last_imbalance = None
        self._last_heatmap = None
        self._last_audit_results = []
        self._last_audit_meta = None
        self._last_cluster = None
        self._last_cluster_data = None
        self._last_validation = None
        self._last_training_results = None
        self._last_export_dir = None
        self._last_report_dir = None
        self._pair_class_cache = {}

        self._audit_model_cache = None
        self._audit_model_cache_path = None
        self._heatmap_colorbar = None

        self._val_thread = None
        self._dup_groups_by_row = []

        self._audit_model_path = None
        self._audit_thread = None
        self._cluster_thread = None
        self._report_thread = None

        # --- подключения ---
        self.dataset_type_combo.currentIndexChanged.connect(
            self._on_dataset_type_changed)
        self.input_format_combo.currentIndexChanged.connect(
            self._on_input_format_changed)
        self.class_table.itemChanged.connect(self.on_class_table_item_changed)
        self.include_empty_annotations.stateChanged.connect(
            self._on_include_empty_toggled)
        self.file_list.itemDoubleClicked.connect(
            self._on_file_list_double_clicked)

        self._copy_shortcut = QShortcut(QKeySequence.Copy, self.file_list)
        self._copy_shortcut.setContext(Qt.WidgetShortcut)
        self._copy_shortcut.activated.connect(
            self._copy_selected_rows_to_clipboard)

        self.btn_apply_filters.clicked.connect(
            self.apply_filters_and_update_metrics)
        self.btn_export_filtered.clicked.connect(self.export_filtered_dataset)
        self.toggle_log_btn.clicked.connect(self.toggle_log)

        self.btn_run_quality.clicked.connect(self.run_quality_analysis)
        self.btn_stop_quality.clicked.connect(self.stop_quality_analysis)
        self.btn_run_duplicates.clicked.connect(self.run_duplicate_analysis)
        self.btn_stop_duplicates.clicked.connect(self.stop_duplicate_analysis)
        self.btn_run_heatmap.clicked.connect(self.run_heatmap_analysis)
        self.btn_stop_heatmap.clicked.connect(self.stop_heatmap_analysis)
        self.btn_load_audit_model.clicked.connect(self.load_audit_model)
        self.btn_run_audit.clicked.connect(self.run_loss_inspection)
        self.btn_stop_audit.clicked.connect(self.stop_loss_inspection)
        self.btn_run_cluster.clicked.connect(self.run_embedding_clustering)
        self.btn_stop_cluster.clicked.connect(self.stop_embedding_clustering)
        self.btn_generate_report.clicked.connect(self.generate_full_report)

        self.btn_browse_val_data.clicked.connect(self.browse_val_data)
        self.btn_run_validation.clicked.connect(self.run_yolo_validation)
        self.btn_stop_validation.clicked.connect(self.stop_yolo_validation)

        self.btn_browse_training_results.clicked.connect(
            self.browse_training_results)
        self.btn_load_training_results.clicked.connect(
            self.load_training_results)

        self.btn_open_last_export.clicked.connect(
            self.open_last_export_folder)
        self.btn_open_report_folder.clicked.connect(
            self.open_report_folder)

        self.audit_table.cellDoubleClicked.connect(self._open_audit_row)

        self.dup_table.itemSelectionChanged.connect(
            self._on_dup_row_selected)
        self.dup_table.cellDoubleClicked.connect(
            self._on_dup_row_double_clicked)

        self.cluster_canvas.mpl_connect(
            'button_press_event', self._on_cluster_click)

        self._sync_all_states()
        self._sync_analyzer()
        self._audit_log("Вкладка анализа датасета инициализирована.",
                        tag="INIT")

    # ==================================================================
    # Логирование
    # ==================================================================
    def _audit_log(self, message, tag='AUDIT', level='INFO'):
        prefix = f"[{tag}]"
        if level == 'WARN':
            prefix += "[WARN]"
        elif level == 'ERROR':
            prefix += "[ERROR]"
        elif level == 'OK':
            prefix += "[OK]"
        line = f"{prefix} {message}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            self.log(message)
        except Exception:
            pass

    def _audit_exception(self, context, exc=None, tag='AUDIT'):
        tb = traceback.format_exc()
        try:
            print(f"[{tag}][ERROR] {context}: {exc}")
            print(tb)
        except Exception:
            pass
        try:
            self.log(f"❌ {context}: {exc}")
            self.log(tb)
        except Exception:
            pass

    # ==================================================================
    # Хуки
    # ==================================================================
    def _after_load_hook(self):
        self._pair_class_cache = {}
        self._last_cluster_data = None
        self._audit_log(
            f"Датасет загружен: {len(self.pairs)} пар, "
            f"тип={self.dataset_type}, формат={self.input_format}",
            tag="ANALYSIS", level='OK')
        self.apply_filters_and_update_metrics()

    def _after_type_change_hook(self, new_type):
        self._pair_class_cache = {}
        self._last_cluster_data = None
        self._audit_log(f"Смена dataset_type → {new_type}",
                        tag="ANALYSIS")
        self._sync_analyzer()
        if new_type == 2:
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()

    def _sync_analyzer(self):
        self.analyzer.set_dataset_type(self.dataset_type)
        self.analyzer.set_input_format(self.input_format)

    def toggle_log(self, checked):
        self.log_widget.setVisible(bool(checked))
        self.main_v_splitter.setSizes([500, 200] if checked else [700, 0])
        self.toggle_log_btn.setText(
            "Скрыть лог" if checked else "Показать лог")

    def get_class_names_from_table(self):
        if self.dataset_type == 2:
            return {}
        return DatasetAnalyzer.get_class_names_included(self.class_table)

    def _build_extra_info(self):
        if not self.annotation_types_stats:
            return ""
        return "  [" + ", ".join(f"{k}:{v}"
                                  for k, v in
                                  self.annotation_types_stats.items()) + "]"

    def _get_analysis_pairs(self):
        if self.filtered_pairs:
            self._audit_log(
                f"Используются отфильтрованные пары: "
                f"{len(self.filtered_pairs)} (из {len(self.pairs)})",
                tag="ANALYSIS")
            return self.filtered_pairs
        return self.pairs

    def _on_include_empty_toggled(self, *_):
        include_empty = bool(self.include_empty_annotations.isChecked())
        self._audit_log(
            f"Переключён include_empty_annotations: {include_empty}",
            tag="ANALYSIS")
        super()._on_include_empty_toggled()
        if self.pairs:
            self.apply_filters_and_update_metrics()

    # ==================================================================
    # Легенда классов
    # ==================================================================
    def update_included_classes(self):
        super().update_included_classes()
        try:
            self._update_class_legend()
        except Exception:
            pass

    def _update_class_legend(self, class_counts=None):
        if not hasattr(self, 'class_legend_layout'):
            return

        while self.class_legend_layout.count() > 0:
            it = self.class_legend_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        if self.class_table.rowCount() == 0:
            self.class_legend_label.setVisible(False)
            self.class_legend_scroll.setVisible(False)
            return

        self.class_legend_label.setVisible(True)
        self.class_legend_scroll.setVisible(True)

        name_by_orig = DatasetAnalyzer.get_class_names_by_orig_id(
            self.class_table)
        for row in range(self.class_table.rowCount()):
            try:
                orig = int(self.class_table.item(row, 1).text())
            except (ValueError, TypeError, AttributeError):
                continue

            cb = self.class_table.cellWidget(row, 4)
            is_included = bool(cb and cb.isChecked())
            name = name_by_orig.get(orig, str(orig))

            count_str = ""
            if class_counts and orig in class_counts:
                count_str = f" ({class_counts[orig]})"

            rw = QWidget()
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)

            box = QLabel()
            box.setFixedSize(14, 14)
            try:
                bgr = get_class_color(orig)
                if isinstance(bgr, (tuple, list)) and len(bgr) >= 3:
                    r_, g_, b_ = int(bgr[2]), int(bgr[1]), int(bgr[0])
                    box.setStyleSheet(
                        f"background-color: rgb({r_},{g_},{b_}); "
                        f"border: 1px solid #333;")
                else:
                    box.setStyleSheet(
                        "background-color: #ccc; border: 1px solid #333;")
            except Exception:
                box.setStyleSheet(
                    "background-color: #ccc; border: 1px solid #333;")
            rl.addWidget(box)

            text = f"{orig}: {name}{count_str}"
            if not is_included:
                text += " (искл.)"
            lbl = QLabel(text)
            if not is_included:
                lbl.setStyleSheet(
                    "color: #888; text-decoration: line-through;")
            rl.addWidget(lbl)
            rl.addStretch()

            self.class_legend_layout.addWidget(rw)

        self.class_legend_layout.addStretch()

    # ==================================================================
    # Фильтрация
    # ==================================================================
    def apply_filters_and_update_metrics(self):
        self._sync_analyzer()
        if not self.pairs:
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()
            self.pair_count_label.setText(
                f"Всего пар: 0{self._build_extra_info()}")
            return

        included_class_ids = DatasetAnalyzer.get_included_class_ids(
            self.class_table)
        total_classes = self.class_table.rowCount()
        params = FilterParams(
            min_w=self.filter_min_w.value(),
            min_h=self.filter_min_h.value(),
            min_objs=self.filter_min_objs.value(),
            max_objs=self.filter_max_objs.value(),
            min_area=self.filter_min_area.value(),
            max_area=self.filter_max_area.value(),
        )
        include_empty = bool(self.include_empty_annotations.isChecked())

        self._audit_log(
            f"Старт фильтрации: pairs={len(self.pairs)}, "
            f"classes={len(included_class_ids)}/{total_classes}, "
            f"include_empty={include_empty}",
            tag="ANALYSIS")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.pairs))
        self.progress_bar.setValue(0)

        def on_progress(c, t):
            self.progress_bar.setValue(c)
            if c % 50 == 0:
                QApplication.processEvents()

        result = self.analyzer.filter_pairs(
            self.pairs, included_class_ids, total_classes, params,
            progress_cb=on_progress, include_empty=include_empty)
        self.progress_bar.setVisible(False)

        if result['error']:
            self._audit_log(f"Ошибка фильтрации: {result['error']}",
                            tag="ANALYSIS", level='ERROR')
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()
            self.pair_count_label.setText(
                f"Всего пар: {len(self.pairs)} | после фильтрации: 0")
            return

        self.filtered_pairs = result['filtered_pairs']
        empty_imgs = result.get('empty_images', 0)
        self._audit_log(
            f"Фильтрация: {len(self.pairs)} → "
            f"{len(self.filtered_pairs)} пар, "
            f"объектов={result['total_objects']}, "
            f"пустых={empty_imgs}, "
            f"классов={len(result['class_counts'])}",
            tag="ANALYSIS", level='OK')

        label = (f"Всего пар: {len(self.pairs)} | после фильтрации: "
                 f"{len(self.filtered_pairs)}")
        if empty_imgs:
            label += f" (пустых: {empty_imgs})"
        label += self._build_extra_info()
        self.pair_count_label.setText(label)

        self._render_metrics(result['class_counts'],
                             result['total_objects'],
                             result['object_areas'],
                             result['class_areas'])
        self._update_imbalance_table(result['class_counts'])
        self._update_class_legend(result['class_counts'])
        self._restyle_and_sort_file_list()

    def _update_imbalance_table(self, class_counts):
        self.imbalance_table.setRowCount(0)
        if not class_counts:
            self.imbalance_summary_label.setText("Нет классов.")
            self._last_imbalance = None
            return
        res = ClassImbalanceAnalyzer.analyze(class_counts)
        rows = [
            ("Классов", str(len(class_counts))),
            ("Imbalance ratio", f"{res['imbalance_ratio']:.2f}:1"),
            ("Gini", f"{res['gini']:.4f}"),
            ("Entropy", f"{res['entropy']:.4f}"),
            ("Majority class", str(res['majority_class'])),
            ("Minority class", str(res['minority_class'])),
        ]
        TIPS = {
            "Классов":
                "Число классов, реально присутствующих в отфильтрованном "
                "наборе.",
            "Imbalance ratio":
                "Отношение max/min количества объектов между классами.\n"
                "≥3:1 — умеренный; ≥10:1 — сильный; ≥100:1 — критический.",
            "Gini":
                "Коэффициент Джини (0..1). 0 — равномерно, "
                "1 — всё в одном классе.",
            "Entropy":
                "Энтропия Шеннона. Чем выше — тем равномернее. "
                "Максимум = ln(N).",
            "Majority class":
                "Класс с наибольшим числом объектов.",
            "Minority class":
                "Класс с наименьшим числом объектов.",
        }
        self.imbalance_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ik = QTableWidgetItem(k)
            ik.setToolTip(TIPS.get(k, ""))
            iv = QTableWidgetItem(v)
            iv.setToolTip(TIPS.get(k, ""))
            self.imbalance_table.setItem(i, 0, ik)
            self.imbalance_table.setItem(i, 1, iv)
        self.imbalance_summary_label.setText(res['warning'])
        self._last_imbalance = res
        self._audit_log(
            f"Дисбаланс: ratio={res['imbalance_ratio']:.2f}, "
            f"gini={res['gini']:.4f}, entropy={res['entropy']:.4f}",
            tag="IMBALANCE", level='OK')

    def _render_metrics(self, class_counts, total_objects,
                        object_areas, class_areas=None):
        if class_areas is None:
            class_areas = {}
        name_by_orig = DatasetAnalyzer.get_class_names_by_orig_id(
            self.class_table)
        DatasetAnalyzer.render_metrics(
            self.metrics_summary_label, self.class_stats_table,
            self.ax, self.ax_classes, self.canvas,
            self.filtered_pairs, self.pairs,
            class_counts, total_objects, object_areas, class_areas,
            name_by_orig,
            ax_pie=getattr(self, 'ax_pie', None),
            ax_bar=getattr(self, 'ax_bar', None))

    # ==================================================================
    # Перекраска + сортировка file_list
    # ==================================================================
    def _get_pair_dominant_class(self, pair):
        key = (pair[0], pair[1] if len(pair) > 1 else None)
        if key in self._pair_class_cache:
            return self._pair_class_cache[key]

        cls_id = None
        lbl = pair[1] if len(pair) > 1 else None
        try:
            if self.dataset_type == 0 and lbl and os.path.exists(lbl):
                anns = load_annotations(
                    lbl, 1, 1, format_hint=self.input_format)
                cls_ids = [int(a[1]) for a in anns if len(a) >= 2]
                if cls_ids:
                    cls_id = Counter(cls_ids).most_common(1)[0][0]
            elif self.dataset_type == 1 and lbl and os.path.exists(lbl):
                import cv2
                import numpy as np
                mask = cv2.imread(lbl, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    u = [int(c) for c in np.unique(mask) if c != 0]
                    if u:
                        cls_id = u[0]
        except Exception:
            pass
        self._pair_class_cache[key] = cls_id
        return cls_id

    def _restyle_and_sort_file_list(self):
        if not self.pairs or not hasattr(self, 'file_list'):
            return

        filtered_set = set()
        for p in (self.filtered_pairs or []):
            img = p[0] if isinstance(p, (tuple, list)) and p else p
            aux = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else None
            filtered_set.add((img, aux))

        self.file_list.blockSignals(True)
        try:
            items = [self.file_list.takeItem(0)
                     for _ in range(self.file_list.count())]

            kept, rejected = [], []
            for item in items:
                if item is None:
                    continue
                pair = item.data(Qt.UserRole + 1)
                if not pair:
                    item.setForeground(QColor(0, 0, 0))
                    kept.append(item)
                    continue

                if pair in filtered_set:
                    cls_id = self._get_pair_dominant_class(pair)
                    if cls_id is not None:
                        try:
                            bgr = get_class_color(cls_id)
                            if (isinstance(bgr, (tuple, list))
                                    and len(bgr) >= 3):
                                b, g, r = (int(bgr[0]), int(bgr[1]),
                                           int(bgr[2]))
                                item.setForeground(QColor(r, g, b))
                            else:
                                item.setForeground(QColor(0, 0, 0))
                        except Exception:
                            item.setForeground(QColor(0, 0, 0))
                    else:
                        item.setForeground(QColor(0, 0, 0))
                    kept.append(item)
                else:
                    item.setForeground(QColor(128, 128, 128))
                    rejected.append(item)

            for it in kept:
                self.file_list.addItem(it)
            for it in rejected:
                self.file_list.addItem(it)
        finally:
            self.file_list.blockSignals(False)

    # ==================================================================
    # Экспорт
    # ==================================================================
    def export_filtered_dataset(self):
        if not self.filtered_pairs:
            QMessageBox.warning(self, "Ошибка",
                                "Нет данных для экспорта.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Папка для датасета")
        if not out_dir:
            return
        try:
            class_names = DatasetAnalyzer.get_class_names_included(
                self.class_table)

            included_class_ids = None
            if self.chk_exclude_classes_from_files.isChecked():
                included_class_ids = DatasetAnalyzer.get_included_class_ids(
                    self.class_table)
                self._audit_log(
                    f"Экспорт с фильтрацией .txt по классам: "
                    f"оставлены только {sorted(included_class_ids)}",
                    tag="EXPORT")

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(self.filtered_pairs))

            def on_progress(c, t):
                self.progress_bar.setValue(c)
                if c % 50 == 0:
                    QApplication.processEvents()

            DatasetAnalyzer.export_subset(
                self.filtered_pairs, out_dir, class_names,
                progress_cb=on_progress, log_cb=self.log,
                included_class_ids=included_class_ids)

            self.progress_bar.setVisible(False)
            self._last_export_dir = out_dir
            self.btn_open_last_export.setEnabled(True)
            self._audit_log(
                f"Экспорт: {len(self.filtered_pairs)} снимков → {out_dir}",
                tag="EXPORT", level='OK')
            QMessageBox.information(
                self, "Успех",
                f"Датасет создан: {out_dir}\n"
                f"Снимков: {len(self.filtered_pairs)}")
        except Exception as e:
            self.progress_bar.setVisible(False)
            self._audit_exception("Ошибка экспорта", e, tag="EXPORT")

    # ==================================================================
    # Качество
    # ==================================================================
    def run_quality_analysis(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Нет пар для анализа.")
            return
        self._stop_requested = False
        self.btn_run_quality.setEnabled(False)
        self.btn_stop_quality.setEnabled(True)
        self._audit_log(f"Старт анализа качества: {len(pairs)} пар.",
                        tag="QUALITY")

        def progress(c, t):
            QApplication.processEvents()

        def cancel_check():
            return self._stop_requested

        try:
            res = ResolutionAnalyzer.analyze(
                pairs, progress_cb=progress, cancel_check=cancel_check)
            bright = BrightnessAnalyzer.analyze(
                pairs, progress_cb=progress, cancel_check=cancel_check)
        except Exception as e:
            self._audit_exception("Ошибка анализа качества", e,
                                  tag="QUALITY")
            self.btn_run_quality.setEnabled(True)
            self.btn_stop_quality.setEnabled(False)
            return

        self.btn_run_quality.setEnabled(True)
        self.btn_stop_quality.setEnabled(False)

        if self._stop_requested:
            self._audit_log("Анализ качества остановлен.",
                            tag="QUALITY", level='WARN')
        if res is None:
            self.quality_summary_label.setText(
                "Не удалось прочитать изображения.")
            return

        self._last_quality = {'resolution': res, 'brightness': bright}

        lines = [f"<b>Разрешение:</b> медиана {res['median_w']:.0f}×"
                 f"{res['median_h']:.0f}, "
                 f"<b>imgsz</b> ≈ {res['recommended_imgsz']}"]
        self._audit_log(
            f"Разрешение: {res['median_w']:.0f}x{res['median_h']:.0f}, "
            f"imgsz≈{res['recommended_imgsz']}", tag="QUALITY")

        if bright:
            lines.append(
                f"<b>Яркость:</b> mean {bright['mean_brightness']:.1f}; "
                f"тёмных: {bright['dark_count']}, "
                f"пересвеченных: {bright['bright_count']}")
            if bright['dark_count'] or bright['bright_count']:
                lines.append("⚠️ Проверьте тёмные/пересвеченные.")
            else:
                lines.append("✅ Освещённость в норме.")
            self._audit_log(
                f"Яркость: mean={bright['mean_brightness']:.1f}, "
                f"dark={bright['dark_count']}, "
                f"bright={bright['bright_count']}", tag="QUALITY")

        self.quality_summary_label.setText("<br>".join(lines))
        self._draw_quality_plots(res, bright)
        self._audit_log("Анализ качества завершён.",
                        tag="QUALITY", level='OK')

    def stop_quality_analysis(self):
        self._stop_requested = True
        QApplication.processEvents()

    def _draw_quality_plots(self, res, bright):
        BA = BrightnessAnalyzer
        for ax in (self.quality_ax1, self.quality_ax2,
                   self.quality_ax3, self.quality_ax4):
            ax.clear()

        self.quality_ax1.hist(res['widths'], bins=50,
                              color='skyblue', edgecolor='black')
        self.quality_ax1.set_title("Ширина (px)")
        self.quality_ax1.set_ylabel("Кол-во")

        self.quality_ax2.hist(res['heights'], bins=50,
                              color='lightgreen', edgecolor='black')
        self.quality_ax2.set_title("Высота (px)")

        self.quality_ax3.hist(res['aspects'], bins=50,
                              color='salmon', edgecolor='black')
        self.quality_ax3.set_title("Соотношение сторон W/H")

        if bright:
            self.quality_ax4.hist(bright['means'], bins=50,
                                  color='gold', edgecolor='black')
            self.quality_ax4.axvline(BA.DARK_BOUND, color='blue',
                                     ls='--', label='Dark')
            self.quality_ax4.axvline(BA.BRIGHT_BOUND, color='red',
                                     ls='--', label='Bright')
            self.quality_ax4.set_title("Средняя яркость")
            self.quality_ax4.legend(fontsize=7)
        else:
            self.quality_ax4.text(0.5, 0.5, "Нет данных",
                                  ha='center', va='center')
            self.quality_ax4.set_title("Яркость")

        self.quality_figure.tight_layout()
        self.quality_canvas.draw()

    # ==================================================================
    # Дубликаты
    # ==================================================================
    def run_duplicate_analysis(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Нет пар для анализа.")
            return
        from utils.dataset_audit_logic import (
            DuplicateDetector, LeakageChecker)
        self._stop_requested = False
        self.btn_run_duplicates.setEnabled(False)
        self.btn_stop_duplicates.setEnabled(True)
        self._audit_log(
            f"Старт поиска дубликатов: {len(pairs)} пар.", tag="DUP")
        self.dup_table.setRowCount(0)
        self._dup_groups_by_row = []

        def progress(c, t):
            if c % 20 == 0:
                QApplication.processEvents()

        try:
            dups = DuplicateDetector.find_duplicates(
                pairs, phash_threshold=8, progress_cb=progress,
                cancel_check=lambda: self._stop_requested)
        except Exception as e:
            self._audit_exception("Ошибка поиска дубликатов", e,
                                  tag="DUP")
            self.btn_run_duplicates.setEnabled(True)
            self.btn_stop_duplicates.setEnabled(False)
            return

        self.btn_run_duplicates.setEnabled(True)
        self.btn_stop_duplicates.setEnabled(False)

        if dups.get('cancelled'):
            self._audit_log("Поиск дубликатов остановлен.",
                            tag="DUP", level='WARN')

        exact = dups.get('exact', [])
        near = dups.get('near', [])
        leak_report = LeakageChecker.check(exact + near, pairs)
        leaked = leak_report.get('leaked_count', 0)

        self._last_dup = {
            'exact': exact, 'near': near,
            'leak': leak_report,
            'cancelled': dups.get('cancelled', False),
        }

        self._audit_log(
            f"Дубликаты: exact={len(exact)} групп, "
            f"near={len(near)} групп, утечка={leaked}",
            tag="DUP", level='ERROR' if leaked else 'OK')

        self._clear_duplicate_highlights()
        self._highlight_duplicates_in_file_list(dups, leak_report)

        rows = []
        self._dup_groups_by_row = []
        for kind, groups in (('MD5', exact), ('pHash', near)):
            for g_idx, group in enumerate(groups, 1):
                splits = Counter(
                    LeakageChecker._infer_split(p) for p in group)
                leak = len({s for s in splits
                            if s != 'unknown'}) >= 2
                files_short = ", ".join(
                    os.path.basename(p) for p in group[:3])
                if len(group) > 3:
                    files_short += f" … (+{len(group) - 3})"
                rows.append((
                    f"{kind} #{g_idx}",
                    str(len(group)),
                    ", ".join(f"{k}:{v}" for k, v in splits.items()),
                    files_short,
                    "⚠️" if leak else "✅",
                    leak,
                ))
                self._dup_groups_by_row.append(list(group))

        self.dup_table.setRowCount(len(rows))
        for r, (kind, n, splits_str, files, leak_mark, leak) in enumerate(rows):
            for c, txt in enumerate(
                    (kind, n, splits_str, files, leak_mark)):
                it = QTableWidgetItem(txt)
                it.setToolTip(
                    f"{kind}\n"
                    f"Изображений: {n}\n"
                    f"Сплиты: {splits_str}\n"
                    f"Файлы: {files}\n"
                    f"Утечка: {'ДА' if leak else 'нет'}\n"
                    f"Клик — выделить пары; двойной клик — открыть.")
                if leak:
                    it.setBackground(QColor(255, 220, 220))
                self.dup_table.setItem(r, c, it)

        lines = [f"Точных групп: {len(exact)}, "
                 f"почти-групп: {len(near)}."]
        if leaked:
            lines.append(f"🚨 Утечка в {leaked} группах — "
                         f"проверьте сплиты!")
        else:
            lines.append("✅ Утечек нет.")
        if exact or near:
            lines.append(
                "Найденные изображения подсвечены в списке пар слева. "
                "Клик по строке таблицы выделяет пары группы.")
        self.dup_summary_label.setText("<br>".join(lines))

        for rg in leak_report['groups']:
            if rg['leak']:
                names = ", ".join(os.path.basename(p)
                                  for p in rg['paths'][:3])
                self._audit_log(
                    f"  утечка {rg['splits']}: {names}", tag="DUP")

    def stop_duplicate_analysis(self):
        self._stop_requested = True
        QApplication.processEvents()

    def _clear_duplicate_highlights(self):
        if not hasattr(self, 'file_list'):
            return
        self.file_list.blockSignals(True)
        try:
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                it.setData(Qt.BackgroundRole, None)
                it.setToolTip(
                    "Ctrl+C — копировать.\n"
                    "Двойной клик — открыть пару.")
        finally:
            self.file_list.blockSignals(False)

    def _highlight_duplicates_in_file_list(self, dups, leak_report):
        if not hasattr(self, 'file_list'):
            return

        exact_paths = set()
        for g in dups.get('exact', []):
            exact_paths.update(g)

        near_paths = set()
        for g in dups.get('near', []):
            near_paths.update(g)
        near_paths -= exact_paths

        leak_paths = set()
        for g in leak_report.get('groups', []):
            if g.get('leak'):
                leak_paths.update(g['paths'])

        self.file_list.blockSignals(True)
        try:
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                pair = it.data(Qt.UserRole + 1)
                if not pair:
                    continue
                p = pair[0]
                if p in leak_paths:
                    it.setBackground(QColor(255, 180, 180))
                    it.setToolTip(
                        (it.toolTip() or "") +
                        "\n🚨 Утечка между сплитами (train/val/test)")
                elif p in exact_paths:
                    it.setBackground(QColor(255, 210, 210))
                    it.setToolTip(
                        (it.toolTip() or "") +
                        "\n🔴 Точный дубликат (MD5)")
                elif p in near_paths:
                    it.setBackground(QColor(255, 235, 200))
                    it.setToolTip(
                        (it.toolTip() or "") +
                        "\n🟠 Похожее изображение (pHash ≤ 8)")
        finally:
            self.file_list.blockSignals(False)

    def _on_dup_row_selected(self):
        """Клик по строке таблицы дубликатов → выделить её пары."""
        if not hasattr(self, 'file_list'):
            return
        try:
            sel_model = self.dup_table.selectionModel()
            if sel_model is None:
                return
            rows = sel_model.selectedRows()
        except Exception:
            return

        if not rows:
            # Пустое выделение → снимаем в file_list
            self.file_list.blockSignals(True)
            try:
                self.file_list.clearSelection()
            finally:
                self.file_list.blockSignals(False)
            return

        row = rows[0].row()
        groups = getattr(self, '_dup_groups_by_row', [])
        if row < 0 or row >= len(groups):
            return
        paths = set(groups[row])
        if not paths:
            return

        self.file_list.blockSignals(True)
        try:
            self.file_list.clearSelection()
            first = None
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                pair = it.data(Qt.UserRole + 1)
                if pair and pair[0] in paths:
                    it.setSelected(True)
                    if first is None:
                        first = it
            if first is not None:
                self.file_list.scrollToItem(
                    first, QAbstractItemView.PositionAtCenter)
        finally:
            self.file_list.blockSignals(False)

        self._audit_log(
            f"[DUP] Выбрана группа #{row + 1} ({len(paths)} пар).",
            tag="DUP")

    def _on_dup_row_double_clicked(self, row, col):
        groups = getattr(self, '_dup_groups_by_row', [])
        if row < 0 or row >= len(groups):
            return
        paths = groups[row]
        if not paths:
            return
        target = paths[0]
        for p in self.pairs:
            if p[0] == target:
                self._open_pair_validation(p)
                return

    # ==================================================================
    # Тепловая карта
    # ==================================================================
    def run_heatmap_analysis(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Нет пар для анализа.")
            return
        from utils.dataset_audit_logic import (
            ObjectDistributionHeatmap)
        self._stop_requested = False
        self.btn_run_heatmap.setEnabled(False)
        self.btn_stop_heatmap.setEnabled(True)
        grid_size = int(self.heatmap_grid_spin.value())
        self._audit_log(
            f"Старт тепловой карты: pairs={len(pairs)}, "
            f"grid={grid_size}", tag="HEATMAP")

        def progress(c, t):
            if c % 20 == 0:
                QApplication.processEvents()

        try:
            result = ObjectDistributionHeatmap.compute(
                pairs, dataset_type=self.dataset_type,
                grid_size=grid_size, progress_cb=progress,
                cancel_check=lambda: self._stop_requested)
        except Exception as e:
            self._audit_exception("Ошибка тепловой карты", e,
                                  tag="HEATMAP")
            self.btn_run_heatmap.setEnabled(True)
            self.btn_stop_heatmap.setEnabled(False)
            return

        self.btn_run_heatmap.setEnabled(True)
        self.btn_stop_heatmap.setEnabled(False)

        if result.get('cancelled'):
            self._audit_log("Тепловая карта остановлена.",
                            tag="HEATMAP", level='WARN')

        self._last_heatmap = result
        heatmap = result['heatmap']
        total_objs = result['total_objects']
        grid = result['grid_size']

        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except Exception:
                pass
            self._heatmap_colorbar = None

        self.heatmap_ax.clear()
        if total_objs == 0:
            self.heatmap_ax.text(0.5, 0.5, "Нет объектов",
                                 ha='center', va='center')
            self.heatmap_ax.set_title("Тепловая карта: пусто", fontsize=9)
            self._audit_log("Тепловая карта: нет объектов.",
                            tag="HEATMAP", level='WARN')
        else:
            im = self.heatmap_ax.imshow(
                heatmap, cmap='hot', interpolation='nearest',
                extent=[0, 100, 100, 0], aspect='auto')
            self.heatmap_ax.set_xlabel(
                "X (% ширины, 0 = левый край)", fontsize=8)
            self.heatmap_ax.set_ylabel(
                "Y (% высоты, 0 = верх)", fontsize=8)
            self.heatmap_ax.set_title(
                f"Распределение центров объектов "
                f"(всего {total_objs}, сетка {grid}×{grid})",
                fontsize=9)
            cbar = self.heatmap_figure.colorbar(
                im, ax=self.heatmap_ax, fraction=0.046, pad=0.04)
            cbar.set_label("Число объектов в ячейке",
                           rotation=270, labelpad=16, fontsize=8)
            self._heatmap_colorbar = cbar
            self._audit_log(
                f"Тепловая карта: объектов={total_objs}, "
                f"grid={grid}x{grid}", tag="HEATMAP", level='OK')

        try:
            self.heatmap_figure.tight_layout()
        except Exception:
            pass
        self.heatmap_canvas.draw()
        self.heatmap_summary_label.setText(
            f"Объектов: {total_objs}. Сетка {grid}×{grid}. "
            f"Оси — в % от размера изображения; цвет — число "
            f"центров объектов в ячейке.")

    def stop_heatmap_analysis(self):
        self._stop_requested = True
        QApplication.processEvents()

    # ==================================================================
    # Аудит модели
    # ==================================================================
    def load_audit_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Модель YOLO", "",
            "YOLO weights (*.pt *.onnx *.engine);;Все файлы (*)")
        if not path:
            return
        self._audit_model_path = path
        self.audit_model_edit.setText(path)
        self._audit_model_cache = None
        self._audit_model_cache_path = None
        self._audit_log(f"Модель выбрана: {path}",
                        tag="AUDIT", level='OK')

    def run_loss_inspection(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Нет пар для анализа.")
            return
        if not self._audit_model_path:
            QMessageBox.warning(self, "Модель не выбрана",
                                "Загрузите YOLO-модель.")
            return
        if self._audit_thread and self._audit_thread.isRunning():
            QMessageBox.information(self, "Идёт аудит",
                                    "Дождитесь завершения.")
            return

        from utils.dataset_audit_logic import LossInspectionThread
        iou_thr = float(self.audit_iou_spin.value())
        conf_thr = float(self.audit_conf_spin.value())
        imgsz = int(self.audit_imgsz_spin.value())

        self._audit_log(
            f"Старт аудита: model={self._audit_model_path}, "
            f"iou={iou_thr}, conf={conf_thr}, imgsz={imgsz}, "
            f"device={self._device}, pairs={len(pairs)}", tag="AUDIT")

        self.audit_table.setRowCount(0)
        self.audit_summary_label.setText("Аудит запущен…")
        self.btn_run_audit.setEnabled(False)
        self.btn_stop_audit.setEnabled(True)

        self._audit_thread = LossInspectionThread(
            pairs, self._audit_model_path,
            dataset_type=self.dataset_type,
            iou_threshold=iou_thr, conf_threshold=conf_thr,
            imgsz=imgsz,
            input_format=self.input_format,
            device=self._device)
        self._audit_thread.log_signal.connect(self.log)
        self._audit_thread.log_signal.connect(
            lambda m: print(f"[AUDIT][THREAD] {m}", flush=True))
        self._audit_thread.progress.connect(self._on_audit_progress)
        self._audit_thread.finished_signal.connect(self._on_audit_finished)
        self._audit_thread.start()

    def stop_loss_inspection(self):
        if self._audit_thread and self._audit_thread.isRunning():
            self._audit_thread.cancel()
            self._audit_log("Аудит: запрошена остановка.",
                            tag="AUDIT", level='WARN')

    def _on_audit_progress(self, c, t):
        self.audit_summary_label.setText(f"⏳ Аудит: {c}/{t}…")
        QApplication.processEvents()

    def _on_audit_finished(self, result):
        self.btn_run_audit.setEnabled(True)
        self.btn_stop_audit.setEnabled(False)
        err = result.get('error')
        if err:
            self.audit_summary_label.setText(f"❌ {err}")
            self._audit_log(f"Аудит ошибка: {err}",
                            tag="AUDIT", level='ERROR')
            return

        suspicious = result.get('suspicious', [])
        processed = result.get('processed', 0)
        total = result.get('total', 0)
        errors = result.get('errors', 0)

        self.audit_table.setRowCount(len(suspicious))
        for row, item in enumerate(suspicious):
            self.audit_table.setItem(row, 0, QTableWidgetItem(
                os.path.basename(item['img_path'])))
            self.audit_table.setItem(row, 1, QTableWidgetItem(
                str(item['gt_count'])))
            self.audit_table.setItem(row, 2, QTableWidgetItem(
                str(item['pred_count'])))
            self.audit_table.setItem(row, 3, QTableWidgetItem(
                f"{item['mean_iou']:.3f}"))

        self._last_audit_results = suspicious
        self._last_audit_meta = {
            'model': self._audit_model_path,
            'iou_thr': float(self.audit_iou_spin.value()),
            'conf_thr': float(self.audit_conf_spin.value()),
            'imgsz': int(self.audit_imgsz_spin.value()),
            'processed': processed, 'total': total,
            'errors': errors, 'suspicious_count': len(suspicious),
        }

        self.audit_summary_label.setText(
            f"Обработано: {processed}/{total}. "
            f"Подозрительных: {len(suspicious)}. Ошибок: {errors}. "
            f"Двойной клик — открыть двойное окно сравнения "
            f"(GT ↔ Pred).")
        self._audit_log(
            f"Аудит завершён: processed={processed}/{total}, "
            f"errors={errors}, подозрительных={len(suspicious)}",
            tag="AUDIT", level='OK')

        for item in suspicious[:5]:
            self._audit_log(
                f"  {os.path.basename(item['img_path'])}: "
                f"gt={item['gt_count']}, pred={item['pred_count']}, "
                f"iou={item['mean_iou']:.3f}", tag="AUDIT")

    def _open_audit_row(self, row, col):
        if row < 0 or row >= len(self._last_audit_results):
            return
        item = self._last_audit_results[row]
        self._open_audit_comparison(item)


    def _get_audit_model(self):
        path = self._audit_model_path
        if not path:
            return None
        if (self._audit_model_cache is not None
                and self._audit_model_cache_path == path):
            return self._audit_model_cache
        try:
            from ultralytics import YOLO
            m = YOLO(path)
            if self._device.startswith('cuda'):
                try:
                    m.to(self._device)
                except Exception as e:
                    self._audit_log(
                        f"[AUDIT] Не удалось переместить модель на "
                        f"{self._device} ({e}) — остаёмся на CPU",
                        tag="AUDIT", level='WARN')
                    try:
                        m.to('cpu')
                    except Exception:
                        pass
            self._audit_model_cache = m
            self._audit_model_cache_path = path
            self._audit_log(
                f"[AUDIT] Модель загружена в кэш: "
                f"{os.path.basename(path)} (device: {self._device})",
                tag="AUDIT")
            return self._audit_model_cache
        except Exception as e:
            self._audit_log(
                f"[AUDIT] Не удалось загрузить модель: {e}",
                tag="AUDIT", level='ERROR')
            return None


    # ==================================================================
    # Открытие папок в проводнике
    # ==================================================================
    def _open_folder_in_explorer(self, path):
        if not path:
            QMessageBox.information(
                self, "Нет данных",
                "Сначала выполните действие (экспорт / отчёт).")
            return
        if not os.path.isdir(path):
            QMessageBox.warning(
                self, "Папка не найдена",
                f"Путь не существует:\n{path}")
            return
        try:
            if _sys.platform.startswith('win'):
                os.startfile(path)  # noqa: S606
            elif _sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
            self._audit_log(
                f"Открываю папку: {path}", tag="FS", level='OK')
        except Exception as e:
            QMessageBox.warning(
                self, "Не удалось открыть папку",
                f"{path}\n{e}")

    def open_last_export_folder(self):
        self._open_folder_in_explorer(self._last_export_dir)

    def open_report_folder(self):
        self._open_folder_in_explorer(self._last_report_dir)

    def _open_audit_comparison(self, item):
        import cv2
        import numpy as np

        img_path = item['img_path']
        lbl_path = item.get('lbl_path')
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Нет файла", img_path)
            return

        base = cv2.imread(img_path)
        if base is None:
            QMessageBox.warning(self, "Не удалось прочитать", img_path)
            return

        results = None
        model = self._get_audit_model()
        if model is not None:
            try:
                results = model(
                    img_path,
                    conf=float(self.audit_conf_spin.value()),
                    imgsz=int(self.audit_imgsz_spin.value()),
                    device=self._device,
                    verbose=False)
            except Exception as e:
                self._audit_log(f"[AUDIT] Ошибка инференса: {e}",
                                tag="AUDIT", level='ERROR')

        h, w = base.shape[:2]
        img_gt = base.copy()
        img_pred = base.copy()

        try:
            if (self.dataset_type == 0
                    and lbl_path and os.path.exists(lbl_path)):
                anns = load_annotations(
                    lbl_path, 1, 1, format_hint=self.input_format)
                for ann in anns:
                    if not ann:
                        continue
                    if ann[0] == 'detect':
                        _, cls, cx, cy, bw, bh = ann
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2),
                                      (0, 200, 0), 2)
                        cv2.putText(img_gt, f"c{cls}",
                                    (x1, max(y1 - 5, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 0), 1)
                    elif ann[0] in ('segment', 'obb'):
                        pts = ann[2]
                        if not pts or len(pts) < 6:
                            continue
                        arr = np.array(
                            [[int(pts[i] * w), int(pts[i + 1] * h)]
                             for i in range(0, len(pts), 2)],
                            dtype=np.int32)
                        cv2.polylines(img_gt, [arr], True,
                                      (0, 200, 0), 2)
            elif (self.dataset_type == 1
                    and lbl_path and os.path.exists(lbl_path)):
                mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
                if mask is not None and mask.ndim == 2:
                    overlay = np.zeros_like(img_gt)
                    overlay[mask > 0] = (0, 200, 0)
                    img_gt = cv2.addWeighted(
                        img_gt, 1.0, overlay, 0.35, 0)
                    contours, _ = cv2.findContours(
                        (mask > 0).astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_gt, contours, -1,
                                     (0, 200, 0), 2)
        except Exception as e:
            self._audit_log(f"[AUDIT] Ошибка отрисовки GT: {e}",
                            tag="AUDIT")

        try:
            if results:
                r = results[0]
                if getattr(r, 'masks', None) is not None:
                    try:
                        m_arr = r.masks.data.cpu().numpy()
                        for m in m_arr:
                            m_resized = cv2.resize(
                                m, (w, h),
                                interpolation=cv2.INTER_NEAREST)
                            overlay = np.zeros_like(img_pred)
                            overlay[m_resized > 0.5] = (0, 0, 220)
                            img_pred = cv2.addWeighted(
                                img_pred, 1.0, overlay, 0.35, 0)
                    except Exception:
                        pass
                if (getattr(r, 'boxes', None) is not None
                        and len(r.boxes)):
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int,
                                             box.xyxy[0].tolist())
                        cv2.rectangle(img_pred, (x1, y1), (x2, y2),
                                      (0, 0, 220), 2)
                        cv2.putText(img_pred, f"c{cls} {conf:.2f}",
                                    (x1, max(y1 - 5, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 220), 1)
        except Exception as e:
            self._audit_log(f"[AUDIT] Ошибка отрисовки Pred: {e}",
                            tag="AUDIT")

        _, buf_gt = cv2.imencode('.png', img_gt)
        _, buf_pr = cv2.imencode('.png', img_pred)
        pix_gt = QPixmap()
        pix_gt.loadFromData(buf_gt.tobytes(), 'PNG')
        pix_pr = QPixmap()
        pix_pr.loadFromData(buf_pr.tobytes(), 'PNG')

        dlg = QDialog(self)
        dlg.setWindowTitle(
            f"Аудит: {os.path.basename(img_path)} | "
            f"GT {item['gt_count']} / Pred {item['pred_count']} | "
            f"IoU mean={item['mean_iou']:.3f} "
            f"min={item['min_iou']:.3f}")
        dlg.resize(1500, 800)
        v = QVBoxLayout(dlg)

        info = QLabel(
            f"<b>Файл:</b> {img_path}<br>"
            f"<b>Метка:</b> {lbl_path or '—'}<br>"
            f"<b>Причина:</b> {item.get('reason', '—')}<br>"
            f"<b>GT:</b> {item['gt_count']}, "
            f"<b>Pred:</b> {item['pred_count']}, "
            f"<b>Mean IoU:</b> {item['mean_iou']:.3f}, "
            f"<b>Min IoU:</b> {item['min_iou']:.3f}, "
            f"<b>Device:</b> {self._device}")
        info.setWordWrap(True)
        v.addWidget(info)

        row_h = QHBoxLayout()
        panels = (
            ("🟢 GT (оригинал)", pix_gt, "#1a7f1a"),
            ("🔴 Предсказание модели", pix_pr, "#a51b1b"),
        )
        for title, pix, color in panels:
            col = QVBoxLayout()
            header = QLabel(f"<b style='color:{color}'>{title}</b>")
            header.setAlignment(Qt.AlignCenter)
            col.addWidget(header)

            lbl = QLabel()
            if not pix.isNull():
                scaled = pix.scaled(720, 640, Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation)
                lbl.setPixmap(scaled)
            else:
                lbl.setText("(не удалось отрисовать)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "border:1px solid #888; background:#111; padding:2px;")
            lbl.setMinimumSize(300, 300)
            col.addWidget(lbl, 1)
            row_h.addLayout(col)

        v.addLayout(row_h, 1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_close)
        v.addLayout(btn_box)

        dlg.exec_()

    # ==================================================================
    # Валидация YOLO на тестовом датасете
    # ==================================================================
    def browse_val_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "data.yaml тестового датасета", "",
            "YAML (*.yaml *.yml);;Все файлы (*)")
        if not path:
            path = QFileDialog.getExistingDirectory(
                self, "Папка тестового датасета")
        if path:
            self.val_data_edit.setText(path)

    def _resolve_val_data_path(self, path):
        if not path:
            return None
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for name in ('data.yaml', 'data.yml'):
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    return cand
            return None
        return None

    def run_yolo_validation(self):
        if not self._audit_model_path:
            QMessageBox.warning(self, "Модель не выбрана",
                                "Загрузите YOLO-модель.")
            return
        raw = self.val_data_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "Нет данных",
                                "Укажите путь к data.yaml или папке.")
            return
        if not os.path.exists(raw):
            QMessageBox.warning(self, "Не найдено",
                                f"Путь не существует:\n{raw}")
            return
        data_path = self._resolve_val_data_path(raw)
        if data_path is None:
            QMessageBox.warning(
                self, "data.yaml не найден",
                f"В папке нет data.yaml:\n{raw}")
            return
        if self._val_thread and self._val_thread.isRunning():
            QMessageBox.information(self, "Идёт валидация",
                                    "Дождитесь завершения.")
            return

        split = self.val_split_combo.currentText()
        imgsz = int(self.val_imgsz_spin.value())
        conf = float(self.val_conf_spin.value())
        iou = float(self.val_iou_spin.value())

        self._audit_log(
            f"Старт валидации: model={self._audit_model_path}, "
            f"data={data_path}, split={split}, imgsz={imgsz}, "
            f"conf={conf}, iou={iou}, device={self._device}", tag="VAL")

        self.val_table.setRowCount(0)
        self.val_summary_label.setText("Валидация запущена…")
        self.btn_run_validation.setEnabled(False)
        self.btn_stop_validation.setEnabled(True)

        self._val_thread = YoloValThread(
            model_path=self._audit_model_path,
            data_path=data_path,
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self._device,
        )
        self._val_thread.log_signal.connect(self.log)
        self._val_thread.log_signal.connect(
            lambda m: print(f"[VAL][THREAD] {m}", flush=True))
        self._val_thread.finished_signal.connect(self._on_validation_finished)
        self._val_thread.start()

    def stop_yolo_validation(self):
        if self._val_thread and self._val_thread.isRunning():
            self._val_thread.request_stop()
            self._audit_log("Валидация: запрошена остановка.",
                            tag="VAL", level='WARN')

    def _on_validation_finished(self, result):
        self.btn_run_validation.setEnabled(True)
        self.btn_stop_validation.setEnabled(False)
        err = result.get('error')
        if err:
            self.val_summary_label.setText(f"❌ {err}")
            self._audit_log(f"Валидация ошибка: {err}",
                            tag="VAL", level='ERROR')
            return

        m = result.get('metrics') or {}
        summary = m.get('summary', {})
        per_class = m.get('per_class', [])
        self._last_validation = m

        def _fmt(v, nd=4):
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return "—"

        self.val_summary_label.setText(
            f"<b>Precision:</b> {_fmt(summary.get('Precision'))} | "
            f"<b>Recall:</b> {_fmt(summary.get('Recall'))} | "
            f"<b>F1:</b> {_fmt(summary.get('F1'))} | "
            f"<b>mAP50:</b> {_fmt(summary.get('mAP50'))} | "
            f"<b>mAP50-95:</b> {_fmt(summary.get('mAP50-95'))} | "
            f"<b>mAP75:</b> {_fmt(summary.get('mAP75'))}")

        self.val_table.setRowCount(len(per_class))
        for r, row in enumerate(per_class):
            for c, key in enumerate(
                    ('class', 'p', 'r', 'f1', 'ap50', 'ap')):
                it = QTableWidgetItem(str(row.get(key, '')))
                it.setToolTip(
                    f"{row.get('class', '')}\n"
                    f"Precision: {row.get('p', '')}\n"
                    f"Recall:    {row.get('r', '')}\n"
                    f"F1:        {row.get('f1', '')}\n"
                    f"mAP50:     {row.get('ap50', '')}\n"
                    f"mAP50-95:  {row.get('ap', '')}")
                self.val_table.setItem(r, c, it)

        self._audit_log(
            f"Валидация завершена: "
            f"P={_fmt(summary.get('Precision'))}, "
            f"R={_fmt(summary.get('Recall'))}, "
            f"F1={_fmt(summary.get('F1'))}, "
            f"mAP50={_fmt(summary.get('mAP50'))}, "
            f"mAP50-95={_fmt(summary.get('mAP50-95'))}",
            tag="VAL", level='OK')

    # ==================================================================
    # Кластеризация
    # ==================================================================
    def run_embedding_clustering(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Нет пар для анализа.")
            return
        if self._cluster_thread and self._cluster_thread.isRunning():
            QMessageBox.information(self, "Идёт кластеризация",
                                    "Дождитесь.")
            return

        from utils.dataset_audit_logic import (
            EmbeddingClusterThread, CLIP_AVAILABLE)
        if not CLIP_AVAILABLE:
            QMessageBox.warning(self, "CLIP не установлен",
                                "pip install open_clip_torch")
            return

        self.btn_run_cluster.setEnabled(False)
        self.btn_stop_cluster.setEnabled(True)
        self.cluster_summary_label.setText("Извлечение эмбеддингов…")
        self.cluster_ax.clear()
        self.cluster_ax.text(0.5, 0.5, "Обработка…",
                             ha='center', va='center')
        self.cluster_canvas.draw()
        self._audit_log(
            f"Старт кластеризации: pairs={len(pairs)}, "
            f"device={self._device}", tag="CLUSTER")

        self._cluster_thread = EmbeddingClusterThread(
            pairs, batch_size=16, device=self._device)
        self._cluster_thread.log_signal.connect(self.log)
        self._cluster_thread.log_signal.connect(
            lambda m: print(f"[CLUSTER][THREAD] {m}", flush=True))
        self._cluster_thread.progress.connect(self._on_cluster_progress)
        self._cluster_thread.finished_signal.connect(
            self._on_cluster_finished)
        self._cluster_thread.start()

    def stop_embedding_clustering(self):
        if self._cluster_thread and self._cluster_thread.isRunning():
            self._cluster_thread.cancel()
            self._audit_log("Кластеризация: запрошена остановка.",
                            tag="CLUSTER", level='WARN')

    def _on_cluster_progress(self, c, t):
        self.cluster_summary_label.setText(f"⏳ CLIP: {c}/{t}…")
        QApplication.processEvents()

    def _on_cluster_finished(self, result):
        from matplotlib import pyplot as plt
        self.btn_run_cluster.setEnabled(True)
        self.btn_stop_cluster.setEnabled(False)

        err = result.get('error')
        if err:
            self.cluster_summary_label.setText(f"❌ {err}")
            self._audit_log(f"Кластеризация ошибка: {err}",
                            tag="CLUSTER", level='ERROR')
            return

        paths = result['paths']
        emb_2d = np.array(result['embeddings_2d'])
        clusters = np.array(result['clusters'])

        self._last_cluster = {
            'count': len(paths),
            'n_clusters': len(set(clusters.tolist()) - {-1}),
            'noise': int((clusters == -1).sum()),
            'sizes': dict(Counter(clusters.tolist())),
        }
        self._last_cluster_data = {
            'embeddings_2d': emb_2d.tolist(),
            'paths': paths,
            'clusters': clusters.tolist(),
        }

        self.cluster_ax.clear()
        unique = sorted(set(clusters.tolist()))
        cmap = plt.get_cmap('tab20')
        for i, cl in enumerate(unique):
            mask = clusters == cl
            color = 'lightgray' if cl == -1 else cmap(i % 20)
            label = 'Шум' if cl == -1 else f'Кластер {cl}'
            self.cluster_ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                                    c=[color], label=label,
                                    s=20, alpha=0.7)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = int((clusters == -1).sum())
        self.cluster_ax.set_title(
            f"UMAP + HDBSCAN (кластеров: {n_clusters})")
        self.cluster_ax.set_xlabel("UMAP-1")
        self.cluster_ax.set_ylabel("UMAP-2")
        if len(unique) <= 20:
            self.cluster_ax.legend(fontsize=6, loc='best')
        self.cluster_figure.tight_layout()
        self.cluster_canvas.draw()

        self.cluster_summary_label.setText(
            f"Изображений: {len(paths)}. Кластеров: {n_clusters}. "
            f"Шум: {n_noise}. Клик по точке — открыть пару. "
            f"Панель инструментов — зум/pan/сохранение. "
            f"Device: {self._device}.")
        self._audit_log(
            f"Кластеризация готова: images={len(paths)}, "
            f"clusters={n_clusters}, noise={n_noise}",
            tag="CLUSTER", level='OK')

    def _on_cluster_click(self, event):
        if not self._last_cluster_data:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        tb = getattr(self, 'cluster_toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return

        emb = np.asarray(self._last_cluster_data['embeddings_2d'])
        paths = self._last_cluster_data['paths']
        if not len(emb):
            return

        d = np.sqrt((emb[:, 0] - event.xdata) ** 2
                    + (emb[:, 1] - event.ydata) ** 2)
        idx = int(np.argmin(d))
        try:
            threshold = float(self.cluster_click_radius_spin.value())
        except AttributeError:
            threshold = 0.5

        if d[idx] > threshold:
            self._audit_log(
                f"[CLUSTER] Клик мимо "
                f"(d={d[idx]:.2f} > {threshold:.2f})",
                tag="CLUSTER", level='WARN')
            return
        if idx < 0 or idx >= len(paths):
            return
        target = paths[idx]

        for p in self.pairs:
            if p[0] == target:
                self._audit_log(
                    f"[CLUSTER] Открытие: {os.path.basename(target)}",
                    tag="CLUSTER")
                self._open_pair_validation(p)
                return

    # ==================================================================
    # Результаты обучения модели
    # ==================================================================
    def browse_training_results(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Папка run/ с результатами обучения",
            self.training_results_edit.text() or "")
        if folder:
            self.training_results_edit.setText(folder)

    def load_training_results(self):
        folder = self.training_results_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            # Сразу открываем диалог выбора папки
            folder = QFileDialog.getExistingDirectory(
                self, "Папка run/ с результатами обучения")
            if not folder:
                return
            self.training_results_edit.setText(folder)

        data = TrainingResultsLoader.scan(folder)
        if data.get('error'):
            self.training_results_summary_label.setText(
                f"❌ {data['error']}")
            self._last_training_results = data
            return

        self._last_training_results = data
        self.training_results_summary_label.setText(
            TrainingResultsLoader.summary_text(data).replace("\n", "<br>"))

        # Отображение 3-х ключевых файлов
        keys_map = {
            'results': 'results.png',
            'labels': 'labels.jpg',
            'confusion': 'confusion_matrix_normalized.png',
        }
        for key, fname in keys_map.items():
            widget = getattr(self, f'training_img_{key}', None)
            if widget is None:
                continue
            path = data['main'].get(fname)
            if not path:
                widget.set_image(None)
                continue
            pix = QPixmap(path)
            widget.set_image(pix)

        self._audit_log(
            f"Результаты обучения загружены из {folder}: "
            f"main={sorted(data['main'].keys())}, "
            f"curves={len(data['curves'])}, "
            f"batches={len(data['batches'])}, "
            f"weights={sorted(data['weights'].keys())}",
            tag="TRAIN", level='OK')

    # ==================================================================
    # Отчёт (в фоне)
    # ==================================================================
    def _save_figures_for_report(self, assets_dir):
        rel = {}
        base_abs = os.path.abspath(assets_dir)
        parent_dir = os.path.dirname(base_abs)

        def _save_fig(fig, name):
            try:
                path = os.path.join(base_abs, name)
                fig.savefig(path, dpi=110, bbox_inches='tight')
                return os.path.relpath(path, parent_dir).replace('\\', '/')
            except Exception as e:
                self._audit_log(
                    f"Не удалось сохранить {name}: {e}",
                    tag="REPORT", level='WARN')
                return None

        def _copy_file(src, name, key):
            try:
                dst = os.path.join(base_abs, name)
                shutil.copy2(src, dst)
                rel[key] = os.path.relpath(dst,
                                           parent_dir).replace('\\', '/')
            except Exception as e:
                self._audit_log(
                    f"Не удалось скопировать {name}: {e}",
                    tag="REPORT", level='WARN')

        # Стандартные фигуры
        if self.class_stats_table.rowCount() > 0:
            p = _save_fig(self.figure, 'areas.png')
            if p:
                rel['areas'] = p

        if (hasattr(self, 'class_chart_figure')
                and self.class_stats_table.rowCount() > 0):
            p = _save_fig(self.class_chart_figure, 'classes.png')
            if p:
                rel['classes'] = p

        if self._last_quality:
            p = _save_fig(self.quality_figure, 'quality.png')
            if p:
                rel['quality'] = p

        if self._last_heatmap:
            p = _save_fig(self.heatmap_figure, 'heatmap.png')
            if p:
                rel['heatmap'] = p

        if self._last_cluster_data:
            p = _save_fig(self.cluster_figure, 'cluster.png')
            if p:
                rel['cluster'] = p

        # Результаты обучения
        tr = self._last_training_results
        if tr and not tr.get('error'):
            for src_name, key in (
                ('results.png', 'training_results_png'),
                ('labels.jpg', 'training_labels'),
                ('confusion_matrix_normalized.png',
                 'training_confusion'),
            ):
                src = tr['main'].get(src_name)
                if src and os.path.isfile(src):
                    _copy_file(src, src_name, key)

            for name, src in tr.get('curves', {}).items():
                if os.path.isfile(src):
                    _copy_file(src, name, f"train_curve_{name}")

            if tr.get('best_model') and os.path.isfile(tr['best_model']):
                _copy_file(tr['best_model'], 'best.pt', 'best_model')

            if (tr.get('best_model_meta')
                    and os.path.isfile(tr['best_model_meta'])):
                _copy_file(tr['best_model_meta'],
                           'best_meta.json', 'best_model_meta')

        return rel

    def generate_full_report(self):
        if not self.pairs:
            QMessageBox.warning(self, "Нет данных",
                                "Загрузите датасет.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "dataset_report.md",
            "Markdown (*.md);;Все файлы (*)")
        if not save_path:
            return

        base, _ext = os.path.splitext(save_path)
        assets_dir = f"{base}_assets"
        try:
            os.makedirs(assets_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка",
                f"Не удалось создать папку для картинок:\n{e}")
            return

        self._audit_log(f"Сохранение картинок в {assets_dir}",
                        tag="REPORT")
        images = self._save_figures_for_report(assets_dir)
        self._audit_log(
            f"Файлы для отчёта: {sorted(images.keys())}",
            tag="REPORT")

        data = self._collect_report_data()
        data['images'] = images

        sections = {
            'quality': self.chk_report_include_quality.isChecked(),
            'dup': self.chk_report_include_dup.isChecked(),
            'heatmap': self.chk_report_include_heatmap.isChecked(),
            'audit': self.chk_report_include_audit.isChecked(),
            'validation': self.chk_report_include_validation.isChecked(),
            'cluster': self.chk_report_include_cluster.isChecked(),
            'training_results':
                self.chk_report_include_training_results.isChecked(),
        }

        self.btn_generate_report.setEnabled(False)
        self.btn_generate_report.setText("Генерация…")

        from utils.dataset_audit_logic import ReportThread
        self._report_thread = ReportThread(data, save_path, sections)
        self._report_thread.finished_signal.connect(
            self._on_report_finished)
        self._report_thread.start()

    def _on_report_finished(self, result):
        self.btn_generate_report.setEnabled(True)
        self.btn_generate_report.setText(
            "💾 Сгенерировать .md отчёт")
        err = result.get('error')
        if err:
            self._audit_log(f"Ошибка отчёта: {err}",
                            tag="REPORT", level='ERROR')
            QMessageBox.critical(self, "Ошибка", err)
            return
        path = result['path']
        self._last_report_dir = os.path.dirname(path)
        self.btn_open_report_folder.setEnabled(True)
        self._audit_log(f"Отчёт сохранён: {path}",
                        tag="REPORT", level='OK')
        QMessageBox.information(
            self, "Готово", f"Отчёт сохранён:\n{path}")

    def _collect_report_data(self):
        data = {}

        general = {
            'Pairs': len(self.pairs),
            'Filtered': len(self.filtered_pairs),
            'Dataset type': self.dataset_type,
            'Input format': self.input_format,
            'Classes in table': self.class_table.rowCount(),
            'Device': self._device,
        }
        if self.images_folder:
            general['Images'] = self.images_folder
        if self.labels_folder:
            general['Labels'] = self.labels_folder
        if self.masks_folder:
            general['Masks'] = self.masks_folder
        data['general'] = general

        try:
            all_ids = sorted(
                DatasetAnalyzer.get_class_names_by_orig_id(
                    self.class_table).keys())
            included_ids = sorted(
                DatasetAnalyzer.get_included_class_ids(self.class_table))
        except Exception:
            all_ids, included_ids = [], []
        excluded_ids = sorted(set(all_ids) - set(included_ids))

        data['filter_params'] = {
            'min_w': self.filter_min_w.value(),
            'min_h': self.filter_min_h.value(),
            'min_objs': self.filter_min_objs.value(),
            'max_objs': self.filter_max_objs.value(),
            'min_area': self.filter_min_area.value(),
            'max_area': self.filter_max_area.value(),
            'include_empty': self.include_empty_annotations.isChecked(),
            'exclude_classes_from_files':
                self.chk_exclude_classes_from_files.isChecked(),
            'all_class_ids': all_ids,
            'included_class_ids': included_ids,
            'excluded_class_ids': excluded_ids,
        }

        cs = []
        for r in range(self.class_stats_table.rowCount()):
            row = []
            for c in range(3):
                it = self.class_stats_table.item(r, c)
                row.append(it.text() if it else "")
            cs.append(row)
        data['class_stats'] = cs

        data['imbalance'] = (dict(self._last_imbalance)
                             if self._last_imbalance else None)

        if self._last_quality:
            res = self._last_quality.get('resolution')
            br = self._last_quality.get('brightness')
            data['quality'] = {
                'median_w': res['median_w'] if res else 0,
                'median_h': res['median_h'] if res else 0,
                'recommended_imgsz':
                    res['recommended_imgsz'] if res else 0,
                'brightness': br,
            }
        else:
            data['quality'] = None

        if self._last_dup:
            d = self._last_dup
            data['dup'] = {
                'exact_count': len(d['exact']),
                'exact_images': sum(len(g) for g in d['exact']),
                'near_count': len(d['near']),
                'near_images': sum(len(g) for g in d['near']),
                'leaked_count': d['leak']['leaked_count'],
                'leaked_groups': [
                    f"{rg['splits']}: " + ", ".join(
                        os.path.basename(p) for p in rg['paths'][:5])
                    for rg in d['leak']['groups'] if rg['leak']
                ],
            }
        else:
            data['dup'] = None

        if self._last_heatmap:
            hm = self._last_heatmap
            h = hm['heatmap']
            g = hm['grid_size']
            top = np.argsort(h.ravel())[::-1][:3]
            top_zones = []
            for idx in top:
                gy, gx = divmod(int(idx), g)
                top_zones.append(f"({gx},{gy}): {h[gy, gx]:.0f}")
            data['heatmap'] = {
                'total_objects': hm['total_objects'],
                'grid': g,
                'top_zones': top_zones,
            }
        else:
            data['heatmap'] = None

        data['audit_meta'] = (dict(self._last_audit_meta)
                              if self._last_audit_meta else None)
        data['audit_rows'] = [
            (os.path.basename(it['img_path']), it['gt_count'],
             it['pred_count'], it['mean_iou'])
            for it in self._last_audit_results[:20]
        ]

        data['validation'] = (dict(self._last_validation)
                              if self._last_validation else None)

        if self._last_cluster:
            c = self._last_cluster
            sizes = sorted(c['sizes'].items(),
                           key=lambda x: -x[1])[:15]
            data['cluster'] = {
                'count': c['count'],
                'n_clusters': c['n_clusters'],
                'noise': c['noise'],
                'sizes': sizes,
            }
        else:
            data['cluster'] = None

        data['training_results'] = (
            dict(self._last_training_results)
            if self._last_training_results else None)

        return data


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetAnalysisWindow()
    window.show()
    sys.exit(app.exec_())