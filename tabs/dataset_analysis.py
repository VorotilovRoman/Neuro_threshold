# Direct and module execution share the same project imports.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

from runtime_torch import prepare_windows_torch

prepare_windows_torch()
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
from utils_new.dataset_audit.analysis import DatasetAnalyzer
from ui.dataset_audit.lifecycle import close_workers
from utils_new.dataset_validation_new import ImageStats, PairValidator

from ui_tabs.dataset_analysis_ui import setup_dataset_analysis_ui
from utils_new.dataset_loader_new import DatasetLoaderMixin
from ui.dataset_audit.tasks import ScanController
from ui.dataset_audit.quality_controller import DatasetQualityController
from ui.dataset_audit.session import DatasetSessionController
from ui.dataset_audit.selection import DatasetSelectionController
from ui.dataset_audit.duplicates import DatasetDuplicatesController
from ui.dataset_audit.spatial import DatasetSpatialController
from ui.dataset_audit.models import DatasetModelController
from ui.dataset_audit.embeddings import DatasetEmbeddingController
from ui.dataset_audit.training_results import DatasetTrainingResultsController
from ui.dataset_audit.reporting import DatasetReportController


class DatasetAnalysisWindow(
    DatasetSessionController,
    DatasetSelectionController,
    DatasetDuplicatesController,
    DatasetSpatialController,
    DatasetModelController,
    DatasetEmbeddingController,
    DatasetTrainingResultsController,
    DatasetReportController,
    DatasetQualityController,
    DatasetLoaderMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self._scan_busy = False
        self._closing = False
        self._dataset_revision = 0
        self._filters_applied = False
        self._analysis_splits = {}
        self.setWindowTitle("Анализ и фильтрация датасета")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        setup_dataset_analysis_ui(self)

        self._device = None  # Resolve Torch/CUDA only for an explicit model operation.
        self.scans = ScanController(self)
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
        self.input_format = "auto"
        self.mask_pixel_offset_in = 1
        self.validator = PairValidator(dataset_type=0, input_format="auto")
        self.analyzer = DatasetAnalyzer(dataset_type=0, input_format="auto")
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
        self._last_audit_all_results = []
        self._last_audit_display = []
        self._last_audit_meta = None
        self._last_cluster = None
        self._last_cluster_data = None
        self._last_validation = None
        self._last_training_results = None
        self._last_export_dir = None
        self._last_report_dir = None
        self._last_filter_stats = None
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
            self._on_dataset_type_changed
        )
        self.input_format_combo.currentIndexChanged.connect(
            self._on_input_format_changed
        )
        self.class_table.itemChanged.connect(self.on_class_table_item_changed)
        self.include_empty_annotations.stateChanged.connect(
            self._on_include_empty_toggled
        )
        self.file_list.itemDoubleClicked.connect(self._on_file_list_double_clicked)

        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(
            self._on_file_list_context_menu
        )

        self._copy_shortcut = QShortcut(QKeySequence.Copy, self.file_list)
        self._copy_shortcut.setContext(Qt.WidgetShortcut)
        self._copy_shortcut.activated.connect(self._copy_selected_rows_to_clipboard)

        self.btn_apply_filters.clicked.connect(self.apply_filters_and_update_metrics)
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

        self.btn_browse_training_results.clicked.connect(self.browse_training_results)
        self.btn_load_training_results.clicked.connect(self.load_training_results)

        self.btn_open_last_export.clicked.connect(self.open_last_export_folder)
        self.btn_open_report_folder.clicked.connect(self.open_report_folder)

        self.audit_table.cellDoubleClicked.connect(self._open_audit_row)
        self.audit_filter_combo.currentIndexChanged.connect(self._rebuild_audit_table)
        self.dup_table.itemSelectionChanged.connect(self._on_dup_row_selected)
        self.dup_table.cellDoubleClicked.connect(self._on_dup_row_double_clicked)

        self.cluster_canvas.mpl_connect("button_press_event", self._on_cluster_click)

        self._sync_all_states()
        self._sync_analyzer()
        self._audit_log("Вкладка анализа датасета инициализирована.", tag="INIT")

        self.refresh_workspace_summary()

    def _start_audit_worker(self, worker):
        worker.revision = self._dataset_revision
        worker.start()

    def shutdown_audit(self):
        close_workers(self)
        self.scans.invalidate()
        self.shutdown_annotation_review()

    def closeEvent(self, event):
        self.shutdown_audit()
        if self._scan_busy:
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetAnalysisWindow()
    window.show()
    from ui.application import exec_window

    sys.exit(exec_window(window, app))
