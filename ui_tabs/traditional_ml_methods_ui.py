from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.morphology_ui import MorphologyWidget
from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout


def setup_traditional_ml_ui(parent):
    controls_layout = begin_layout(parent, 'traditional_ml')

    # Выбор модели
    model_layout = QHBoxLayout()
    model_layout.addWidget(QLabel("Model:"))
    parent.model_combo = QComboBox()
    models = ["K-Means Clustering", "MeanShift", "SVM (pixel-wise)", "Random Forest", "XGBoost", "Decision Tree"]
    parent.model_combo.addItems(models)
    model_layout.addWidget(parent.model_combo)
    controls_layout.addLayout(model_layout)

    # Параметры K-Means
    parent.kmeans_container = QWidget()
    kmeans_layout = QGridLayout(parent.kmeans_container)
    kmeans_layout.addWidget(QLabel("Number of clusters K:"), 0, 0)
    parent.kmeans_clusters = QSpinBox()
    parent.kmeans_clusters.setRange(2, 10)
    parent.kmeans_clusters.setValue(3)
    kmeans_layout.addWidget(parent.kmeans_clusters, 0, 1)
    controls_layout.addWidget(parent.kmeans_container)

    # Параметры SVM / Random Forest (общие)
    parent.svm_container = QWidget()
    svm_layout = QGridLayout(parent.svm_container)
    svm_layout.addWidget(QLabel("Kernel:"), 0, 0)
    parent.svm_kernel_combo = QComboBox()
    parent.svm_kernel_combo.addItems(["linear", "rbf", "poly"])
    svm_layout.addWidget(parent.svm_kernel_combo, 0, 1)
    svm_layout.addWidget(QLabel("C:"), 1, 0)
    parent.svm_c = QSpinBox()
    parent.svm_c.setRange(1, 100)
    parent.svm_c.setValue(10)
    svm_layout.addWidget(parent.svm_c, 1, 1)
    controls_layout.addWidget(parent.svm_container)

    # Параметры Random Forest
    parent.rf_container = QWidget()
    rf_layout = QGridLayout(parent.rf_container)
    rf_layout.addWidget(QLabel("Number of trees:"), 0, 0)
    parent.rf_trees = QSpinBox()
    parent.rf_trees.setRange(10, 200)
    parent.rf_trees.setValue(50)
    rf_layout.addWidget(parent.rf_trees, 0, 1)
    controls_layout.addWidget(parent.rf_container)

    # Параметры XGBoost
    parent.xgb_container = QWidget()
    xgb_layout = QGridLayout(parent.xgb_container)
    xgb_layout.addWidget(QLabel("Number of trees:"), 0, 0)
    parent.xgb_trees = QSpinBox()
    parent.xgb_trees.setRange(10, 200)
    parent.xgb_trees.setValue(50)
    xgb_layout.addWidget(parent.xgb_trees, 0, 1)
    controls_layout.addWidget(parent.xgb_container)

    # Скрыть контейнеры по умолчанию
    parent.kmeans_container.setVisible(True)
    parent.svm_container.setVisible(False)
    parent.rf_container.setVisible(False)
    parent.xgb_container.setVisible(False)

    # ---------- Кластеры для K-Means / MeanShift ----------
    parent.cluster_group = QGroupBox("Cluster selection (K‑Means / MeanShift)")
    cluster_layout = QVBoxLayout()
    parent.cluster_label = QLabel("Target cluster: (not trained yet)")
    parent.cluster_combo = QComboBox()
    parent.cluster_combo.setEnabled(False)

    # Горизонтальный ряд: комбобокс + кнопка поиска K
    cluster_combo_layout = QHBoxLayout()
    cluster_combo_layout.addWidget(parent.cluster_combo, 1)
    parent.find_best_k_btn = QPushButton("Find best K (elbow)")
    cluster_combo_layout.addWidget(parent.find_best_k_btn)

    cluster_layout.addWidget(parent.cluster_label)
    cluster_layout.addLayout(cluster_combo_layout)
    parent.cluster_group.setLayout(cluster_layout)
    controls_layout.addWidget(parent.cluster_group)

    # ---------- Выбор признаков (в одну строку) ----------
    features_group = QGroupBox("Feature extraction")
    features_layout = QHBoxLayout()
    parent.color_feature = QCheckBox("Intensity (gray)")
    parent.color_feature.setChecked(True)
    parent.texture_feature = QCheckBox("Texture (LBP)")
    parent.texture_feature.setChecked(False)
    parent.spatial_feature = QCheckBox("Spatial coordinates (x,y)")
    parent.spatial_feature.setChecked(False)
    features_layout.addWidget(parent.color_feature)
    features_layout.addWidget(parent.texture_feature)
    features_layout.addWidget(parent.spatial_feature)
    parent.normalize_features = QCheckBox("Normalize features (StandardScaler)")
    parent.normalize_features.setToolTip("Apply StandardScaler to features (recommended for SVM)")
    features_layout.addWidget(parent.normalize_features)
    features_layout.addStretch()
    features_group.setLayout(features_layout)
    controls_layout.addWidget(features_group)

    # ---------- Суперпиксели (в одну строку) ----------
    superpixel_group = QGroupBox("Superpixel processing")
    superpixel_layout = QHBoxLayout()
    parent.use_superpixels = QCheckBox("Use superpixels (SLIC)")
    parent.use_superpixels.setChecked(False)
    superpixel_layout.addWidget(parent.use_superpixels)
    superpixel_layout.addWidget(QLabel("Superpixel size:"))
    parent.superpixel_size = QSpinBox()
    parent.superpixel_size.setRange(10, 200)
    parent.superpixel_size.setValue(50)
    parent.superpixel_size.setEnabled(False)
    superpixel_layout.addWidget(parent.superpixel_size)
    superpixel_layout.addStretch()
    superpixel_group.setLayout(superpixel_layout)
    controls_layout.addWidget(superpixel_group)

    # Кнопки обучения, загрузки, сохранения
    train_layout = QHBoxLayout()
    parent.train_button = QPushButton("Train Model")
    parent.load_model_button = QPushButton("Load Model")
    parent.save_model_btn = QPushButton("Save Model")
    train_layout.addWidget(parent.train_button)
    train_layout.addWidget(parent.load_model_button)
    train_layout.addWidget(parent.save_model_btn)
    controls_layout.addLayout(train_layout)

    # ----- Отображение пути к модели -----
    model_path_layout = QHBoxLayout()
    model_path_layout.addWidget(QLabel("Model file:"))
    parent.model_path_edit = QLineEdit()
    parent.model_path_edit.setReadOnly(True)
    parent.model_path_edit.setPlaceholderText("No model loaded")
    model_path_layout.addWidget(parent.model_path_edit, 1)
    controls_layout.addLayout(model_path_layout)

    # Индикатор соответствия настроек и кнопка сброса
    model_status_layout = QHBoxLayout()
    parent.reset_to_model_btn = QPushButton("↻ Сбросить к настройкам модели")
    parent.reset_to_model_btn.setEnabled(False)
    model_status_layout.addWidget(parent.reset_to_model_btn)
    parent.model_mismatch_label = QLabel("⚙️ Модель не обучена")
    parent.model_mismatch_label.setStyleSheet("color: gray;")
    model_status_layout.addWidget(parent.model_mismatch_label)
    model_status_layout.addStretch()
    controls_layout.addLayout(model_status_layout)

    parent.apply_button = QPushButton("Segment Current Image")
    controls_layout.addWidget(parent.apply_button)

    # Invert mask
    parent.invert_checkbox = QCheckBox("Invert mask")
    controls_layout.addWidget(parent.invert_checkbox)

    # Морфологические операции
    morph_label = QLabel("Morphological operations (applied after segmentation):")
    controls_layout.addWidget(morph_label)
    parent.morphology_widget = MorphologyWidget(contour_available=False)
    controls_layout.addWidget(parent.morphology_widget)

    # Режим отрисовки объектов
    controls_layout.addWidget(ResultPanel(parent))

    finish_layout(parent)
