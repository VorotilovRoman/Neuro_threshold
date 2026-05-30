# interactive_params_ui.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QSlider, QGroupBox,
                             QComboBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt


class InteractiveParametersUI(QWidget):
    """Виджет для управления параметрами всех методов сегментации"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === GrabCut параметры ===
        self.grabcut_group = QGroupBox("GrabCut Parameters")
        gc_layout = QGridLayout()

        gc_layout.addWidget(QLabel("Iterations:"), 0, 0)
        self.grabcut_iters = QSpinBox()
        self.grabcut_iters.setRange(1, 20)
        self.grabcut_iters.setValue(5)
        gc_layout.addWidget(self.grabcut_iters, 0, 1)

        gc_layout.addWidget(QLabel("Mode:"), 1, 0)
        self.grabcut_mode = QComboBox()
        self.grabcut_mode.addItems(["GC_INIT_WITH_RECT", "GC_INIT_WITH_MASK", "Both"])
        self.grabcut_mode.setCurrentText("Both")
        gc_layout.addWidget(self.grabcut_mode, 1, 1)

        self.grabcut_group.setLayout(gc_layout)
        layout.addWidget(self.grabcut_group)

        # === Lazy Snapping параметры ===
        self.lazy_group = QGroupBox("Lazy Snapping / SuperCut / OneCut Parameters")
        lazy_layout = QGridLayout()

        lazy_layout.addWidget(QLabel("Superpixel size:"), 0, 0)
        self.lazy_superpixel = QSpinBox()
        self.lazy_superpixel.setRange(10, 200)
        self.lazy_superpixel.setValue(50)
        lazy_layout.addWidget(self.lazy_superpixel, 0, 1)

        lazy_layout.addWidget(QLabel("Compactness:"), 1, 0)
        self.lazy_compactness = QDoubleSpinBox()
        self.lazy_compactness.setRange(1, 50)
        self.lazy_compactness.setValue(10)
        self.lazy_compactness.setSingleStep(1)
        lazy_layout.addWidget(self.lazy_compactness, 1, 1)

        lazy_layout.addWidget(QLabel("Sigma (smoothing):"), 2, 0)
        self.lazy_sigma = QDoubleSpinBox()
        self.lazy_sigma.setRange(0, 5)
        self.lazy_sigma.setValue(1)
        self.lazy_sigma.setSingleStep(0.1)
        lazy_layout.addWidget(self.lazy_sigma, 2, 1)

        lazy_layout.addWidget(QLabel("Color sigma:"), 3, 0)
        self.lazy_color_sigma = QDoubleSpinBox()
        self.lazy_color_sigma.setRange(0.1, 50)
        self.lazy_color_sigma.setValue(10)
        lazy_layout.addWidget(self.lazy_color_sigma, 3, 1)

        self.lazy_group.setLayout(lazy_layout)
        layout.addWidget(self.lazy_group)

        # === SuperCut специфичные параметры ===
        self.supercut_group = QGroupBox("SuperCut Specific")
        sc_layout = QGridLayout()

        sc_layout.addWidget(QLabel("Lambda (unary weight):"), 0, 0)
        self.supercut_lambda = QDoubleSpinBox()
        self.supercut_lambda.setRange(0, 10)
        self.supercut_lambda.setValue(0.5)
        self.supercut_lambda.setSingleStep(0.1)
        sc_layout.addWidget(self.supercut_lambda, 0, 1)

        sc_layout.addWidget(QLabel("Sigma (color distance):"), 1, 0)
        self.supercut_sigma = QDoubleSpinBox()
        self.supercut_sigma.setRange(0, 100)
        self.supercut_sigma.setValue(10)
        sc_layout.addWidget(self.supercut_sigma, 1, 1)

        self.supercut_group.setLayout(sc_layout)
        layout.addWidget(self.supercut_group)

        # === OneCut специфичные параметры ===
        self.onecut_group = QGroupBox("OneCut Specific")
        oc_layout = QGridLayout()

        oc_layout.addWidget(QLabel("Spatial weight:"), 0, 0)
        self.onecut_spatial = QDoubleSpinBox()
        self.onecut_spatial.setRange(0, 10)
        self.onecut_spatial.setValue(1)
        oc_layout.addWidget(self.onecut_spatial, 0, 1)

        oc_layout.addWidget(QLabel("Data term weight:"), 1, 0)
        self.onecut_data = QDoubleSpinBox()
        self.onecut_data.setRange(0, 10)
        self.onecut_data.setValue(1)
        oc_layout.addWidget(self.onecut_data, 1, 1)

        self.onecut_group.setLayout(oc_layout)
        layout.addWidget(self.onecut_group)

        # === Watershed параметры ===
        self.watershed_group = QGroupBox("Watershed Parameters")
        ws_layout = QGridLayout()

        ws_layout.addWidget(QLabel("Distance threshold (%):"), 0, 0)
        self.watershed_thresh = QSlider(Qt.Horizontal)
        self.watershed_thresh.setRange(1, 100)
        self.watershed_thresh.setValue(50)
        self.watershed_thresh.valueChanged.connect(
            lambda v: self.watershed_thresh_label.setText(f"{v}%"))
        ws_layout.addWidget(self.watershed_thresh, 0, 1)
        self.watershed_thresh_label = QLabel("50%")
        ws_layout.addWidget(self.watershed_thresh_label, 0, 2)

        ws_layout.addWidget(QLabel("Min distance:"), 1, 0)
        self.watershed_min_dist = QSpinBox()
        self.watershed_min_dist.setRange(1, 100)
        self.watershed_min_dist.setValue(20)
        ws_layout.addWidget(self.watershed_min_dist, 1, 1)

        self.watershed_group.setLayout(ws_layout)
        layout.addWidget(self.watershed_group)

        # === Random Walker параметры ===
        self.rw_group = QGroupBox("Random Walker Parameters")
        rw_layout = QGridLayout()

        rw_layout.addWidget(QLabel("Beta (edge weight):"), 0, 0)
        self.rw_beta = QDoubleSpinBox()
        self.rw_beta.setRange(0, 1000)
        self.rw_beta.setValue(130)
        self.rw_beta.setSingleStep(10)
        rw_layout.addWidget(self.rw_beta, 0, 1)

        rw_layout.addWidget(QLabel("Mode:"), 1, 0)
        self.rw_mode = QComboBox()
        self.rw_mode.addItems(["cg_j", "cg_mg", "bf"])
        rw_layout.addWidget(self.rw_mode, 1, 1)

        self.rw_group.setLayout(rw_layout)
        layout.addWidget(self.rw_group)

        # === Active Contours параметры ===
        self.ac_group = QGroupBox("Active Contours Parameters")
        ac_layout = QGridLayout()

        ac_layout.addWidget(QLabel("Alpha (elasticity):"), 0, 0)
        self.ac_alpha = QDoubleSpinBox()
        self.ac_alpha.setRange(0, 1)
        self.ac_alpha.setValue(0.01)
        self.ac_alpha.setSingleStep(0.005)
        self.ac_alpha.setDecimals(3)
        ac_layout.addWidget(self.ac_alpha, 0, 1)

        ac_layout.addWidget(QLabel("Beta (rigidity):"), 1, 0)
        self.ac_beta = QDoubleSpinBox()
        self.ac_beta.setRange(0, 1)
        self.ac_beta.setValue(0.1)
        self.ac_beta.setSingleStep(0.05)
        ac_layout.addWidget(self.ac_beta, 1, 1)

        ac_layout.addWidget(QLabel("Gamma (step size):"), 2, 0)
        self.ac_gamma = QDoubleSpinBox()
        self.ac_gamma.setRange(0, 1)
        self.ac_gamma.setValue(0.01)
        self.ac_gamma.setSingleStep(0.005)
        ac_layout.addWidget(self.ac_gamma, 2, 1)

        ac_layout.addWidget(QLabel("Max iterations:"), 3, 0)
        self.ac_max_iter = QSpinBox()
        self.ac_max_iter.setRange(50, 1000)
        self.ac_max_iter.setValue(250)
        ac_layout.addWidget(self.ac_max_iter, 3, 1)

        ac_layout.addWidget(QLabel("Convergence threshold:"), 4, 0)
        self.ac_convergence = QDoubleSpinBox()
        self.ac_convergence.setRange(0.0001, 0.1)
        self.ac_convergence.setValue(0.001)
        self.ac_convergence.setDecimals(4)
        ac_layout.addWidget(self.ac_convergence, 4, 1)

        ac_layout.addWidget(QLabel("Initial contour type:"), 5, 0)
        self.ac_init_type = QComboBox()
        self.ac_init_type.addItems(["Rectangle", "Circle", "Ellipse"])
        ac_layout.addWidget(self.ac_init_type, 5, 1)

        self.ac_smooth_check = QCheckBox("Apply Gaussian smoothing")
        self.ac_smooth_check.setChecked(True)
        ac_layout.addWidget(self.ac_smooth_check, 6, 0, 1, 2)

        self.ac_group.setLayout(ac_layout)
        layout.addWidget(self.ac_group)

        # Изначально скрываем специфичные группы
        self.supercut_group.setVisible(False)
        self.onecut_group.setVisible(False)

        layout.addStretch()

    def get_grabcut_params(self):
        return {
            'iterations': self.grabcut_iters.value(),
            'mode': self.grabcut_mode.currentText()
        }

    def get_superpixel_params(self):
        return {
            'superpixel_size': self.lazy_superpixel.value(),
            'compactness': self.lazy_compactness.value(),
            'sigma': self.lazy_sigma.value(),
            'color_sigma': self.lazy_color_sigma.value()   # новый параметр
        }

    def get_supercut_params(self):
        params = self.get_superpixel_params()
        params.update({
            'lambda_val': self.supercut_lambda.value(),
            'sigma_color': self.supercut_sigma.value()
        })
        return params

    def get_onecut_params(self):
        params = self.get_superpixel_params()
        params.update({
            'spatial_weight': self.onecut_spatial.value(),
            'data_weight': self.onecut_data.value()
        })
        return params

    def get_watershed_params(self):
        return {
            'threshold': self.watershed_thresh.value(),
            'min_distance': self.watershed_min_dist.value()
        }

    def get_random_walker_params(self):
        return {
            'beta': self.rw_beta.value(),
            'mode': self.rw_mode.currentText()
        }

    def get_active_contour_params(self):
        return {
            'alpha': self.ac_alpha.value(),
            'beta': self.ac_beta.value(),
            'gamma': self.ac_gamma.value(),
            'max_iter': self.ac_max_iter.value(),
            'convergence': self.ac_convergence.value(),
            'init_type': self.ac_init_type.currentText(),
            'smooth': self.ac_smooth_check.isChecked()
        }