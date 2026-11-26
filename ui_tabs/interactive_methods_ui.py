from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ui.morphology_ui import MorphologyWidget
from ui.segmentation.prompt_tools import PromptTools

# interactive_methods_ui.py
from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout


def setup_interactive_ui(parent):
    """Создаёт UI для вкладки интерактивных методов сегментации."""
    controls_layout = begin_layout(parent, 'interactive')

    # Выбор метода
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Method:"))
    parent.method_combo = QComboBox()
    methods = ["GrabCut", "Lazy Snapping", "SuperCut", "OneCut", "Random Walker", "Watershed (marker)", "Active Contours", "Morphological Chan–Vese"]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)
    parent.method_help = QLabel()
    parent.method_help.setWordWrap(True)
    descriptions = {
        "GrabCut": "ROI должен охватывать объект и оставлять фон снаружи. Both учитывает ROI и жёсткие штрихи; без ROI нужны FG и BG. Итерации переоценивают цветовые модели только текущего снимка.",
        "Lazy Snapping": "Вариант проекта: SLIC + цветовые GMM по FG/BG + разрез графа. Если противоположные метки попали в один суперпиксель, уменьшите Superpixel size. Это не точная реализация одноимённой статьи.",
        "SuperCut": "Вариант проекта: разрез SLIC-графа с априорной вероятностью из ROI. Lambda усиливает влияние ROI; color sigma ослабляет влияние различий цвета. Это не точная реализация одноимённой статьи.",
        "OneCut": "Вариант проекта: цветовые GMM из ROI/фона и разрез SLIC-графа. ROI должен содержать преимущественно объект. Data weight — цветовые модели; spatial weight — согласованность соседей. Это не точная реализация одноимённой статьи.",
        "Random Walker": "Нанесите FG/BG в разных областях. Beta повышает чувствительность к границам; RGB учитывает цвет. cg_j — основной решатель; bf ограничен малыми снимками, cg_mg требует pyamg.",
        "Watershed (marker)": "scribbles: заполнение от ручных FG/BG по градиенту. automatic: тёмные объекты на светлом фоне, Otsu + distance transform. Min distance — расстояние между пиками в рабочих пикселях, не площадь маркера.",
        "Active Contours": "ROI задаёт начальную замкнутую кривую возле границы. Alpha стягивает контур, beta подавляет изгибы, gamma задаёт численный шаг. Метод зависит от начального контура и границ изображения.",
        "Morphological Chan–Vese": "ROI задаёт начальную область. Метод ищет однородные внутреннюю/внешнюю области и подходит для слабых границ. Smoothing сглаживает контур; Lambda1/Lambda2 — параметры областей skimage. Текстурные и неоднородные объекты могут потребовать другой метод.",
    }
    def update_help():
        parent.method_help.setText(descriptions[parent.method_combo.currentText()])
    parent.method_combo.currentIndexChanged.connect(update_help)
    update_help()
    controls_layout.addWidget(parent.method_help)

    # Группа для отображения требований
    parent.requirements_frame = QFrame()
    parent.requirements_frame.setFrameShape(QFrame.StyledPanel)
    req_layout = QHBoxLayout(parent.requirements_frame)
    parent.requirements_label = QLabel("Требования:")
    parent.requirements_label.setStyleSheet("font-weight: bold;")
    req_layout.addWidget(parent.requirements_label)
    parent.requirements_text = QLabel("")
    parent.requirements_text.setWordWrap(True)
    req_layout.addWidget(parent.requirements_text)
    controls_layout.addWidget(parent.requirements_frame)

    # Параметры методов
    from ui.interactive_params_ui import InteractiveParametersUI
    parent.params_widget = InteractiveParametersUI(parent)
    controls_layout.addWidget(parent.params_widget)

    # ---------- Инструменты интерактивного ввода ----------
    input_group = QGroupBox("Подсказки сегментации")
    input_layout = QVBoxLayout()

    parent.prompt_tools = PromptTools()
    parent.prompt_tools.full_image.setToolTip(
        'Рамка по умолчанию — весь снимок. GrabCut, SuperCut, OneCut и Chan–Vese '
        'используют край как начальный фон. Если объект касается края, уточните рамку '
        'или используйте метод с метками объекта и фона.')
    parent.rect_mode_btn = parent.prompt_tools.box
    parent.fg_scribble_btn = parent.prompt_tools.positive
    parent.bg_scribble_btn = parent.prompt_tools.negative
    parent.merge_objects_checkbox = parent.prompt_tools.merge
    parent.reset_mask_btn = parent.prompt_tools.clear
    input_layout.addWidget(parent.prompt_tools)

    row2 = QHBoxLayout()
    parent.run_seg_btn = QPushButton("Run Segmentation")
    row2.addWidget(parent.run_seg_btn)
    input_layout.addLayout(row2)

    input_group.setLayout(input_layout)
    controls_layout.addWidget(input_group)

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
