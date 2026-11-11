"""Common layouts for method tabs. Algorithm-specific controls remain in ui_tabs."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.annotation_review_view import DatasetReviewGraphicsView
from ui.image_navigation import ImageNavigationWidget
from ui.log_widget import LogWidget
from ui.segmentation.histogram import HistogramCanvas
from ui.segmentation.interactive_view import SegmentationInputView
from utils_new.annotations.modes import MODE_LABELS

STAGES = {
    'threshold': [('original_view','Исходное'),('binary_view','Бинаризация'),('morph_view','Морфология'),('annotated_view','Разметка')],
    'gradient': [('original_view','Исходное'),('gradient_view','Градиент'),('binary_view','Бинаризация'),('morph_view','Морфология'),('filled_view','Заполнение'),('annotated_view','Разметка')],
    'interactive': [('original_view','Ввод ROI / FG / BG'),('segmentation_view','Маска'),('morph_view','Морфология'),('annotated_view','Разметка')],
    'traditional_ml': [('original_view','Исходное'),('ml_view','Маска'),('morph_view','Морфология'),('annotated_view','Разметка')],
    'deep_learning': [('original_view','Исходное'),('dl_view','Маска'),('morph_view','Морфология'),('annotated_view','Разметка')],
}


class PresetPanel(QWidget):
    def __init__(self, owner):
        super().__init__()
        row=QHBoxLayout(self);row.setContentsMargins(0,0,0,0)
        row.addWidget(QLabel('Пресеты:'))
        owner.preset_combo=QComboBox();owner.preset_combo.setMinimumWidth(150)
        owner.save_preset_btn=QPushButton('Сохранить как пресет')
        owner.delete_preset_btn=QPushButton('Удалить пресет')
        for widget in (owner.preset_combo,owner.save_preset_btn,owner.delete_preset_btn):row.addWidget(widget)


class ResultPanel(QWidget):
    def __init__(self, owner):
        super().__init__()
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout();row.addWidget(QLabel('Режим разметки:'))
        owner.draw_combo=QComboBox()
        for mode,label in MODE_LABELS.items():owner.draw_combo.addItem(label,mode)
        row.addWidget(owner.draw_combo);layout.addLayout(row)
        owner.hull_checkbox=QCheckBox('Выпуклая оболочка (OBB / Segment)')
        owner.hull_checkbox.setVisible(False);layout.addWidget(owner.hull_checkbox)
        layout.addWidget(QLabel('Объекты — галочки задают сохранение и видимость:'))
        owner.object_list=QListWidget();owner.object_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(owner.object_list)
        layout.addWidget(QLabel('Координаты отмеченных объектов:'))
        owner.coord_text=QTextEdit();owner.coord_text.setReadOnly(True);owner.coord_text.setMaximumHeight(150)
        layout.addWidget(owner.coord_text)
        owner.save_button=QPushButton('Сохранить разметку');layout.addWidget(owner.save_button)
        owner.result_status_label=QLabel('Маска не вычислена');owner.result_status_label.setWordWrap(True)
        layout.addWidget(owner.result_status_label)


def begin_layout(owner,kind):
    owner.segmentation_kind=kind
    central=QWidget();owner.setCentralWidget(central)
    root=QVBoxLayout(central);root.setContentsMargins(0,0,0,0)
    row=QHBoxLayout();owner.nav_widget=ImageNavigationWidget();row.addWidget(owner.nav_widget)
    owner.reset_zoom_button=QPushButton('Сбросить масштаб');row.addWidget(owner.reset_zoom_button);root.addLayout(row)
    if kind == 'interactive':
        owner.pan_zoom_btn=QPushButton('Pan/Zoom');owner.pan_zoom_btn.setCheckable(True);row.addWidget(owner.pan_zoom_btn)
    splitter=QSplitter(Qt.Horizontal);root.addWidget(splitter,1)
    left=QWidget();left_layout=QVBoxLayout(left);left_layout.setContentsMargins(0,0,0,0)
    grid=QGridLayout();columns=3 if kind=='gradient' else 2
    for index,(name,title) in enumerate(STAGES[kind]):
        group=QGroupBox(title);group_layout=QVBoxLayout(group)
        view=SegmentationInputView() if kind=='interactive' and name=='original_view' else DatasetReviewGraphicsView()
        view.setMinimumSize(280,220);view.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        if name=='annotated_view':view.set_display_style(label_mode='all')
        setattr(owner,name,view);group_layout.addWidget(view);grid.addWidget(group,index//columns,index%columns)
    for index in range(columns):grid.setColumnStretch(index,1)
    for index in range(2):grid.setRowStretch(index,1)
    left_layout.addLayout(grid);owner.info_label=QLabel('Нет изображения');left_layout.addWidget(owner.info_label)
    splitter.addWidget(left)
    scroll=QScrollArea();scroll.setWidgetResizable(True)
    owner.controls_panel=QGroupBox('Параметры метода');controls=QVBoxLayout(owner.controls_panel)
    scroll.setWidget(owner.controls_panel);splitter.addWidget(scroll);splitter.setSizes([850,420])
    controls.addWidget(PresetPanel(owner))
    owner._method_root_layout=root
    return controls


def finish_layout(owner):
    from ui.segmentation import results
    owner.object_list.currentRowChanged.connect(lambda row: results.focus_row(owner,row))
    owner.annotated_view.set_callbacks(on_selection_changed=lambda index: results.focus_annotation(owner,index))
    root=owner._method_root_layout
    owner.hist_canvas=HistogramCanvas();owner.hist_ax=owner.hist_canvas.axes
    owner.hist_container=QWidget();layout=QVBoxLayout(owner.hist_container)
    layout.setContentsMargins(0,0,0,0);layout.addWidget(owner.hist_canvas)
    owner.hist_container.hide();root.addWidget(owner.hist_container)
    row=QHBoxLayout()
    owner.toggle_log_btn=QPushButton('Показать лог');owner.toggle_hist_btn=QPushButton('Показать гистограмму')
    for button in (owner.toggle_log_btn,owner.toggle_hist_btn):button.setCheckable(True);row.addWidget(button)
    root.addLayout(row)
    owner.log_widget=LogWidget(show_clear_btn=True,show_progress=False);owner.log_text=owner.log_widget.text
    owner.log_widget.hide();root.addWidget(owner.log_widget)
