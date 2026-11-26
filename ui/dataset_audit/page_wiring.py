"""Dataset analysis UI page construction."""

from ui.log_widget import LogWidget


def connect_source(parent):
    # Лог
    # ============================================================
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.main_v_splitter.addWidget(parent.log_widget)
    parent.main_v_splitter.setSizes([600, 150])
    parent.log_widget.setVisible(False)

    # ============================================================
    # Подключения верхней панели
    # ============================================================
    parent.btn_folder.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_load_yaml.clicked.connect(parent.load_from_yaml)
    parent.btn_load_coco.clicked.connect(parent.load_from_coco)
    parent.btn_add_coco.clicked.connect(parent.add_from_coco)
    parent.btn_add_yaml.clicked.connect(parent.add_from_yaml)
    parent.btn_load_unet.clicked.connect(parent.load_from_unet_yaml)
