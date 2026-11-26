"""Host-independent background loading, prompting, cancellation and preview."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

from pathlib import Path
from math import isfinite
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
from ui import dataset_source as source_dialogs
from ui.segmentation.jobs import BackgroundTask
from utils_new.import_export.sam_weights import MODELS
from utils_new.import_export.training import read_model_meta
from utils_new.segmentation.backends.sam import MODELS_DIR, SamPrompt, find_sam3_tokenizer, load_sam_backend


class SamController:
    def __init__(self, host):
        self.host = host
        self.owner, self.panel, self.view = host.owner, host.panel, host.view
        self.backend = self.job = None
        self.loading = False
        self.closed = False
        self.revision = 0
        self.frame = None
        self.preview_key = None
        self.preview_result = None
        panel = self.panel
        panel.prompt_tools.bind_view(self.view)
        panel.browse_weights.clicked.connect(self.browse_weights)
        panel.browse_tokenizer.clicked.connect(self.browse_tokenizer)
        panel.browse_dino.clicked.connect(self.browse_dino)
        panel.load.clicked.connect(self.load)
        panel.run.clicked.connect(self.run)
        panel.accept.clicked.connect(self.accept)
        panel.cancel.clicked.connect(self.cancel)
        panel.clear.clicked.connect(self.view.clear_prompts)
        panel.selected_box.clicked.connect(self.selected_box)
        panel.prompt_tools.tool_requested.connect(self.tool)
        panel.model.currentIndexChanged.connect(self.model_changed)
        panel.mode.currentIndexChanged.connect(self.mode_changed)
        for edit in (panel.weights, panel.tokenizer, panel.dino_directory):
            edit.textChanged.connect(self.model_changed)
        panel.device.currentIndexChanged.connect(self.model_changed)
        panel.dino_device.currentIndexChanged.connect(self.model_changed)
        for widget, signal in ((panel.text, "textChanged"), (panel.class_id, "valueChanged"),
                               (panel.kind, "currentIndexChanged"), (panel.area, "valueChanged"),
                               (panel.score, "valueChanged"), (panel.replace, "toggled"),
                               (panel.dino_box_score, "valueChanged"), (panel.dino_text_score, "valueChanged"),
                               (panel.dino_nms, "valueChanged"), (panel.dino_max_objects, "valueChanged")):
            getattr(widget, signal).connect(self.invalidate)
        self.view.prompts_changed.connect(self.invalidate)
        panel.merge.toggled.connect(self.merge_changed)
        host.bind(self)
        from ui.sam.downloads import SamDownloads
        self.downloads = SamDownloads(self)
        panel.settings_restored.connect(self.model_changed)
        self.update_load_state()

    def update_load_state(self):
        self.panel.set_model_loaded(not self.closed and self.backend is not None and
                                    self.settings_key() == getattr(self, 'loaded_key', None))

    def identity(self):
        return self.host.identity()

    def invalidate(self, *_, cancel_load=False):
        self.revision += 1
        self.preview_key = None
        self.preview_result = None
        self.view.prompt_preview = []
        self.view.viewport().update()
        self.panel.accept.setEnabled(False)
        if self.job is not None and (cancel_load or not self.loading):
            self.job.cancel()
        self.update_load_state()

    def frame_changed(self):
        if self.identity() != self.frame:
            self.frame = self.identity()
            self.stop_tool()
            self.view.clear_prompts()
            self.host.frame_changed()

    def stop_tool(self):
        self.view.set_prompt_tool(None)

    def tool(self, name):
        if self.job is not None or not self.host.image_ready():
            return
        previous = self.view.prompt_tool
        self.host.reset_tool()
        self.view.set_prompt_tool(None if previous == name else name)
        self.panel.status.setText({"positive": "Кликните внутри объекта; ПКМ добавляет фон.",
                                   "negative": "Кликните по области фона.",
                                   "box": "Протяните рамку вокруг объекта."}[name])

    def merge_changed(self):
        if self.job is not None and not self.loading:
            self.invalidate()
        if self.preview_result is not None:
            panel = self.panel
            annotations = self.preview_result.annotations(class_id=panel.class_id.value(), kind=panel.kind.currentData(),
                                                         min_area=panel.area.value(), merge_all=panel.merge.isChecked())
            self.view.prompt_preview = annotations
            self.view.viewport().update()
            panel.accept.setEnabled(self.host.can_accept(self.preview_result, annotations))
            panel.status.setText(f'Предпросмотр: {len(annotations)} объектов. Проверьте контур перед принятием.')
        self.host.output_changed()

    def provider_changed(self):
        self.invalidate(cancel_load=True)
        self.stop_tool()
        self.view.clear_prompts()

    def mode_changed(self):
        self.invalidate(cancel_load=True)
        self.stop_tool()
        self.view.clear_prompts()
        self.panel.capabilities()
        self.host.model_changed()
        self.panel.status.setText(f"Режим изменён. Нажмите «{self.panel.load.text()}».")

    def model_changed(self, *_):
        self.invalidate(cancel_load=True)
        self.panel.capabilities()
        self.host.model_changed()
        self.update_checkpoint_info()

    def update_checkpoint_info(self):
        panel = self.panel
        try:
            meta = read_model_meta(panel.weights.text().strip()) if panel.weights.text().strip() else {}
            text = ""
            if meta.get("kind") == "sam":
                key = meta.get("model_key")
                name = MODELS[key].name if key in MODELS else str(key)
                text = f"Дообученная {name}. "
                if meta.get("complete_epoch"):
                    text += f"Эпоха {meta.get('epoch')}. "
                    metrics = meta.get("metrics") or {}
                    if isinstance(metrics, dict):
                        for field, label in (("val_iou", "IoU"), ("val_dice", "Dice")):
                            value = metrics.get(field)
                            if type(value) in (int, float) and isfinite(value):
                                text += f"val {label}: {value:.4f}. "
                    text += "Оценка по подсказкам, не точность поиска объектов. "
                else:
                    text += "Неполная эпоха; валидационные метрики этих весов отсутствуют. "
                text += f"Рабочая сторона при обучении: {meta.get('max_side', '—')} px."
                if key != panel.model.currentData():
                    text += " Выберите соответствующую архитектуру перед загрузкой."
            panel.checkpoint_info.setText(text)
            panel.checkpoint_info.setVisible(bool(text))
        except (OSError, ValueError, TypeError) as error:
            panel.checkpoint_info.setText(f"Не удалось прочитать метаданные весов: {error}")
            panel.checkpoint_info.setVisible(True)

    def settings_key(self):
        panel = self.panel
        return (panel.model.currentData(), panel.weights.text().strip(),
                panel.device.currentText(), panel.tokenizer.text().strip(),
                panel.mode.currentData(),
                panel.dino_directory.text().strip() if panel.mode.currentData() == "grounded" else "",
                panel.dino_device.currentData() if panel.mode.currentData() == "grounded" else "cpu")

    def browse_weights(self):
        path, _ = source_dialogs.get_open_file(self.owner, "model_torch", directory=str(MODELS_DIR))
        if path:
            try:
                meta = read_model_meta(path)
            except (OSError, ValueError, TypeError) as error:
                QMessageBox.warning(self.owner, "Метаданные SAM", str(error))
                return
            key = meta.get("model_key") if meta.get("kind") == "sam" else None
            if key not in MODELS:
                key = next((key for key, model in MODELS.items()
                            if Path(path).name.casefold() == Path(model.assets[0].path).name.casefold()), None)
            if key in MODELS:
                self.panel.model.setCurrentIndex(self.panel.model.findData(key))
            self.panel.weights.setText(path)
            if self.panel.model.currentData() == "sam3":
                bpe = find_sam3_tokenizer(path)
                self.panel.tokenizer.setText(str(bpe) if bpe else "")

    def browse_tokenizer(self):
        path, _ = source_dialogs.get_open_file(self.owner, "sam_tokenizer", directory=str(MODELS_DIR))
        if path:
            self.panel.tokenizer.setText(path)

    def browse_dino(self):
        path = source_dialogs.get_directory(self.owner, "model_directory", directory=str(MODELS_DIR))
        if path:
            self.panel.dino_directory.setText(path)

    def selected_box(self):
        ann = self.host.selected_annotation()
        if ann is None:
            self.panel.status.setText("Сначала выделите объект в списке или на снимке.")
            return
        self.panel.class_id.setValue(ann.class_id)
        self.view.set_prompt_box(tuple(ann.points_px.min(axis=0)) + tuple(ann.points_px.max(axis=0)))

    def _start(self, compute, accept, status, *, loading=False, job=None):
        if self.closed or self.job is not None or not self.host.can_start():
            return
        self.stop_tool()
        downloading = job is not None and compute is None
        job = self.job = job or BackgroundTask(compute)
        self.loading = loading
        self.update_load_state()
        self.panel.controls.setEnabled(False)
        self.host.busy_changed(True)
        self.panel.progress.setRange(0, 0)
        self.panel.progress.setVisible(True)
        self.panel.cancel.setText("Отменить скачивание" if downloading else "Отменить вычисление")
        self.panel.cancel.setEnabled(True)
        self.panel.cancel.setVisible(True)
        self.panel.status.setText(status)

        def result(value):
            if not self.closed and self.job is job and not job.cancelled:
                try:
                    accept(value)
                except Exception as error:
                    failed(str(error))

        def failed(message):
            if not self.closed and self.job is job and not job.cancelled:
                self.panel.status.setText(message)
                self.owner.log(f"[SAM] {message}")
                QMessageBox.warning(self.owner, "SAM", message)

        def settled():
            if self.closed or self.job is not job:
                return
            self.job = None
            self.loading = False
            self.panel.controls.setEnabled(True)
            self.host.busy_changed(False)
            self.panel.capabilities()
            self.update_load_state()
            self.panel.progress.setVisible(False)
            self.panel.cancel.setEnabled(False)
            self.panel.cancel.setVisible(False)
            if job.cancelled:
                self.panel.status.setText("Операция отменена; результат не применён." +
                                          (" Готовые файлы сохранены." if downloading else ""))
            job.compute = None  # Release the worker closure's model/image references.
            job.deleteLater()

        job.result_ready.connect(result, Qt.QueuedConnection)
        job.failed.connect(failed, Qt.QueuedConnection)
        job.settled.connect(settled, Qt.QueuedConnection)
        job.start()

    def load(self):
        if self.job is not None or not self.host.can_start():
            return
        self.invalidate()
        key = self.settings_key()
        self.backend = None
        self.update_load_state()

        def accept(backend):
            if key == self.settings_key():
                self.backend = backend
                self.loaded_key = key
                self.update_load_state()
                self.panel.status.setText(f"Модель загружена: {key[0]} ({backend.device}). Укажите подсказки.")
                self.update_checkpoint_info()
                self.owner.log(f"[SAM] Локальные веса: {key[1]}; архитектура: {key[0]}; устройство: {backend.device}")
                if key[4] == "grounded":
                    detector_device = getattr(getattr(backend, "detector", None), "device", key[6])
                    self.owner.log(f"[Grounding DINO] Локальная папка: {key[5]}; устройство: {detector_device}")
                    self.panel.status.setText(f"SAM2: {backend.device}; Grounding DINO: {detector_device}. Введите текстовый запрос.")

        def compute(cancel):
            return load_sam_backend(key[0], key[1], device=key[2], tokenizer=key[3] or None,
                                    prompt_mode=key[4], detector_directory=key[5], detector_device=key[6], cancel=cancel)

        self._start(compute, accept, "Загрузка SAM2 и Grounding DINO…" if key[4] == "grounded"
                    else "Загрузка локальной модели SAM…", loading=True)

    def run(self):
        if self.job is not None or not self.host.can_start():
            return
        if self.backend is None or self.settings_key() != getattr(self, "loaded_key", None):
            self.panel.status.setText("Сначала загрузите модель с выбранными параметрами.")
            return
        if not self.host.image_ready():
            self.panel.status.setText("Дождитесь загрузки текущего снимка.")
            return
        image = self.host.image()
        panel = self.panel
        box = None if panel.mode.currentData() == 'text' and self.view.region_percent == 100 else self.view.effective_box()
        prompt = SamPrompt(tuple(self.view.prompt_points), tuple(self.view.prompt_labels), box,
                           panel.text.text() if panel.mode.currentData() in ("text", "grounded") else "")
        if panel.mode.currentData() in ("text", "grounded") and not prompt.text.strip():
            panel.status.setText("Введите текстовый запрос или выберите режим «Точки и рамка» и загрузите модель заново.")
            return
        try:
            prompt.validated(image.shape[1], image.shape[0], supports_text=(self.backend.model_key == "sam3" or getattr(self.backend, "supports_text", False)))
        except ValueError as error:
            panel.status.setText(str(error))
            return
        self.invalidate()
        key = (self.identity(), self.revision, id(image))
        backend = self.backend
        options = dict(class_id=panel.class_id.value(), kind=panel.kind.currentData(), min_area=panel.area.value(),
                       merge_all=panel.merge.isChecked())
        min_score = panel.score.value()
        detector_options = dict(box_threshold=panel.dino_box_score.value(), text_threshold=panel.dino_text_score.value(),
                                nms_threshold=panel.dino_nms.value(), max_objects=panel.dino_max_objects.value()) if panel.mode.currentData() == "grounded" else {}

        def compute(cancel):
            result = backend.predict(image, prompt, image_key=key[0], min_score=min_score, cancel=cancel, **detector_options)
            return result, result.annotations(**options)

        def accept(value):
            if key[:2] != (self.identity(), self.revision):
                return
            result, annotations = value
            scores, detections, phrases, detected_count = result.scores, result.detection_scores, result.phrases, result.detected_count
            self.preview_key = key
            self.preview_result = result
            self.view.prompt_preview = annotations
            self.view.viewport().update()
            panel.accept.setEnabled(self.host.can_accept(result, annotations))
            score_text = ", ".join(f"{score:.3f}" for score in scores[:10])
            panel.status.setText(f"Предпросмотр: {len(annotations)} объектов. Оценки: {score_text or '—'}. "
                                 "Проверьте контуры и нажмите «Принять результат».")
            if detections is not None:
                detection_text = ", ".join(f"{phrase}: {score:.3f}" for phrase, score in list(zip(phrases, detections))[:10])
                count = len(detections) if detected_count is None else detected_count
                panel.status.setText(f"Рамок DINO: {count}; масок SAM2: {len(scores)}; объектов: {len(annotations)}. "
                                     f"IoU SAM2: {score_text or '—'}. Grounding DINO: {detection_text or '—'}. "
                                     "Проверьте результат и пороги перед принятием.")

        self._start(compute, accept, "Поиск Grounding DINO и сегментация SAM2…" if detector_options
                    else "Вычисление маски SAM…")

    def accept(self):
        if (self.preview_key is None or self.preview_key[:2] != (self.identity(), self.revision)
                or not self.host.image_ready()):
            return
        image = self.host.image()
        if id(image) != self.preview_key[2]:
            self.invalidate()
            return
        annotations = list(self.view.prompt_preview)
        if self.host.confirm_replace and self.panel.replace.isChecked() and QMessageBox.question(
                self.owner, "SAM", "Заменить все существующие аннотации текущего снимка?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.host.apply_result(self.preview_result, annotations, self.panel.replace.isChecked())
        self.owner.log(f"[SAM] Принято объектов: {len(annotations)}; класс: {self.panel.class_id.value()}")
        self.view.clear_prompts()
        self.stop_tool()
        self.panel.status.setText(self.host.accepted_message(len(annotations)))

    def cancel(self):
        self.invalidate(cancel_load=True)
        self.panel.status.setText("Отмена… Ожидание завершения текущей операции.")

    def close(self):
        self.closed = True
        if self.job is not None:
            self.job.close()
        self.backend = None
        self.update_load_state()
