"""Explicit online preparation of local SAM assets for both SAM hosts."""

import os
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from ui.segmentation.jobs import BackgroundTask
from utils_new.import_export.sam_weights import DEFAULT_OUTPUT, MODELS, download_model, plan_model_download


class SamDownloadTask(BackgroundTask):
    # Byte counts may exceed Qt's signed 32-bit int (e.g. SAM ViT-H).
    progress = pyqtSignal(str, str, object, object)

    def __init__(self, plan, token):
        super().__init__(lambda cancel: download_model(
            plan, cancel=cancel, token=token,
            progress=lambda *args: self.publish(self.progress, *args)))

    def run(self):
        try:
            super().run()
        finally:
            self.compute = None  # Release credentials even when the window was closed.


class SamDownloads:
    def __init__(self, controller):
        self.controller = controller
        self.panel = panel = controller.panel
        self.paths = {}
        panel.weights.textChanged.connect(self.remember_weights)
        panel.settings_restored.connect(self.remember_weights)
        self.remember_weights()
        # activated is a user action; preset restoration must never start a download.
        panel.model.activated.connect(self.select_architecture)
        panel.download.clicked.connect(self.request)

    def remember_weights(self, *_):
        path = self.panel.weights.text().strip()
        if path and Path(path).is_file():
            self.paths[self.panel.model.currentData()] = path

    def plan(self, *, selecting=False):
        panel = self.panel
        key = panel.model.currentData()
        weights = panel.weights.text().strip()
        if selecting:
            expected = Path(MODELS[key].assets[0].path).name.casefold()
            weights = weights if Path(weights).name.casefold() == expected else self.paths.get(key, "")
        weights = weights if weights and Path(weights).is_file() else None
        tokenizer = panel.tokenizer.text().strip()
        tokenizer = tokenizer if tokenizer and Path(tokenizer).is_file() else None
        return plan_model_download(key, DEFAULT_OUTPUT, weights_file=weights, tokenizer_file=tokenizer)

    def apply_paths(self, plan, *, existing_only=False):
        panel = self.panel
        blockers = [QSignalBlocker(panel.weights), QSignalBlocker(panel.tokenizer)]
        for asset, path in plan.files:
            text = str(path) if not existing_only or path.is_file() else ""
            if asset.kind == "weights":
                panel.weights.setText(text)
                if text:
                    self.paths[plan.model_key] = text
            elif asset.kind == "tokenizer":
                panel.tokenizer.setText(text)
        if plan.model_key != "sam3":
            panel.tokenizer.clear()
        del blockers
        self.controller.invalidate()
        self.controller.host.model_changed()

    def select_architecture(self, *_):
        self.request(selecting=True)

    def request(self, *_, selecting=False):
        controller, panel = self.controller, self.panel
        if controller.closed or controller.job is not None or not controller.host.can_start():
            return
        try:
            plan = self.plan(selecting=selecting)
        except (OSError, ValueError) as error:
            QMessageBox.warning(controller.owner, "Файлы SAM", str(error))
            return
        self.apply_paths(plan, existing_only=True)
        if not plan.missing:
            panel.status.setText("Локальные файлы найдены. Нажмите «Загрузить SAM» для запуска модели.")
            return
        details = "\n".join(f"• {path}" for _, path in plan.missing)
        answer = QMessageBox.question(
            controller.owner, "Скачать файлы SAM?",
            f"{MODELS[plan.model_key].name}: отсутствуют файлы:\n{details}\n\n"
            "Скачать их из официальных источников? Веса могут занимать несколько ГБ. "
            "Готовые файлы будут проверены и использованы повторно.\n"
            "Скачивание идёт в фоне; его можно отменить. Пакеты Python устанавливаются отдельно.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            panel.status.setText("Скачивание не запущено. Выберите локальные веса или нажмите «Скачать файлы…».")
            return
        token = os.environ.get("HF_TOKEN", "").strip() or None
        if plan.model_key == "sam3" and any(asset.is_weights for asset, _ in plan.missing) and not token:
            token, accepted = QInputDialog.getText(
                controller.owner, "Доступ к весам SAM3",
                "Для SAM3 примите условия и получите доступ к https://huggingface.co/facebook/sam3.\n"
                "Введите токен Hugging Face с правом чтения (хранится только на время скачивания):",
                QLineEdit.Password)
            if not accepted:
                panel.status.setText("Скачивание SAM3 отменено.")
                return
            token = token.strip() or None
        key = panel.model.currentData()
        controller.invalidate()
        job = SamDownloadTask(plan, token)

        def progress(name, phase, done, total):
            if controller.closed or controller.job is not job or job.cancelled:
                return
            if total and phase == "Скачивание":
                panel.progress.setRange(0, 100)
                panel.progress.setValue(min(100, int(100 * done / total)))
            else:
                panel.progress.setRange(0, 0)
            size = f" — {done / (1024 * 1024):.1f} МБ" if done else ""
            if total and done:
                size += f" из {total / (1024 * 1024):.1f} МБ"
            panel.status.setText(f"{phase}: {Path(name).name}{size}")

        def accept(paths):
            if panel.model.currentData() != key:
                return
            self.apply_paths(plan)
            panel.status.setText("Файлы готовы. Нажмите «Загрузить SAM», чтобы загрузить модель в память.")
            for path in paths.values():
                controller.owner.log(f"[SAM] Локальный файл готов: {path}")

        job.progress.connect(progress, Qt.QueuedConnection)
        controller._start(None, accept, "Подготовка файлов SAM…", loading=True, job=job)
