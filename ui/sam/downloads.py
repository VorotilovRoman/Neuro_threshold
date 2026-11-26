"""Explicit online preparation of local SAM assets for both SAM hosts."""

from pathlib import Path

from PyQt5.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMenu, QMessageBox

from ui.segmentation.jobs import BackgroundTask
from utils_new.import_export.sam_weights import (DEFAULT_OUTPUT, DOWNLOAD_MODELS, DOWNLOAD_ONLY_MODELS,
    download_model, gated_repository, plan_model_download, resolve_hf_token)


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
        self.menu = QMenu(panel.download_more)
        for key, model in DOWNLOAD_ONLY_MODELS.items():
            action = self.menu.addAction(model.name)
            action.triggered.connect(lambda _, model_key=key: self.request(model_key=model_key))
        panel.download_more.setMenu(self.menu)

    def remember_weights(self, *_):
        path = self.panel.weights.text().strip()
        if path and Path(path).is_file():
            self.paths[self.panel.model.currentData()] = path

    def plan(self, *, selecting=False):
        panel = self.panel
        key = panel.model.currentData()
        weights = panel.weights.text().strip()
        if selecting:
            expected = Path(DOWNLOAD_MODELS[key].assets[0].path).name.casefold()
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

    def request(self, *_, selecting=False, model_key=None):
        controller, panel = self.controller, self.panel
        if controller.closed or controller.job is not None or not controller.host.can_start():
            return
        standalone = model_key in DOWNLOAD_ONLY_MODELS
        try:
            plan = plan_model_download(model_key, DEFAULT_OUTPUT) if standalone else self.plan(selecting=selecting)
        except (OSError, ValueError) as error:
            QMessageBox.warning(controller.owner, "Файлы SAM", str(error))
            return
        if not standalone:
            self.apply_paths(plan, existing_only=True)
        if not plan.missing:
            panel.status.setText(f"Файлы SAM3.1 уже есть: {plan.files[0][1]}. Видеобэкенд не подключён к вкладкам."
                                if standalone else "Локальные файлы найдены. Нажмите «Загрузить SAM» для запуска модели.")
            return
        details = "\n".join(f"• {path}" for _, path in plan.missing)
        answer = QMessageBox.question(
            controller.owner, "Скачать файлы SAM?",
            f"{DOWNLOAD_MODELS[plan.model_key].name}: отсутствуют файлы:\n{details}\n\n"
            "Скачать их из официальных источников? Веса могут занимать несколько ГБ. "
            "Готовые файлы будут проверены и использованы повторно.\n"
            "Скачивание идёт в фоне; его можно отменить. Пакеты Python устанавливаются отдельно.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            panel.status.setText("Скачивание не запущено. Выберите локальные веса или нажмите «Скачать файлы…».")
            return
        repository = next((gated_repository(asset) for asset, _ in plan.missing if gated_repository(asset)), None)
        token = resolve_hf_token() if repository else None
        if repository and not token:
            token, accepted = QInputDialog.getText(
                controller.owner, "Доступ к Hugging Face",
                f"Примите условия и получите доступ к {repository}.\n"
                "Введите токен Hugging Face с правом чтения (хранится только на время скачивания):",
                QLineEdit.Password)
            if not accepted:
                panel.status.setText("Скачивание отменено.")
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
            if standalone:
                panel.status.setText(f"SAM3.1 скачана: {plan.files[0][1]}. Нужен отдельный видеобэкенд; текущая модель не заменена.")
            else:
                self.apply_paths(plan)
                panel.status.setText("Файлы готовы. Нажмите «Загрузить SAM», чтобы загрузить модель в память.")
            for path in paths.values():
                controller.owner.log(f"[SAM] Локальный файл готов: {path}")

        job.progress.connect(progress, Qt.QueuedConnection)
        controller._start(None, accept, "Подготовка файлов SAM…", loading=True, job=job)
