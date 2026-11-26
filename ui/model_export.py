"""Shared export controls; native model operations run in an isolated process."""
import json
from pathlib import Path
import sys
import tempfile

from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                            QProgressBar, QPushButton, QSpinBox, QTextBrowser, QVBoxLayout, QWidget)

from path_setup import get_project_root
from ui import dataset_source as sources
from utils_new.import_export.model_load import GGUF_UNSUPPORTED
from utils_new.import_export.sam_weights import MODELS


class ModelExportWidget(QWidget):
    log_message = pyqtSignal(str)
    completed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.workspace = None
        self.loaded = None
        self.cancelled = False
        layout = QVBoxLayout(self)
        hint = QLabel("1. Выберите тип и локальные веса. 2. Загрузите модель. "
                      "3. Укажите размер входа и папку результатов. 4. Выберите формат. "
                      "Экспорт проверяется обратной загрузкой на CPU; исходные веса сохраняются.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItems(["YOLO", "U-Net", "DeepLabV3+", "SAM", "ONNX / NCNN"])
        form.addRow("Тип модели:", self.kind)
        self.sam_key = QComboBox()
        for key, model in MODELS.items():
            if key != "sam3.1":
                self.sam_key.addItem(model.name, key)
        form.addRow("Архитектура SAM:", self.sam_key)
        self.sam_label = form.labelForField(self.sam_key)
        self.path = QLineEdit()
        self.path.setPlaceholderText("Локальные веса; raw U-Net/DeepLab требуют *_meta.json")
        self.browse = QPushButton("Веса…")
        self.load = QPushButton("Загрузить модель")
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(self.browse)
        row.addWidget(self.load)
        form.addRow("Модель:", row)
        self.size = QSpinBox()
        self.size.setRange(32, 4096)
        self.size.setSingleStep(32)
        self.size.setValue(640)
        self.size.setToolTip("Квадратный вход, batch=1. Для YOLO и NCNN размер должен быть кратен 32. NCNN фиксирует этот размер.")
        form.addRow("Вход, px (YOLO/NCNN: кратно 32):", self.size)
        row = QHBoxLayout()
        self.opset = QComboBox()
        self.opset.addItems(['13', '16', '18'])
        self.opset.setCurrentText('18')
        self.dynamic = QCheckBox('Динамические H/W')
        self.nms = QCheckBox('Встроить NMS (YOLO)')
        self.nms.setToolTip('Только ONNX. Conf, IoU и max_det фиксируются в графе. Для гибкой настройки оставьте NMS в runtime.')
        row.addWidget(QLabel('opset:'))
        row.addWidget(self.opset)
        row.addWidget(self.dynamic)
        row.addWidget(self.nms)
        form.addRow('ONNX:', row)
        row = QHBoxLayout()
        self.conf, self.iou, self.threshold = (QDoubleSpinBox() for _ in range(3))
        for widget, value in ((self.conf, .25), (self.iou, .45), (self.threshold, .5)):
            widget.setRange(0, 1)
            widget.setDecimals(3)
            widget.setSingleStep(.05)
            widget.setValue(value)
        self.max_det = QSpinBox()
        self.max_det.setRange(1, 3000)
        self.max_det.setValue(300)
        for caption, widget in (('Conf', self.conf), ('IoU', self.iou), ('max_det', self.max_det)):
            row.addWidget(QLabel(caption))
            row.addWidget(widget)
        form.addRow('Встроенный NMS:', row)
        self.threshold.setToolTip('Порог бинарной семантической маски сохраняется в JSON; логиты сети не меняются. Для >2 классов применяется argmax.')
        form.addRow('Порог маски (U-Net/DeepLab):', self.threshold)
        row = QHBoxLayout()
        self.output = QLineEdit(str(Path(get_project_root()) / "models" / "exports"))
        self.output_browse = sources.make_source_button("model_export_dir", self)
        row.addWidget(self.output, 1)
        row.addWidget(self.output_browse)
        form.addRow("Результаты:", row)
        layout.addLayout(form)
        self.info = QTextBrowser()
        self.info.setMinimumHeight(150)
        layout.addWidget(self.info, 1)
        row = QHBoxLayout()
        self.buttons = {}
        for format in ("ONNX", "NCNN", "GGUF"):
            button = QPushButton("Экспорт " + format)
            button.clicked.connect(lambda _, value=format: self.start_export(value))
            row.addWidget(button)
            self.buttons[format] = button
        self.stop = QPushButton("Отмена")
        self.stop.clicked.connect(self.cancel)
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("Модель не загружена. GGUF требует backend конкретной архитектуры.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.kind.currentIndexChanged.connect(self.invalidate)
        self.sam_key.currentIndexChanged.connect(self.invalidate)
        self.path.textChanged.connect(self.invalidate)
        self.browse.clicked.connect(self.choose_model)
        self.load.clicked.connect(self.inspect)
        self.output_browse.clicked.connect(self.choose_output)
        self.size.valueChanged.connect(self._controls)
        self.nms.toggled.connect(self._controls)
        self.invalidate()

    def invalidate(self, *_):
        self.loaded = None
        self.load.setStyleSheet("")
        self.info.setPlainText("Выберите и загрузите модель для просмотра параметров и доступных форматов.")
        self.sam_key.setVisible(self.kind.currentText() == "SAM")
        self.sam_label.setVisible(self.kind.currentText() == "SAM")
        if self.kind.currentText() != getattr(self, '_last_kind', None):
            self._last_kind = self.kind.currentText()
            self.dynamic.setChecked(self._last_kind != 'YOLO')
            if self._last_kind != 'YOLO':
                self.nms.setChecked(False)
        self._controls()

    def _controls(self):
        busy = self.process is not None
        for widget in (self.kind, self.sam_key, self.path, self.browse, self.load,
                       self.size, self.output, self.output_browse, self.opset, self.dynamic):
            widget.setEnabled(not busy)
        self.nms.setEnabled(not busy and self.kind.currentText() == 'YOLO')
        for widget in (self.conf, self.iou, self.max_det):
            widget.setEnabled(not busy and self.kind.currentText() == 'YOLO' and self.nms.isChecked())
        self.threshold.setEnabled(not busy and self.kind.currentText() in ('U-Net', 'DeepLabV3+'))
        self.stop.setEnabled(busy)
        for format, button in self.buttons.items():
            reason = self.loaded["capabilities"][format] if self.loaded else "Сначала загрузите исходную модель."
            if format == "GGUF":
                reason = GGUF_UNSUPPORTED
            elif self.loaded and (format == 'NCNN' or self.kind.currentText() == 'YOLO') and self.size.value() % 32:
                reason = 'Укажите размер входа, кратный 32 (например 512).'
            elif format == 'NCNN' and self.nms.isChecked():
                reason = 'NCNN выполняет NMS в runtime. Снимите флажок встроенного NMS.'
            button.setEnabled(not busy and self.loaded is not None and not reason)
            button.setToolTip(reason or "Экспорт и проверка совпадения выходов на CPU.")

    def choose_model(self):
        kind = self.kind.currentText()
        key = "model_yolo" if kind == "YOLO" else "model_onnx" if kind == "ONNX / NCNN" else "model_torch"
        path, _ = sources.get_open_file(self, key, directory=str(Path(get_project_root()) / "models"))
        if path:
            self.path.setText(path)

    def choose_output(self):
        path = sources.get_directory(self, "model_export_dir", directory=self.output.text())
        if path:
            self.output.setText(path)

    def _request(self):
        path = self.path.text().strip()
        return dict(kind=self.kind.currentText(), path=str(Path(path).expanduser().resolve()) if path else '',
                    model_key=self.sam_key.currentData())

    def inspect(self):
        self.loaded = None
        self.load.setStyleSheet("")
        self.info.clear()
        self._start(dict(self._request(), operation="inspect"))

    def start_export(self, format):
        if self.loaded is None or self.loaded['capabilities'][format]:
            return
        try:
            output = Path(self.output.text().strip()).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            if not output.is_dir():
                raise ValueError("Укажите папку результатов.")
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
            return
        self._start(dict(self._request(), operation="export", format=format,
                         output_dir=str(output.resolve()), size=self.size.value(),
                         options=dict(opset=int(self.opset.currentText()), dynamic=self.dynamic.isChecked(),
                                      nms=self.nms.isChecked(), conf=self.conf.value(), iou=self.iou.value(),
                                      max_det=self.max_det.value(), threshold=self.threshold.value())))

    def _start(self, request):
        if self.process is not None:
            return
        if not request['path'] or not Path(request['path']).exists():
            self.status.setText("Выберите существующий локальный файл или папку модели.")
            self._controls()
            return
        self.cancelled = False
        self.operation = request['operation']
        try:
            self.workspace = tempfile.TemporaryDirectory(
                prefix=".neuro-model-export-", dir=request.get('output_dir'))
            directory = Path(self.workspace.name)
            if self.operation == 'export':
                request['work_dir'] = str(directory)
            (directory / 'request.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        except OSError as error:
            if self.workspace is not None:
                self.workspace.cleanup()
                self.workspace = None
            self.status.setText('Ошибка подготовки операции: ' + str(error))
            self._controls()
            return
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(get_project_root()))
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_logs)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.progress.setRange(0, 0)
        self.status.setText("Проверка и загрузка модели…" if self.operation == 'inspect' else "Экспорт и проверка модели…")
        self._controls()
        process.start(sys.executable, ['-m', 'utils_new.import_export.model_export',
                                      '--request', str(directory / 'request.json'),
                                      '--result', str(directory / 'result.json')])

    def _read_logs(self):
        if self.process is not None:
            message = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if message.strip():
                self.log_message.emit(message.rstrip())

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code, *_):
        if self.process is None:
            return
        self._read_logs()
        try:
            result_path = Path(self.workspace.name) / 'result.json'
            result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            if self.cancelled:
                self.status.setText("Операция отменена.")
            elif code != 0 or not result.get('ok'):
                self.status.setText("Ошибка: " + result.get('error',
                    'Воркер завершился без результата. Возможен сбой native backend или неподдерживаемая операция. '
                    'Экспорт не опубликован; смотрите лог, попробуйте ONNX или другой размер входа.'))
            elif self.operation == 'inspect':
                self.loaded = result['value']
                metadata = self.loaded['metadata']
                parameters = self.loaded['parameters']
                lines = [f"Тип: {self.loaded['kind']}", f"Файл: {self.loaded['path']}",
                         f"Параметров: {parameters:,}" if parameters is not None else "Параметры: готовый граф инференса"]
                captions = dict(model_type='Модель', task='Задача', model_key='Архитектура SAM',
                                architecture='Архитектура', encoder='Энкодер', backbone='Backbone',
                                input_size='Размер входа', num_classes='Классов (с фоном)',
                                normalize='Нормализация ImageNet', resize_mode='Масштабирование', names='Классы',
                                class_names='Имена pixel ID', epoch='Эпоха checkpoint',
                                validation_metrics='Метрики валидации', metrics='Метрики SAM',
                                export_parameters='Параметры экспорта')
                for key, caption in captions.items():
                    if key in metadata:
                        value = metadata[key]
                        if isinstance(value, bool):
                            value = 'да' if value else 'нет'
                        lines.append(f"{caption}: {value}")
                lines.append("\nДоступность экспорта:")
                for format in ('ONNX', 'NCNN', 'GGUF'):
                    reason = self.loaded['capabilities'][format]
                    if format == 'GGUF':
                        reason = 'нет совместимого runtime выбранной архитектуры; подробности — в подсказке кнопки.'
                    lines.append(f"{format}: {reason or 'доступен; проверка выходов на CPU'}")
                self.info.setPlainText('\n'.join(lines))
                size = self.loaded['metadata'].get('input_size')
                if isinstance(size, int):
                    self.size.setValue(size)
                self.threshold.setValue(self.loaded['metadata'].get('threshold', .5))
                self.load.setStyleSheet('background-color: #b7e4c7; color: #163b24;')
                self.status.setText("Модель загружена и проверена на CPU. Доступные форматы указаны ниже.")
            else:
                self.status.setText("Сохранено: " + result['value'])
                self.log_message.emit("[EXPORT][OK] " + result['value'])
            self.completed.emit(result)
        except (OSError, ValueError, KeyError) as error:
            self.status.setText("Не удалось прочитать результат: " + str(error))
        finally:
            self.process.deleteLater()
            self.process = None
            try:
                self.workspace.cleanup()
            except OSError as error:
                self.log_message.emit("[EXPORT] Не удалось удалить временную папку: " + str(error))
            self.workspace = None
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if code == 0 and not self.cancelled else 0)
            self._controls()

    def cancel(self):
        if self.process is None or self.cancelled:
            return
        self.cancelled = True
        # PNNX is a child executable. Stop it too before removing the staging area.
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            try:
                pid = int(self.process.processId())
                if pid:
                    for child in reversed(psutil.Process(pid).children(recursive=True)):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
            except (OSError, psutil.Error):
                pass
        process = self.process
        process.terminate()
        QTimer.singleShot(1500, lambda: process.kill() if self.process is process else None)

    def shutdown(self):
        busy = self.process is not None
        if busy:
            self.cancel()
        return busy
