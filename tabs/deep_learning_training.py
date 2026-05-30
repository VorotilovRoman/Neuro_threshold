# deep_learning_training.py
from import_libs_internal import *
from import_libs_methods_ui import setup_deep_learning_training_ui

print("load deep_learning_training")

# ------------------------------------------------------------
# Получение версии Ultralytics YOLO (для информационных целей)
# ------------------------------------------------------------
ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    print(f"Ultralytics YOLO version: {version}")
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    print("Не удалось определить версию YOLO (ultralytics не установлена или повреждена)")


class YoloTrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обучение моделей сегментации")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(600)

        # Создаём UI (все вкладки с виджетами обучения)
        setup_deep_learning_training_ui(self)

        # Процессы для каждой модели
        self.train_process = None          # для YOLO
        self.unet_process = None           # для U‑Net
        self.deeplab_process = None
        self.segformer_process = None
        self.sam_process = None

        # Сохраняем параметры последнего запуска YOLO для анализа best.pt
        self.last_yolo_params = None

    # ---------- Логирование (общее) ----------
    def log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    # ---------- Определение параметров загруженной модели YOLO ----------
    def _detect_model_info(self, model_path):
        """Выводит в лог информацию о модели YOLO (без изменения UI)."""
        if not ULTRALYTICS_AVAILABLE:
            self.log("⚠️ Библиотека ultralytics не установлена, невозможно определить параметры модели")
            return
        try:
            from ultralytics import YOLO
            tmp_model = YOLO(model_path)
            task = getattr(tmp_model, 'task', 'unknown')
            version = '?'
            size = '?'

            if hasattr(tmp_model, 'model') and hasattr(tmp_model.model, 'model'):
                inner = tmp_model.model.model
                if hasattr(inner, 'yaml'):
                    yaml_str = str(inner.yaml)
                    if 'yolov8' in yaml_str:
                        version = '8'
                    elif 'yolo11' in yaml_str:
                        version = '11'
                    elif 'yolo26' in yaml_str:
                        version = '26'

                if hasattr(inner, 'parameters'):
                    param_count = sum(p.numel() for p in inner.parameters()) / 1e6
                    if param_count < 5:
                        size = 'n'
                    elif param_count < 10:
                        size = 's'
                    elif param_count < 25:
                        size = 'm'
                    elif param_count < 50:
                        size = 'l'
                    else:
                        size = 'x'
                    self.log(f"📊 Количество параметров: {param_count:.2f}M")

            self.log(f"🔍 Загружена модель: {os.path.basename(model_path)}")
            self.log(f"   Задача: {task}, версия YOLO: {version}, размер: {size}")
        except Exception as e:
            self.log(f"⚠️ Не удалось определить параметры модели: {e}")

    # ---------- Поиск best.pt с учётом project/name ----------
    def _analyze_best_model_yolo(self, project_path=None, experiment_name=None):
        """Поиск best.pt после обучения YOLO с учётом пользовательской папки проекта."""
        if project_path is None:
            base_dir = "runs"
        else:
            base_dir = project_path
        if experiment_name:
            base_dir = os.path.join(base_dir, experiment_name)

        if not os.path.exists(base_dir):
            self.log(f"Папка {base_dir} не найдена, поиск best.pt невозможен.")
            return

        latest_pt = None
        latest_time = 0
        # Ищем в подпапке weights внутри base_dir (стандартная структура YOLO)
        weights_dir = os.path.join(base_dir, "weights")
        if os.path.exists(weights_dir):
            best_path = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_path):
                latest_pt = best_path
                latest_time = os.path.getmtime(best_path)
        else:
            # Рекурсивный поиск на случай другой структуры
            for root, dirs, files in os.walk(base_dir):
                if 'weights' in dirs:
                    weights_path = os.path.join(root, 'weights')
                    best_path = os.path.join(weights_path, 'best.pt')
                    if os.path.exists(best_path):
                        mtime = os.path.getmtime(best_path)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_pt = best_path

        if latest_pt:
            self.log("=== Анализ лучшей модели после обучения ===")
            self._detect_model_info(latest_pt)
        else:
            self.log("Не найден файл best.pt в папке проекта.")

    # ===================== YOLO =====================
    def _run_yolo_training_from_widget(self, params):
        """Запуск обучения YOLO по параметрам из виджета."""
        if not ULTRALYTICS_AVAILABLE:
            QMessageBox.critical(self, "Ошибка", "Библиотека ultralytics не установлена.")
            return

        # Проверка обязательных параметров
        if not params.get("data"):
            QMessageBox.warning(self, "Ошибка", "Не указан YAML файл датасета.")
            return
        if not os.path.exists(params["data"]):
            self.log(f"Ошибка: YAML файл не найден: {params['data']}")
            QMessageBox.warning(self, "Ошибка", "YAML файл не найден.")
            return

        # Сохраняем параметры для последующего анализа после обучения
        self.last_yolo_params = params

        params_json = json.dumps(params, ensure_ascii=False)

        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "utils_training", "train_yolo_worker.py"
        )
        worker_script = os.path.abspath(worker_script)
        if not os.path.exists(worker_script):
            self.log("ОШИБКА: скрипт воркера не найден: " + worker_script)
            QMessageBox.critical(self, "Ошибка", "Не найден скрипт обучения train_yolo_worker.py")
            return

        self.log("=== Параметры обучения YOLO ===")
        for key, val in params.items():
            self.log(f"{key}: {val}")
        self.log("=" * 30)

        self.train_process = QProcess()
        self.train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self._on_yolo_output)
        self.train_process.finished.connect(self._on_yolo_finished)

        self.train_process.start(sys.executable, [worker_script, "--params", params_json])

        if not self.train_process.waitForStarted(3000):
            self.log("Ошибка: не удалось запустить процесс обучения YOLO")
            self.yolo_train_widget.set_running(False)
            return

        self.yolo_train_widget.set_running(True)

    def _on_yolo_output(self):
        data = self.train_process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS:"):
                parts = line.split(":")
                if len(parts) >= 4:
                    try:
                        percent = int(parts[1])
                        epoch = int(parts[2])
                        total = int(parts[3])
                        # Обновляем прогресс-бар виджета YOLO
                        bar = self.yolo_train_widget.progress_bar
                        bar.setValue(percent)
                        bar.setFormat(f"Обучение YOLO: {percent}% ({epoch}/{total} эпох)")
                    except ValueError:
                        pass
                continue
            self.log(line)

    def _on_yolo_finished(self, exit_code, exit_status):
        self.yolo_train_widget.set_running(False)

        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.log("=== Обучение YOLO успешно завершено ===")
            QMessageBox.information(self, "Обучение", "Обучение YOLO успешно завершено!")
            # Включаем кнопку открытия папки проекта
            self.yolo_train_widget.open_folder_btn.setEnabled(True)
            # Анализируем best.pt с учётом сохранённых параметров
            if self.last_yolo_params:
                proj = self.last_yolo_params.get("project")
                name = self.last_yolo_params.get("name")
                self._analyze_best_model_yolo(project_path=proj, experiment_name=name)
        else:
            self.log("=== Процесс обучения YOLO завершился с ошибкой ===")
            QMessageBox.critical(self, "Ошибка", "Обучение YOLO завершилось ошибкой. Смотрите лог.")
        self.train_process = None
        # Сброс прогресс-бара (выполняется также в set_running, но дополнительно)
        self.yolo_train_widget.progress_bar.setValue(0)
        self.yolo_train_widget.progress_bar.setFormat("Готов")

    def on_stop_training(self):
        if self.train_process and self.train_process.state() == QProcess.Running:
            self.log("Останавливаем обучение YOLO...")
            self.train_process.terminate()
            if not self.train_process.waitForFinished(5000):
                self.train_process.kill()
                self.log("Процесс YOLO принудительно завершён.")
            else:
                self.log("Процесс YOLO остановлен корректно.")
            self.yolo_train_widget.set_running(False)

    # ===================== U‑NET =====================
    def _run_unet_training_from_widget(self, params):
        """Запуск обучения U-Net по параметрам из виджета."""
        if not params.get("dataset_path") or not os.path.exists(params["dataset_path"]):
            QMessageBox.warning(self, "Ошибка", "Укажите существующую папку датасета.")
            return

        params_json = json.dumps(params, ensure_ascii=False)

        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "utils_training", "train_unet_worker.py"
        )
        worker_script = os.path.abspath(worker_script)
        if not os.path.exists(worker_script):
            self.log("ОШИБКА: скрипт воркера U-Net не найден: " + worker_script)
            QMessageBox.critical(self, "Ошибка", "Не найден скрипт обучения train_unet_worker.py")
            return

        self.log("=== Параметры обучения U‑Net ===")
        for key, val in params.items():
            self.log(f"{key}: {val}")
        self.log("=" * 30)

        self.unet_process = QProcess()
        self.unet_process.setProcessChannelMode(QProcess.MergedChannels)
        self.unet_process.readyReadStandardOutput.connect(self._on_unet_output)
        self.unet_process.finished.connect(self._on_unet_finished)

        self.unet_process.start(sys.executable, [worker_script, "--params", params_json])

        if not self.unet_process.waitForStarted(3000):
            self.log("Ошибка запуска процесса обучения U-Net")
            self.unet_train_widget.set_running(False)
            return

        self.unet_train_widget.set_running(True)

    def _on_unet_output(self):
        data = self.unet_process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS:"):
                parts = line.split(":")
                if len(parts) >= 4:
                    try:
                        percent = int(parts[1])
                        epoch = int(parts[2])
                        total = int(parts[3])
                        bar = self.unet_train_widget.progress_bar
                        bar.setValue(percent)
                        bar.setFormat(f"Обучение U‑Net: {percent}% ({epoch}/{total} эпох)")
                    except ValueError:
                        pass
                continue
            self.log(line)

    def _on_unet_finished(self, exit_code, exit_status):
        self.unet_train_widget.set_running(False)

        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.log("=== Обучение U-Net успешно завершено ===")
            QMessageBox.information(self, "Обучение", "U-Net успешно обучена!")
            self.unet_train_widget.open_folder_btn.setEnabled(True)
        else:
            self.log("=== Процесс обучения U-Net завершился с ошибкой ===")
            QMessageBox.critical(self, "Ошибка", "Обучение U-Net завершилось ошибкой. Смотрите лог.")
        self.unet_process = None
        self.unet_train_widget.progress_bar.setValue(0)
        self.unet_train_widget.progress_bar.setFormat("Готов")

    def on_stop_unet(self):
        if self.unet_process and self.unet_process.state() == QProcess.Running:
            self.log("Останавливаем обучение U-Net...")
            self.unet_process.terminate()
            if not self.unet_process.waitForFinished(5000):
                self.unet_process.kill()
                self.log("Процесс U-Net принудительно завершён.")
            else:
                self.log("Процесс U-Net остановлен корректно.")
            self.unet_train_widget.set_running(False)

    # ===================== DeepLabV3+ =====================
    def _run_deeplab_training(self, params):
        """Запуск обучения DeepLabV3+ с использованием воркера."""
        # Проверка обязательных параметров
        if not params.get("dataset_path") or not os.path.exists(params["dataset_path"]):
            QMessageBox.warning(self, "Ошибка", "Укажите существующую папку датасета (VOCdevkit или images/masks).")
            return

        params_json = json.dumps(params, ensure_ascii=False)

        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "utils_training", "train_deeplabv3_worker.py"
        )
        worker_script = os.path.abspath(worker_script)
        if not os.path.exists(worker_script):
            self.log("ОШИБКА: скрипт воркера DeepLabV3+ не найден: " + worker_script)
            QMessageBox.critical(self, "Ошибка", "Не найден скрипт обучения train_deeplabv3_worker.py")
            return

        self.log("=== Параметры обучения DeepLabV3+ ===")

        # Красивый вывод параметров с рекурсией
        def log_params(d, indent=0):
            for key, val in d.items():
                if isinstance(val, dict):
                    self.log("  " * indent + f"{key}:")
                    log_params(val, indent + 1)
                else:
                    self.log("  " * indent + f"{key}: {val}")

        log_params(params)
        self.log("=" * 30)

        self.deeplab_process = QProcess()
        self.deeplab_process.setProcessChannelMode(QProcess.MergedChannels)
        self.deeplab_process.readyReadStandardOutput.connect(self._on_deeplab_output)
        self.deeplab_process.finished.connect(self._on_deeplab_finished)

        self.deeplab_process.start(sys.executable, [worker_script, "--params", params_json])

        if not self.deeplab_process.waitForStarted(3000):
            self.log("Ошибка запуска процесса обучения DeepLabV3+")
            self.deeplab_train_widget.set_running(False)
            return

        self.deeplab_train_widget.set_running(True)

    def _on_deeplab_output(self):
        """Обработка вывода воркера DeepLabV3+."""
        data = self.deeplab_process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS:"):
                parts = line.split(":")
                if len(parts) >= 4:
                    try:
                        percent = int(parts[1])
                        epoch = int(parts[2])
                        total = int(parts[3])
                        bar = self.deeplab_train_widget.progress_bar
                        bar.setValue(percent)
                        bar.setFormat(f"Обучение DeepLabV3+: {percent}% ({epoch}/{total} эпох)")
                    except ValueError:
                        pass
                continue
            self.log(line)

    def _on_deeplab_finished(self, exit_code, exit_status):
        """Завершение обучения DeepLabV3+."""
        self.deeplab_train_widget.set_running(False)

        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.log("=== Обучение DeepLabV3+ успешно завершено ===")
            QMessageBox.information(self, "Обучение", "DeepLabV3+ успешно обучена!")
            self.deeplab_train_widget.open_folder_btn.setEnabled(True)
            # Можно добавить анализ лучшей модели, если нужно
        else:
            self.log("=== Процесс обучения DeepLabV3+ завершился с ошибкой ===")
            QMessageBox.critical(self, "Ошибка", "Обучение DeepLabV3+ завершилось ошибкой. Смотрите лог.")
        self.deeplab_process = None
        self.deeplab_train_widget.progress_bar.setValue(0)
        self.deeplab_train_widget.progress_bar.setFormat("Готов")

    def on_stop_deeplab(self):
        """Остановка обучения DeepLabV3+."""
        if self.deeplab_process and self.deeplab_process.state() == QProcess.Running:
            self.log("Останавливаем обучение DeepLabV3+...")
            self.deeplab_process.terminate()
            if not self.deeplab_process.waitForFinished(5000):
                self.deeplab_process.kill()
                self.log("Процесс DeepLabV3+ принудительно завершён.")
            else:
                self.log("Процесс DeepLabV3+ остановлен корректно.")
            self.deeplab_train_widget.set_running(False)



    # ===================== SegFormer (заглушка) =====================
    def _run_segformer_training(self, params):
        """Запуск обучения SegFormer (пока заглушка)."""
        self.log("=== Параметры обучения SegFormer ===")
        for key, val in params.items():
            self.log(f"{key}: {val}")
        self.log("=" * 30)
        QMessageBox.information(self, "Информация",
                                "Обучение SegFormer будет доступно в следующих версиях.\n"
                                "Параметры переданы в лог.")

    def on_stop_segformer(self):
        """Остановка обучения SegFormer (заглушка)."""
        self.log("Остановка обучения SegFormer (не реализовано)")

    # ===================== SAM (заглушка) =====================
    def _run_sam_training(self, params):
        """Запуск обучения SAM (пока заглушка)."""
        self.log("=== Параметры обучения SAM ===")
        for key, val in params.items():
            self.log(f"{key}: {val}")
        self.log("=" * 30)
        QMessageBox.information(self, "Информация",
                                "Обучение SAM будет доступно в следующих версиях.\n"
                                "Параметры переданы в лог.")

    def on_stop_sam(self):
        """Остановка обучения SAM (заглушка)."""
        self.log("Остановка обучения SAM (не реализовано)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloTrainWindow()
    window.show()
    sys.exit(app.exec_())