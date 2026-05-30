from import_libs_internal import *
from main_ui import StartupDialog

COLLECT_LIBS_INFO = False


class ImagePlayer(QMainWindow):
    def __init__(self, selected_tabs):
        super().__init__()
        self.selected_tabs = selected_tabs
        self.setWindowTitle("Threshold‑Researcher")

        # Настройки и тема
        self.settings_manager = settings
        self.settings_manager.settings_changed.connect(self.on_settings_changed)
        apply_theme(QApplication.instance())

        # Виджет с вкладками
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Хранилища для виджетов
        self.demo_window = None
        self.demo_tab_index = -1          # индекс вкладки демо
        self.yolo_sort_window = None
        self.viewing_dataset_window = None
        self.ThresholdWindow_window = None
        self.GradientMethodsWindow_window = None
        self.InteractiveMethodsWindow_window = None
        self.TraditionalMLWindow_window = None
        self.DeepLearningWindow_window = None

        # Создаём вкладки
        self._create_selected_tabs()

    def on_settings_changed(self, new_settings):
        apply_theme(QApplication.instance())

    def on_tab_changed(self, index):
        """При переключении вкладки обновляем пресеты в демо-вкладке, если она активна."""
        if index == self.demo_tab_index and self.demo_window is not None:
            self.demo_window.update_presets()
            self.demo_window.log("Пресеты обновлены при переключении вкладки")

    def _add_tab(self, widget, title):
        """Добавляет вкладку, извлекая centralWidget если widget — QMainWindow.
           Возвращает индекс добавленной вкладки."""
        if isinstance(widget, QMainWindow):
            content = widget.centralWidget()
            if content is None:
                content = widget
        else:
            content = widget
        index = self.tab_widget.addTab(content, title)
        # Если это демо-окно, запоминаем индекс
        if widget is self.demo_window:
            self.demo_tab_index = index
        return index

    def _create_selected_tabs(self):
        for tab_id, tab_title in self.selected_tabs:
            if tab_id == "threshold":
                from tabs.threshold_methods import ThresholdWindow
                self.ThresholdWindow_window = ThresholdWindow()
                self._add_tab(self.ThresholdWindow_window, tab_title)

            elif tab_id == "gradient":
                from tabs.gradient_methods import GradientMethodsWindow
                self.GradientMethodsWindow_window = GradientMethodsWindow()
                self._add_tab(self.GradientMethodsWindow_window, tab_title)

            elif tab_id == "interactive":
                from tabs.interactive_methods import InteractiveMethodsWindow
                self.InteractiveMethodsWindow_window = InteractiveMethodsWindow()
                self._add_tab(self.InteractiveMethodsWindow_window, tab_title)

            elif tab_id == "traditional_ml":
                from tabs.traditional_ml_methods import TraditionalMLWindow
                self.TraditionalMLWindow_window = TraditionalMLWindow()
                self._add_tab(self.TraditionalMLWindow_window, tab_title)

            elif tab_id == "deep_learning":
                from tabs.deep_learning_methods import DeepLearningWindow
                self.DeepLearningWindow_window = DeepLearningWindow()
                self._add_tab(self.DeepLearningWindow_window, tab_title)

            elif tab_id == "reports":
                from tabs.model_research import ModelResearchApp
                self._add_tab(ModelResearchApp(), tab_title)

            elif tab_id == "labeler":
                from tabs.layout_dataset import Labeler
                self._add_tab(Labeler(), tab_title)

            elif tab_id == "test":
                from tabs.viewing_dataset import ViewingDataset
                self.viewing_dataset_window = ViewingDataset()
                self._add_tab(self.viewing_dataset_window, tab_title)

            elif tab_id == "dataset":
                from tabs.preparing_dataset_yaml import DatasetPreparationWindow
                self._add_tab(DatasetPreparationWindow(), tab_title)

            elif tab_id == "yolo_train":
                from tabs.deep_learning_training import YoloTrainWindow
                self._add_tab(YoloTrainWindow(), tab_title)

            elif tab_id == "yolo_demo":
                from tabs.demo_ensemble_methods import YoloDemoWindow
                self.demo_window = YoloDemoWindow()
                self._add_tab(self.demo_window, tab_title)

            elif tab_id == "yolo_sort":
                from tabs.yolo_find_img import FindImagesWindow
                self.yolo_sort_window = FindImagesWindow()
                self._add_tab(self.yolo_sort_window, tab_title)

            elif tab_id == "settings":
                from ui.settings_ui import SettingsWidget
                self._add_tab(SettingsWidget(), tab_title)

            else:
                print(f"Неизвестная вкладка: {tab_id}")

    def closeEvent(self, event):
        # Проверяем глобальный флаг через переменную окружения
        collect_info = os.environ.get('COLLECT_LIBS_INFO', '0').lower() in ('1', 'true', 'yes')
        if collect_info:
            print("Сбор информации о библиотеках включён. Сохраняем libs_info.txt...")
            self._save_libs_info()
        else:
            print("Сбор информации о библиотеках отключён. Файл libs_info.txt не создан.")
        event.accept()

    def _save_libs_info(self):
        """Собирает версии ключевых библиотек и записывает в libs_info.txt."""
        libs = {}

        def get_version(module, attr='__version__'):
            try:
                return getattr(module, attr, 'unknown')
            except:
                return 'not available'

        try:
            import cv2
            libs['opencv-python'] = get_version(cv2)
        except ImportError:
            libs['opencv-python'] = 'not installed'

        try:
            import torch
            libs['torch'] = get_version(torch)
        except ImportError:
            libs['torch'] = 'not installed'

        try:
            import numpy
            libs['numpy'] = get_version(numpy)
        except ImportError:
            libs['numpy'] = 'not installed'

        try:
            import albumentations as A
            libs['albumentations'] = get_version(A)
        except ImportError:
            libs['albumentations'] = 'not installed'

        try:
            import ultralytics
            libs['ultralytics'] = get_version(ultralytics)
        except ImportError:
            libs['ultralytics'] = 'not installed'

        try:
            import matplotlib
            libs['matplotlib'] = get_version(matplotlib)
        except ImportError:
            libs['matplotlib'] = 'not installed'

        try:
            import sklearn
            libs['scikit-learn'] = get_version(sklearn)
        except ImportError:
            libs['scikit-learn'] = 'not installed'

        try:
            import skimage
            libs['scikit-image'] = get_version(skimage)
        except ImportError:
            libs['scikit-image'] = 'not installed'

        try:
            import PIL
            libs['Pillow'] = get_version(PIL)
        except ImportError:
            libs['Pillow'] = 'not installed'

        try:
            import yaml
            libs['PyYAML'] = get_version(yaml)
        except ImportError:
            libs['PyYAML'] = 'not installed'

        try:
            import onnxruntime as ort
            libs['onnxruntime'] = ort.__version__ if hasattr(ort, '__version__') else 'unknown'
        except ImportError:
            libs['onnxruntime'] = 'not installed'

        try:
            from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
            libs['PyQt5'] = PYQT_VERSION_STR
            libs['Qt'] = QT_VERSION_STR
        except ImportError:
            libs['PyQt5'] = 'not installed'

        libs['Python'] = sys.version

        extra_libs = {
            'torchvision': 'torchvision',
            'pandas': 'pandas',
            'seaborn': 'seaborn',
            'requests': 'requests',
            'tqdm': 'tqdm',
            'networkx': 'networkx',
            'joblib': 'joblib',
            'scipy': 'scipy',
            'gradio': 'gradio',
            'imutils': 'imutils',
        }

        for pkg_name, module_name in extra_libs.items():
            try:
                mod = __import__(module_name)
                libs[pkg_name] = get_version(mod)
            except ImportError:
                libs[pkg_name] = 'not installed'

        import platform
        libs['OS'] = platform.platform()
        libs['CUDA available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            libs['CUDA version'] = torch.version.cuda
            libs['GPU device'] = torch.cuda.get_device_name(0)

        lines = [f"Library versions collected at {datetime.now()}"]
        lines.append("=" * 50)
        for name, ver in sorted(libs.items()):
            lines.append(f"{name}: {ver}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'libs_info.txt')
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"Library versions saved to {file_path}")
        except Exception as e:
            print(f"Failed to write libs_info.txt: {e}")


def main():
    app = QApplication(sys.argv)

    splash_pixmap = QPixmap(500, 200)
    splash_pixmap.fill(Qt.white)
    painter = QPainter(splash_pixmap)
    painter.setPen(QColor(50, 50, 50))
    painter.drawText(splash_pixmap.rect(), Qt.AlignCenter, "Threshold‑Researcher\nЗагрузка модулей...")
    painter.end()

    splash = QSplashScreen(splash_pixmap)
    splash.show()
    splash.showMessage("Инициализация приложения...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()

    dialog = StartupDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    selected = dialog.get_selected_tabs()
    if not selected:
        QMessageBox.warning(None, "Предупреждение", "Не выбрано ни одной вкладки. Приложение будет закрыто.")
        sys.exit(0)

    splash.showMessage("Загрузка выбранных вкладок...\nЭто может занять некоторое время",
                       Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()

    window = ImagePlayer(selected)

    splash.finish(window)

    window.show()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()