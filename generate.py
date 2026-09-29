import sys
import os
import traceback
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QFileDialog, QMessageBox, QListWidget,
                             QListWidgetItem, QAbstractItemView, QColorDialog,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor
from PIL import Image, ImageDraw, ImageFont

# ---------------------- Логирование ----------------------
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

# ---------------------- Индекс -> подпись (а) б) в) ...) ----------------------
def index_to_label(idx):
    """
    0 -> 'а)', 1 -> 'б)', ..., 32 -> 'я)', 33 -> 'а1)', 34 -> 'б1)', ...
    """
    alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'  # 33 буквы
    base = len(alphabet)
    if idx < base:
        return f"{alphabet[idx]})"
    else:
        letter = alphabet[idx % base]
        number = idx // base
        return f"{letter}{number})"

# ---------------------- Основное окно приложения ----------------------
class CollageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        log_event("Инициализация приложения")
        self.setWindowTitle("Генератор коллажа")
        self.setMinimumSize(900, 700)

        self.image_paths = []
        self.current_pil_image = None
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update_collage)
        self._updating = False

        # Настройки коллажа
        self.bg_color = (50, 50, 50)
        self.separator_type = "none"
        self.show_labels = False
        self.keep_aspect_ratio = False   # новая опция

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---------- Панель управления ----------
        control_layout = QHBoxLayout()

        self.btn_load = QPushButton("Выбрать снимки")
        self.btn_load.clicked.connect(self.load_images)
        control_layout.addWidget(self.btn_load)

        self.label_count = QLabel("Загружено: 0")
        control_layout.addWidget(self.label_count)

        control_layout.addSpacing(10)

        self.btn_bg_color = QPushButton("Цвет фона")
        self.btn_bg_color.setStyleSheet("background-color: rgb(50,50,50);")
        self.btn_bg_color.clicked.connect(self.choose_bg_color)
        control_layout.addWidget(self.btn_bg_color)

        control_layout.addWidget(QLabel("Разделители:"))
        self.combo_separator = QComboBox()
        self.combo_separator.addItems(["Нет", "Чёрная рамка", "Белая рамка"])
        self.combo_separator.currentTextChanged.connect(self.on_separator_changed)
        control_layout.addWidget(self.combo_separator)

        self.check_labels = QCheckBox("Показывать подписи (а) б) в)...)")
        self.check_labels.toggled.connect(self.on_labels_toggled)
        control_layout.addWidget(self.check_labels)

        # Новая галочка: сохранять пропорции
        self.check_keep_ratio = QCheckBox("Ресайз без растяжения (сохранять пропорции)")
        self.check_keep_ratio.toggled.connect(self.on_keep_ratio_toggled)
        control_layout.addWidget(self.check_keep_ratio)

        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("Строки:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 10)
        self.spin_rows.setValue(2)
        self.spin_rows.valueChanged.connect(self.schedule_update)
        control_layout.addWidget(self.spin_rows)

        control_layout.addWidget(QLabel("Столбцы:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 10)
        self.spin_cols.setValue(3)
        self.spin_cols.valueChanged.connect(self.schedule_update)
        control_layout.addWidget(self.spin_cols)

        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("Ширина коллажа:"))
        self.spin_width = QSpinBox()
        self.spin_width.setRange(100, 4096)
        self.spin_width.setValue(1024)
        self.spin_width.valueChanged.connect(self.schedule_update)
        control_layout.addWidget(self.spin_width)

        control_layout.addWidget(QLabel("Высота коллажа:"))
        self.spin_height = QSpinBox()
        self.spin_height.setRange(100, 4096)
        self.spin_height.setValue(1024)
        self.spin_height.valueChanged.connect(self.schedule_update)
        control_layout.addWidget(self.spin_height)

        control_layout.addStretch()

        self.btn_export = QPushButton("Экспорт (PNG)")
        self.btn_export.clicked.connect(self.export_collage)
        control_layout.addWidget(self.btn_export)

        main_layout.addLayout(control_layout)

        # ---------- Список и область коллажа ----------
        content_layout = QHBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setDragEnabled(True)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.setDropIndicatorShown(True)
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.model().rowsMoved.connect(self.schedule_update)
        self.list_widget.model().rowsInserted.connect(self.schedule_update)
        self.list_widget.model().rowsRemoved.connect(self.schedule_update)
        self.list_widget.setMinimumWidth(200)
        content_layout.addWidget(self.list_widget)

        self.label_collage = QLabel()
        self.label_collage.setAlignment(Qt.AlignCenter)
        self.label_collage.setStyleSheet("border: 1px solid gray; background-color: #2d2d2d;")
        self.label_collage.setMinimumSize(400, 400)
        content_layout.addWidget(self.label_collage, stretch=1)

        main_layout.addLayout(content_layout, stretch=1)

        log_event("Приложение инициализировано, запускаем первое обновление")
        self.schedule_update()

    # ---------------------- Обработчики ----------------------
    def choose_bg_color(self):
        log_event("Выбор цвета фона")
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = (color.red(), color.green(), color.blue())
            log_event(f"Выбран цвет: RGB{self.bg_color}")
            self.btn_bg_color.setStyleSheet(f"background-color: rgb({self.bg_color[0]},{self.bg_color[1]},{self.bg_color[2]});")
            self.schedule_update()

    def on_separator_changed(self, text):
        if text == "Нет":
            self.separator_type = "none"
        elif text == "Чёрная рамка":
            self.separator_type = "black"
        elif text == "Белая рамка":
            self.separator_type = "white"
        log_event(f"Выбран тип разделителя: {self.separator_type}")
        self.schedule_update()

    def on_labels_toggled(self, checked):
        self.show_labels = checked
        log_event(f"Подписи {'включены' if checked else 'выключены'}")
        self.schedule_update()

    def on_keep_ratio_toggled(self, checked):
        self.keep_aspect_ratio = checked
        log_event(f"Ресайз без растяжения {'включён' if checked else 'выключен'}")
        self.schedule_update()

    def schedule_update(self):
        log_event(f"Запланировано обновление коллажа (строки={self.spin_rows.value()}, столбцы={self.spin_cols.value()}, ширина={self.spin_width.value()}, высота={self.spin_height.value()})")
        if not self.update_timer.isActive():
            self.update_timer.start(100)

    def sync_paths_from_list(self):
        paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(Qt.UserRole)
            if path:
                paths.append(path)
        old_len = len(self.image_paths)
        self.image_paths = paths
        new_len = len(self.image_paths)
        if old_len != new_len:
            log_event(f"Список изображений синхронизирован: было {old_len}, стало {new_len}")
        self.label_count.setText(f"Загружено: {len(self.image_paths)}")

    def load_images(self):
        log_event("Вызвана загрузка изображений")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif);;Все файлы (*)"
        )
        if files:
            log_event(f"Выбрано файлов: {len(files)}")
            self.list_widget.clear()
            for path in files:
                name = os.path.basename(path)
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, path)
                self.list_widget.addItem(item)
                log_event(f"  Добавлен файл: {path}")
            self.sync_paths_from_list()
            self.schedule_update()
        else:
            log_event("Загрузка отменена")

    def _get_font(self, size):
        try:
            font_paths = [
                "arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def create_collage(self):
        log_event("Начало создания коллажа (PIL)")
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        collage_width = self.spin_width.value()
        collage_height = self.spin_height.value()
        log_event(f"Параметры: строки={rows}, столбцы={cols}, ширина={collage_width}, высота={collage_height}, разделитель={self.separator_type}, подписи={self.show_labels}, keep_ratio={self.keep_aspect_ratio}")

        if collage_width < cols:
            collage_width = cols
            self.spin_width.blockSignals(True)
            self.spin_width.setValue(collage_width)
            self.spin_width.blockSignals(False)
        if collage_height < rows:
            collage_height = rows
            self.spin_height.blockSignals(True)
            self.spin_height.setValue(collage_height)
            self.spin_height.blockSignals(False)

        separator_width = 0
        separator_color = None
        if self.separator_type == "black":
            separator_width = 2
            separator_color = (0, 0, 0)
        elif self.separator_type == "white":
            separator_width = 2
            separator_color = (255, 255, 255)

        total_sep_w = separator_width * (cols - 1)
        total_sep_h = separator_width * (rows - 1)
        available_w = collage_width - total_sep_w
        available_h = collage_height - total_sep_h

        if available_w < cols or available_h < rows:
            log_event("Ошибка: размеры слишком малы для разделителей")
            return None

        cell_w = available_w // cols
        cell_h = available_h // rows
        if cell_w < 1 or cell_h < 1:
            log_event("Ошибка: размер ячейки меньше 1 пикселя")
            return None

        log_event(f"Размер ячейки: {cell_w}x{cell_h}")

        collage = Image.new("RGB", (collage_width, collage_height), color=self.bg_color)
        self.sync_paths_from_list()

        font = None
        if self.show_labels:
            font_size = max(12, int(min(cell_h, cell_w) * 0.12))
            font = self._get_font(font_size)
            log_event(f"Шрифт для подписей: размер {font_size}")

        # Фиксированный отступ от левого и нижнего краёв ячейки (в пикселях)
        LABEL_MARGIN = 10

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            x1 = col * (cell_w + separator_width)
            y1 = row * (cell_h + separator_width)

            if idx < len(self.image_paths):
                img_path = self.image_paths[idx]
                try:
                    log_event(f"  Обработка ячейки {idx}: {os.path.basename(img_path)}")
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    if self.keep_aspect_ratio:
                        # Масштабируем с сохранением пропорций, вписывая в ячейку
                        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                        # Создаём временный холст размером с ячейку
                        temp = Image.new("RGB", (cell_w, cell_h), color=self.bg_color)
                        # Центрируем изображение
                        offset_x = (cell_w - img.width) // 2
                        offset_y = (cell_h - img.height) // 2
                        temp.paste(img, (offset_x, offset_y))
                        collage.paste(temp, (x1, y1))
                    else:
                        # Обычное растяжение на весь размер ячейки
                        img_resized = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                        collage.paste(img_resized, (x1, y1))

                    if self.show_labels:
                        draw = ImageDraw.Draw(collage)
                        label_text = index_to_label(idx)
                        # Получаем размер текста
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        # Левая координата: x1 + фиксированный отступ
                        text_x = x1 + LABEL_MARGIN
                        # Нижняя координата: y1 + cell_h - отступ - высота текста
                        text_y = y1 + cell_h - LABEL_MARGIN - text_height
                        # Обводка (чёрная) и текст (белый)
                        outline_color = (0, 0, 0)
                        text_color = (255, 255, 255)
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            draw.text((text_x + dx, text_y + dy), label_text, font=font, fill=outline_color)
                        draw.text((text_x, text_y), label_text, font=font, fill=text_color)
                except Exception as e:
                    log_event(f"    ОШИБКА: {e}")
                    traceback.print_exc()
            else:
                log_event(f"  Ячейка {idx} пустая")

        # Рисуем разделители
        if separator_width > 0:
            for row in range(1, rows):
                y = row * (cell_h + separator_width) - separator_width
                for x in range(collage_width):
                    for w in range(separator_width):
                        if y + w < collage_height:
                            collage.putpixel((x, y + w), separator_color)
            for col in range(1, cols):
                x = col * (cell_w + separator_width) - separator_width
                for y in range(collage_height):
                    for w in range(separator_width):
                        if x + w < collage_width:
                            collage.putpixel((x + w, y), separator_color)

        log_event("Создание коллажа завершено")
        return collage

    def _do_update_collage(self):
        if self._updating:
            log_event("Обновление уже выполняется, пропускаем")
            return
        self._updating = True
        log_event(">>> НАЧАЛО ОБНОВЛЕНИЯ КОЛЛАЖА В UI <<<")
        try:
            if not self.image_paths:
                log_event("Нет загруженных изображений, показываем заглушку")
                self.label_collage.setText("Нет изображений\nНажмите 'Выбрать снимки'")
                self.label_collage.setAlignment(Qt.AlignCenter)
                self.current_pil_image = None
                return

            pil_image = self.create_collage()
            if pil_image is None:
                log_event("Ошибка: create_collage вернул None")
                self.label_collage.setText("Ошибка: некорректные размеры коллажа")
                self.current_pil_image = None
                return

            self.current_pil_image = pil_image
            log_event(f"Коллаж создан: размер {pil_image.width}x{pil_image.height}")

            data = pil_image.tobytes("raw", "RGB")
            bytes_per_line = pil_image.width * 3
            qimage = QImage(data, pil_image.width, pil_image.height, bytes_per_line, QImage.Format_RGB888)
            if qimage.isNull():
                raise RuntimeError("QImage is null")
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.label_collage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_collage.setPixmap(scaled)
            log_event("Коллаж отображён в UI")
        except Exception as e:
            log_event(f"ИСКЛЮЧЕНИЕ: {e}")
            traceback.print_exc()
            self.label_collage.setText(f"Ошибка создания коллажа:\n{str(e)}")
            self.current_pil_image = None
        finally:
            self._updating = False
            log_event("<<< ОБНОВЛЕНИЕ ЗАВЕРШЕНО >>>")

    def export_collage(self):
        log_event("Экспорт коллажа")
        if not self.image_paths:
            log_event("Экспорт отменён: нет изображений")
            QMessageBox.warning(self, "Нет изображений", "Загрузите изображения перед экспортом.")
            return

        if self.current_pil_image is None:
            log_event("Текущий коллаж отсутствует, принудительно обновляем")
            self._do_update_collage()
            if self.current_pil_image is None:
                log_event("Экспорт отменён: коллаж не создан")
                QMessageBox.warning(self, "Ошибка", "Коллаж не был создан.")
                return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить коллаж", "collage.png", "PNG (*.png)")
        if file_path:
            log_event(f"Сохранение в {file_path}")
            try:
                self.current_pil_image.save(file_path, "PNG")
                log_event("Сохранение успешно")
                QMessageBox.information(self, "Экспорт", f"Коллаж сохранён:\n{file_path}")
            except Exception as e:
                log_event(f"ОШИБКА сохранения: {e}")
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")
        else:
            log_event("Сохранение отменено")

    def resizeEvent(self, event):
        log_event(f"Изменён размер окна: {event.size().width()}x{event.size().height()}")
        if self.current_pil_image is not None:
            data = self.current_pil_image.tobytes("raw", "RGB")
            bytes_per_line = self.current_pil_image.width * 3
            qimage = QImage(data, self.current_pil_image.width, self.current_pil_image.height,
                            bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.label_collage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_collage.setPixmap(scaled)
        else:
            self.schedule_update()
        super().resizeEvent(event)


if __name__ == "__main__":
    log_event("Запуск приложения")
    app = QApplication(sys.argv)
    window = CollageApp()
    window.show()
    log_event("Главное окно показано")
    sys.exit(app.exec_())