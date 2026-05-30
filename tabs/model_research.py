from import_libs_internal import *
from import_libs_methods_ui import setup_model_research_ui
from path_setup import get_project_root




class ModelResearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Research")
        self.setGeometry(200, 200, 1200, 700)

        self.df_full = None
        self.current_folder = ""
        self.current_display_df = None
        self.is_aggregated = False
        self.is_top_methods = False   # флаг, что загружен датасет с топ-пресетами

        setup_model_research_ui(self)

        self.project_root = get_project_root()

        self.btn_select.clicked.connect(self.select_folder)
        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_save.clicked.connect(self.save_dataset)
        self.btn_first.clicked.connect(self.show_first)
        self.btn_last.clicked.connect(self.show_last)
        self.btn_all.clicked.connect(self.show_all)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_random.clicked.connect(self.show_random)
        self.btn_save_sample.clicked.connect(self.save_current_sample)

        self.btn_filter_zero_true.clicked.connect(self.filter_zero_true)
        self.btn_filter_overseg.clicked.connect(self.filter_oversegmentation)
        self.btn_filter_duplicates.clicked.connect(self.filter_duplicates)
        self.btn_filter_invalid_params.clicked.connect(self.filter_invalid_params)
        self.btn_filter_anomaly_count.clicked.connect(self.filter_anomaly_count)
        self.btn_filter_full_image.clicked.connect(self.filter_full_image_object)

        # Новая кнопка удаления строк без истинных объектов
        self.btn_remove_no_true.clicked.connect(self.filter_no_true_objects)

        # Кнопка meanAP
        self.btn_compute_f1.clicked.connect(self.compute_meanap)
        self.btn_sample_n.clicked.connect(self.sample_n_rows)
        self.btn_filter_f1.clicked.connect(self.filter_by_meanf1_threshold)
        self.btn_aggregate.clicked.connect(self.aggregate_by_params)
        self.btn_cluster.clicked.connect(self.add_cluster_labels)
        self.btn_save_unique_params.clicked.connect(self.save_unique_params)
        self.btn_cluster_co.clicked.connect(self.cluster_open_close)
        self.btn_optimize_invert.clicked.connect(self.optimize_invert)
        self.btn_global_top5.clicked.connect(self.run_global_top5)
        self.btn_load_top_methods.clicked.connect(self.load_top_methods)

        # Кнопка удаления идеальных строк (только для топ-пресетов)
        self.btn_remove_perfect.clicked.connect(self.filter_perfect_rows)

    def log_msg(self, msg):
        self.log_widget.log(msg)
        print(msg)

    def update_row_count_display(self):
        count = len(self.df_full) if self.df_full is not None else 0
        self.lbl_row_count.setText(f"Строк в датасете: {count}")

    def _update_f1_filter_button_state(self):
        if self.df_full is not None and 'mean_f1' in self.df_full.columns:
            self.btn_filter_f1.setEnabled(True)
        else:
            self.btn_filter_f1.setEnabled(False)

    def _update_aggregate_button_state(self):
        if self.df_full is not None and 'mean_f1' in self.df_full.columns and not self.is_aggregated:
            self.btn_aggregate.setEnabled(True)
        else:
            self.btn_aggregate.setEnabled(False)

    def _update_buttons_by_aggregated_flag(self):
        if self.is_aggregated:
            self.btn_filter_full_image.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_cluster.setEnabled(False)
        else:
            if self.df_full is not None and 'objects' in self.df_full.columns:
                self.btn_filter_full_image.setEnabled(True)
                self.btn_compute_f1.setEnabled(True)
            self._update_aggregate_button_state()
            if self.df_full is not None:
                self.btn_cluster.setEnabled(True)

    def _apply_prepare_operation(self, operation_func, *args, **kwargs):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return False

        try:
            new_df, msg = operation_func(self.df_full, *args, **kwargs)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
                return True
            else:
                self.log_msg(msg)
                return False
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
            return False

    # ---------- Фильтры и операции ----------
    def filter_full_image_object(self):
        self._apply_prepare_operation(filter_full_image_object)

    def filter_invalid_params(self):
        self._apply_prepare_operation(filter_invalid_params)

    def filter_anomaly_count(self):
        self._apply_prepare_operation(filter_anomaly_count, max_objects=15)

    def filter_zero_true(self):
        self._apply_prepare_operation(filter_zero_true)

    def filter_oversegmentation(self):
        self._apply_prepare_operation(filter_oversegmentation, ratio=2.0)

    def filter_duplicates(self):
        self._apply_prepare_operation(filter_duplicates)

    def filter_no_true_objects(self):
        """Удаляет строки, в которых нет истинных объектов (num_true_objects == 0)"""
        self._apply_prepare_operation(filter_no_true_objects)

    # ---------- Расчёт meanAP ----------
    def compute_meanap(self):
        if self.is_aggregated:
            QMessageBox.warning(self, "Операция недоступна", "Для агрегированного датасета пересчёт метрик невозможен.")
            return
        self.btn_compute_f1.setEnabled(False)
        try:
            new_df, msg = add_meanap_metrics(self.df_full, iou_thresholds=DEFAULT_IOU_THRESHOLDS)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg("Не удалось рассчитать meanAP")
        except Exception as e:
            self.log_msg(f"Ошибка при расчёте meanAP: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_compute_f1.setEnabled(True)

    # Фильтрация по mean_f1
    def filter_by_meanf1_threshold(self):
        threshold = self.f1_threshold_spin.value()
        if self.df_full is None:
            return
        if 'mean_f1' not in self.df_full.columns:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните 'Расчет meanAP'.")
            return
        before = len(self.df_full)
        self.df_full = self.df_full[self.df_full['mean_f1'] >= threshold].reset_index(drop=True)
        after = len(self.df_full)
        self.log_msg(f"Фильтр по mean_f1 (порог {threshold:.2f}): удалено {before - after} строк. Осталось {after}.")
        self.update_row_count_display()
        self.display_dataframe(self.df_full.head(100))
        self._update_f1_filter_button_state()
        self._update_aggregate_button_state()
        self._update_buttons_by_aggregated_flag()

    def sample_n_rows(self):
        n = self.sample_spin.value()
        self._apply_prepare_operation(sample_rows, n=n)

    def aggregate_by_params(self):
        if self.is_aggregated:
            QMessageBox.warning(self, "Уже агрегирован", "Датасет уже агрегирован.")
            return
        top_k = self.top_k_spin.value()
        success = self._apply_prepare_operation(aggregate_by_params, top_k=top_k)
        if success:
            self.is_aggregated = True
            self._update_buttons_by_aggregated_flag()

    def add_cluster_labels(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return

        result_dir = os.path.join(self.project_root, "result")
        os.makedirs(result_dir, exist_ok=True)

        try:
            self.btn_cluster.setEnabled(False)
            new_df, msg = add_cluster_labels(self.df_full, result_dir)
            if new_df is not None and not new_df.empty:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                QMessageBox.information(self, "Готово", f"Кластеризация завершена. Добавлена колонка 'cluster'.\n{msg}")
            else:
                self.log_msg(msg)
                QMessageBox.warning(self, "Ошибка", msg)
        except Exception as e:
            self.log_msg(f"Ошибка кластеризации: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_cluster.setEnabled(True)

    def save_unique_params(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return

        result_dir = os.path.join(self.project_root, "result")
        os.makedirs(result_dir, exist_ok=True)

        try:
            self.btn_save_unique_params.setEnabled(False)
            _, msg = save_unique_params(self.df_full, result_dir)
            self.log_msg(msg)
            QMessageBox.information(self, "Готово", f"Уникальные параметры сохранены.\n{msg}")
        except Exception as e:
            self.log_msg(f"Ошибка сохранения: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_save_unique_params.setEnabled(True)

    def optimize_invert(self):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return
        if 'mean_f1' not in self.df_full.columns:
            QMessageBox.warning(self, "Нет метрики", "Сначала выполните 'Расчет meanAP'.")
            return
        try:
            new_df, msg = optimize_invert(self.df_full)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg(msg)
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def cluster_open_close(self):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return
        if 'cluster' not in self.df_full.columns:
            QMessageBox.warning(self, "Нет кластеров", "Сначала выполните 'Кластеризация параметров' (добавьте колонку 'cluster').")
            return
        n_clusters = 3
        try:
            new_df, msg = cluster_open_close(self.df_full, n_clusters=n_clusters)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg(msg)
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- Работа с топ-пресетами ----------
    def load_top_methods(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить топ-пресеты по методам", "", "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)
            required = ['method', 'rank', 'params', 'invert', 'close_factor', 'open_factor', 'actual_f1', 'pred_f1', 'cluster', 'cluster_co']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Файл не содержит необходимых колонок: {missing}")
            self.df_full = df
            self.is_top_methods = True
            self.is_aggregated = False
            self.log_msg(f"Загружен датасет топ-пресетов: {len(df)} строк.")
            self.display_dataframe(self.df_full.head(100))
            self._update_buttons_state()
            self.update_row_count_display()
        except Exception as e:
            self.log_msg(f"Ошибка загрузки: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def filter_perfect_rows(self):
        """Удаляет строки с actual_f1 >= заданного порога (только для датасета топ-пресетов)."""
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Датасет пуст.")
            return
        if not self.is_top_methods:
            QMessageBox.warning(self, "Недоступно", "Эта операция доступна только для датасета топ-пресетов.")
            return
        if 'actual_f1' not in self.df_full.columns:
            QMessageBox.warning(self, "Нет колонки", "В датасете отсутствует колонка actual_f1.")
            return

        threshold = self.perfect_threshold_spin.value()
        before = len(self.df_full)
        self.df_full = self.df_full[self.df_full['actual_f1'] < threshold].reset_index(drop=True)
        after = len(self.df_full)
        self.log_msg(f"Удалено строк с actual_f1 >= {threshold}: {before - after}. Осталось {after}.")
        self.update_row_count_display()
        self.display_dataframe(self.df_full.head(100))

    # ---------- Загрузка данных ----------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.current_folder = folder
        self.log_msg(f"Selected folder: {folder}")

        report_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_full_report.txt")]
        if not report_files:
            self.log_msg("No report files (*_full_report.txt) found.")
            return

        self.log_msg(f"Found {len(report_files)} report files.")
        self.df_full = self.parse_reports(report_files)
        if self.df_full is None or self.df_full.empty:
            self.log_msg("No data extracted.")
            return

        self.df_full['num_true_objects'] = self.df_full['true_objects'].apply(
            lambda s: len(literal_eval(s)) if isinstance(s, str) and s.startswith('[') else 0
        )
        self.df_full['num_objects'] = pd.to_numeric(self.df_full['num_objects'], errors='coerce').fillna(0).astype(int)

        self.is_aggregated = False
        self.display_dataframe(self.df_full.head(100))
        self._enable_buttons()
        self._update_f1_filter_button_state()
        self._update_aggregate_button_state()
        self._update_buttons_by_aggregated_flag()
        self.update_row_count_display()
        self.log_msg(f"Loaded {len(self.df_full)} rows.")
        self.is_top_methods = False
        self._update_buttons_state()

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Dataset", "", "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)

            required = ['objects', 'true_objects', 'method', 'params', 'invert', 'close_factor', 'open_factor']
            missing = [c for c in required if c not in df.columns]
            if missing:
                self.log_msg(f"Warning: missing columns: {missing}")

            self.df_full = df
            self.log_msg(f"Loaded {len(df)} rows from {file_path}")

            if 'num_true_objects' not in self.df_full.columns:
                self.df_full['num_true_objects'] = self.df_full['true_objects'].apply(
                    lambda s: len(literal_eval(s)) if isinstance(s, str) and s.startswith('[') else 0
                )
            if 'num_objects' not in self.df_full.columns and 'objects' in self.df_full.columns:
                self.df_full['num_objects'] = self.df_full['objects'].apply(
                    lambda s: len(literal_eval(s)) if isinstance(s, str) and s.startswith('[') else 0
                )

            self.is_aggregated = False
            self.display_dataframe(self.df_full.head(100))
            self._enable_buttons()
            self._update_f1_filter_button_state()
            self._update_aggregate_button_state()
            self._update_buttons_by_aggregated_flag()
            self.update_row_count_display()
        except Exception as e:
            self.log_msg(f"Error loading dataset: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")
        self.is_top_methods = False
        self._update_buttons_state()

    def _enable_buttons(self):
        self.btn_save.setEnabled(True)
        self.btn_first.setEnabled(True)
        self.btn_last.setEnabled(True)
        self.btn_all.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_random.setEnabled(True)
        self.btn_save_sample.setEnabled(True)
        self.btn_filter_zero_true.setEnabled(True)
        self.btn_filter_overseg.setEnabled(True)
        self.btn_filter_duplicates.setEnabled(True)
        self.btn_filter_invalid_params.setEnabled(True)
        self.btn_filter_anomaly_count.setEnabled(True)
        self.btn_filter_full_image.setEnabled(True)
        self.btn_remove_no_true.setEnabled(True)   # новая кнопка
        self.btn_sample_n.setEnabled(True)
        self.btn_cluster.setEnabled(True)
        self.btn_save_unique_params.setEnabled(True)
        self.btn_cluster_co.setEnabled(True)
        self.btn_optimize_invert.setEnabled(True)
        self.btn_global_top5.setEnabled(True)
        self._update_buttons_by_aggregated_flag()

    # ---------- Парсинг отчётов ----------
    def parse_reports(self, report_files):
        rows = []
        for file_path in report_files:
            self.log_msg(f"Processing {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                blocks = self.split_blocks(content)
                true_objects = self.load_true_annotations(file_path)
                for block in blocks:
                    row = self.parse_block(block, file_path)
                    if row:
                        row['true_objects'] = str(true_objects) if true_objects else ''
                        rows.append(row)
            except Exception as e:
                self.log_msg(f"Error processing {file_path}: {e}")
        if not rows:
            return None
        return pd.DataFrame(rows)

    def load_true_annotations(self, report_path):
        base = os.path.basename(report_path)
        if base.endswith('_full_report.txt'):
            base_name = base[:-16]
        else:
            base_name = os.path.splitext(base)[0]
        annotations_path = os.path.join(os.path.dirname(report_path), base_name + '.txt')
        if not os.path.exists(annotations_path):
            self.log_msg(f"  Annotation file not found: {annotations_path}")
            return None
        objects = load_annotations(annotations_path, img_w=1, img_h=1)
        return objects if objects else None

    def split_blocks(self, content):
        pattern = r'(===.*?===)(.*?)(?=\n===|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        return [(h.strip(), b.strip()) for h, b in matches]

    def parse_block(self, block, file_path):
        header, body = block
        try:
            header = header.strip('=')
            parts = header.split('|')
            left_part = parts[0].strip()
            right_part = parts[1].strip() if len(parts) > 1 else ''

            method_match = re.search(r'Метод:\s*([^,]+)', left_part)
            method = method_match.group(1).strip() if method_match else ''

            if 'инверсия=' in left_part:
                params_str = left_part.split('инверсия=')[0].replace(f"Метод: {method},", '').strip().rstrip(',')
            else:
                params_str = ''

            invert_match = re.search(r'инверсия=([^,]+)', left_part)
            invert = invert_match.group(1).strip() if invert_match else ''
            invert_bool = invert == 'Да'

            close_match = re.search(r'close=([\d.]+)', left_part)
            close_factor = float(close_match.group(1)) if close_match else 0.0
            open_match = re.search(r'open=([\d.]+)', left_part)
            open_factor = float(open_match.group(1)) if open_match else 0.0

            num_objects = 0
            if 'Количество объектов:' in right_part:
                num_match = re.search(r'Количество объектов:\s*(\d+)', right_part)
                if num_match:
                    num_objects = int(num_match.group(1))

            objects_list = []
            if body.strip():
                for line in body.strip().split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            cls = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            objects_list.append((cls, cx, cy, w, h))
                        except:
                            pass

            return {
                'image_path': file_path,
                'method': method,
                'params': params_str,
                'invert': invert_bool,
                'close_factor': close_factor,
                'open_factor': open_factor,
                'num_objects': num_objects,
                'objects': str(objects_list)
            }
        except Exception as e:
            self.log_msg(f"Error parsing header: {e}\nHeader: {header}")
            return None

    # ---------- Отображение и сохранение ----------
    def display_dataframe(self, df):
        if df is None or df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.current_display_df = None
            return

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)
        self.current_display_df = df

        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                val = row[col]
                if col in ('objects', 'true_objects') and isinstance(val, str) and len(val) > 100:
                    val = val[:100] + '...'
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def save_dataset(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset", "", "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                self.df_full.to_csv(file_path, index=False, encoding='utf-8')
            elif file_path.endswith('.pkl'):
                self.df_full.to_pickle(file_path)
            else:
                self.df_full.to_csv(file_path, index=False, encoding='utf-8')
            self.log_msg(f"Dataset saved to {file_path}")
            QMessageBox.information(self, "Success", f"Saved to {file_path}")
        except Exception as e:
            self.log_msg(f"Error saving: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def show_first(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full.head(100))

    def show_last(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full.tail(100))

    def show_all(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full)

    def show_random(self):
        if self.df_full is not None and len(self.df_full) > 0:
            sample = self.df_full.sample(n=min(100, len(self.df_full)))
            self.display_dataframe(sample)
        else:
            QMessageBox.warning(self, "No Data", "Нет данных для случайной выборки.")

    def save_current_sample(self):
        if self.current_display_df is None or self.current_display_df.empty:
            QMessageBox.warning(self, "No Data", "Нет данных для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current Sample", "", "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                self.current_display_df.to_csv(file_path, index=False, encoding='utf-8')
            elif file_path.endswith('.pkl'):
                self.current_display_df.to_pickle(file_path)
            else:
                self.current_display_df.to_csv(file_path, index=False, encoding='utf-8')
            self.log_msg(f"Sample saved to {file_path}")
            QMessageBox.information(self, "Success", f"Saved {len(self.current_display_df)} rows.")
        except Exception as e:
            self.log_msg(f"Error saving sample: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    # ---------- Анализ ----------
    def run_analysis(self):
        if self.is_top_methods:
            QMessageBox.warning(self, "Недоступно", "Анализ модели недоступен для датасета топ-пресетов.")
            return
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return

        required = ['method', 'params', 'invert', 'close_factor', 'open_factor']
        missing = [c for c in required if c not in self.df_full.columns]
        if missing:
            QMessageBox.critical(self, "Отсутствуют колонки", f"Отсутствуют необходимые колонки: {missing}")
            return

        if 'cluster' not in self.df_full.columns:
            QMessageBox.critical(self, "Ошибка", "Колонка 'cluster' отсутствует. Сначала выполните 'Кластеризация параметров'.")
            return
        if 'cluster_co' not in self.df_full.columns:
            QMessageBox.critical(self, "Ошибка", "Колонка 'cluster_co' отсутствует. Сначала выполните 'Кластеризация open/close'.")
            return

        has_mean_f1 = 'mean_f1' in self.df_full.columns
        has_avg_iou_tp = 'avg_iou_tp' in self.df_full.columns
        has_objects = 'objects' in self.df_full.columns and 'true_objects' in self.df_full.columns

        if not has_mean_f1 and not has_objects:
            QMessageBox.critical(self, "Нет данных", "Нет ни объектов, ни mean_f1. Невозможно выполнить анализ.")
            return

        self.log_msg("Запуск анализа модели (для каждого метода, с использованием кластеров)...")
        self.btn_analyze.setEnabled(False)

        result_dir = os.path.join(self.project_root, "result")
        os.makedirs(result_dir, exist_ok=True)

        use_diversity = self.chk_use_diversity.isChecked()
        use_iou_penalty = self.chk_iou_penalty.isChecked()
        skip_metrics = has_mean_f1

        try:
            top5 = analyze(
                self.df_full,
                output_dir=result_dir,
                skip_metrics=skip_metrics,
                use_diversity=use_diversity,
                use_iou_penalty=use_iou_penalty,
                penalty_alpha=0.5,
                per_method=True,
                cluster_col='cluster',
                cluster_co_col='cluster_co'
            )
            if top5 is not None and not top5.empty:
                self.results_widget.display_results(top5)
                self.log_msg("Анализ завершён.")
            else:
                self.log_msg("Анализ не вернул данных.")
        except Exception as e:
            self.log_msg(f"Ошибка анализа: {e}")
            QMessageBox.critical(self, "Ошибка анализа", str(e))
        finally:
            self.btn_analyze.setEnabled(True)

    def run_global_top5(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return
        if 'cluster' not in self.df_full.columns or 'cluster_co' not in self.df_full.columns:
            QMessageBox.critical(self, "Ошибка", "Колонки 'cluster' и/или 'cluster_co' отсутствуют.")
            return
        if 'actual_f1' not in self.df_full.columns and 'mean_f1' not in self.df_full.columns:
            QMessageBox.warning(self, "Нет метрики", "Нет колонок actual_f1 или mean_f1.")
            return

        score_col = 'actual_f1' if 'actual_f1' in self.df_full.columns else 'mean_f1'
        try:
            top5 = global_top_diverse(self.df_full, k=10, score_col=score_col)
            if top5 is not None and not top5.empty:
                self.results_widget.display_results(top5)
                self.log_msg(f"Глобальный топ-10 (разнообразные) выбран. Использована метрика: {score_col}")
            else:
                self.log_msg("Не удалось выбрать топ-10.")
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def save_dataframe(self, df, dialog_title="Сохранить датасет"):
        if df is None or df.empty:
            QMessageBox.warning(self, "Нет данных", "Нет данных для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, "", "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8')
            elif file_path.endswith('.pkl'):
                df.to_pickle(file_path)
            else:
                df.to_csv(file_path, index=False, encoding='utf-8')
            self.log_msg(f"Сохранено в {file_path}")
            QMessageBox.information(self, "Успех", f"Сохранено {len(df)} строк.")
        except Exception as e:
            self.log_msg(f"Ошибка сохранения: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- Добавление в пресеты ----------
    def add_to_presets(self, top5):
        if top5 is None or top5.empty:
            QMessageBox.warning(self, "Нет данных", "Нет данных для добавления в пресеты.")
            return
        preset_mgr = PresetManager(main_window=None)
        added = []
        for _, row in top5.iterrows():
            method = row['method']
            params_str = row['params']
            invert = row['invert']
            close = row['close_factor']
            open_f = row['open_factor']
            base_name = f"{method} {params_str} inv={invert} c={close} o={open_f}"
            name = base_name
            counter = 1
            while name in preset_mgr.presets:
                name = f"{base_name} ({counter})"
                counter += 1
            preset_params = self._extract_preset_params(method, params_str)
            preset_dict = {
                "method": method,
                "params": preset_params,
                "invert": invert,
                "close_factor": close,
                "open_factor": open_f,
                "draw_mode": "Contours (simple)",
                "use_hull": False
            }
            if preset_mgr.add_preset_dict(name, preset_dict):
                added.append(name)
        if added:
            self.log_msg(f"Добавлены пресеты: {', '.join(added)}")
            QMessageBox.information(self, "Успех", f"Добавлено {len(added)} пресетов.")

    def _extract_preset_params(self, method, params_str):
        if method == "Simple Threshold":
            try:
                thresh = int(params_str.split('=')[1])
            except:
                thresh = 127
            return {"threshold": thresh}
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            try:
                parts = params_str.split(',')
                ws = int(parts[0].split('=')[1])
                c = int(parts[1].split('=')[1])
            except:
                ws, c = 25, 3
            return {"window": ws, "c": c}
        elif method == "Niblack":
            try:
                parts = params_str.split(',')
                ws = int(parts[0].split('=')[1])
                k = float(parts[1].split('=')[1])
            except:
                ws, k = 25, 0.2
            return {"window": ws, "k": k}
        elif method == "Sauvola":
            try:
                parts = params_str.split(',')
                ws = int(parts[0].split('=')[1])
                k = float(parts[1].split('=')[1])
                r = int(parts[2].split('=')[1]) if len(parts) > 2 else 128
            except:
                ws, k, r = 25, 0.2, 128
            return {"window": ws, "k": k, "r": r}
        elif method == "ISODATA":
            try:
                init = int(params_str.split('=')[1])
            except:
                init = 128
            return {"init": init}
        elif method == "Background Symmetry":
            try:
                excess = float(params_str.split('=')[1])
            except:
                excess = 0.2
            return {"excess": excess}
        elif method == "Row Adaptive":
            try:
                parts = params_str.split(',')
                ws = int(parts[0].split('=')[1])
                k = float(parts[1].split('=')[1])
            except:
                ws, k = 50, 0.5
            return {"window": ws, "k": k}
        else:
            return {}

    def _update_buttons_state(self):
        """Обновляет доступность кнопок в зависимости от типа загруженных данных."""
        if self.df_full is None or self.df_full.empty:
            self.btn_save.setEnabled(False)
            self.btn_first.setEnabled(False)
            self.btn_last.setEnabled(False)
            self.btn_all.setEnabled(False)
            self.btn_random.setEnabled(False)
            self.btn_save_sample.setEnabled(False)
            self.btn_filter_zero_true.setEnabled(False)
            self.btn_filter_overseg.setEnabled(False)
            self.btn_filter_duplicates.setEnabled(False)
            self.btn_filter_invalid_params.setEnabled(False)
            self.btn_filter_anomaly_count.setEnabled(False)
            self.btn_filter_full_image.setEnabled(False)
            self.btn_remove_no_true.setEnabled(False)
            self.btn_sample_n.setEnabled(False)
            self.btn_cluster.setEnabled(False)
            self.btn_save_unique_params.setEnabled(False)
            self.btn_cluster_co.setEnabled(False)
            self.btn_optimize_invert.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_filter_f1.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_global_top5.setEnabled(False)
            self.btn_remove_perfect.setEnabled(False)   # для топ-пресетов
            return

        if self.is_top_methods:
            self.btn_save.setEnabled(True)
            self.btn_first.setEnabled(True)
            self.btn_last.setEnabled(True)
            self.btn_all.setEnabled(True)
            self.btn_random.setEnabled(True)
            self.btn_save_sample.setEnabled(True)
            self.btn_filter_zero_true.setEnabled(False)
            self.btn_filter_overseg.setEnabled(False)
            self.btn_filter_duplicates.setEnabled(False)
            self.btn_filter_invalid_params.setEnabled(False)
            self.btn_filter_anomaly_count.setEnabled(False)
            self.btn_filter_full_image.setEnabled(False)
            self.btn_remove_no_true.setEnabled(False)
            self.btn_sample_n.setEnabled(False)
            self.btn_cluster.setEnabled(False)
            self.btn_save_unique_params.setEnabled(False)
            self.btn_cluster_co.setEnabled(False)
            self.btn_optimize_invert.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_filter_f1.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_global_top5.setEnabled(True)
            self.btn_remove_perfect.setEnabled(True)   # активна для топ-пресетов
        else:
            self._enable_buttons()
            self.btn_remove_perfect.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelResearchApp()
    window.show()
    sys.exit(app.exec_())