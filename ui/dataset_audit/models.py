"""Dataset analysis: models Qt controller."""

import os
import subprocess
import sys as _sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from ui.dataset_audit.lifecycle import stale
from ui.dataset_audit.workers import YoloValidationWorker
from utils_new.dataset_audit.objects import load_pair_annotations
from utils_new.dataset_files_new import read_image, read_index_mask


class DatasetModelController:
    def load_audit_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Модель YOLO",
            "",
            "YOLO weights (*.pt *.onnx *.engine);;Все файлы (*)",
        )
        if not path:
            return
        self._audit_model_path = path
        self.audit_model_edit.setText(path)
        self._audit_model_cache = None
        self._audit_model_cache_path = None
        self._audit_log(f"Модель выбрана: {path}", tag="AUDIT", level="OK")

    def run_loss_inspection(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных", "Нет пар для анализа.")
            return
        if not self._audit_model_path:
            QMessageBox.warning(self, "Модель не выбрана", "Загрузите YOLO-модель.")
            return
        if self._audit_thread and self._audit_thread.isRunning():
            QMessageBox.information(self, "Идёт аудит", "Дождитесь завершения.")
            return

        from ui.dataset_audit.workers import LossInspectionWorker

        iou_thr = float(self.audit_iou_spin.value())
        conf_thr = float(self.audit_conf_spin.value())
        imgsz = int(self.audit_imgsz_spin.value())

        self._audit_log(
            f"Старт аудита: model={self._audit_model_path}, "
            f"iou={iou_thr}, conf={conf_thr}, imgsz={imgsz}, "
            f"device={self._device}, pairs={len(pairs)}",
            tag="AUDIT",
        )

        self.audit_table.setRowCount(0)
        self.audit_summary_label.setText("Аудит запущен…")
        self.btn_run_audit.setEnabled(False)
        self.btn_stop_audit.setEnabled(True)

        self._audit_thread = LossInspectionWorker(
            pairs,
            self._audit_model_path,
            dataset_type=self.dataset_type,
            iou_threshold=iou_thr,
            conf_threshold=conf_thr,
            imgsz=imgsz,
            input_format=self.input_format,
            device=self._device,
            mask_class_map=dict(self.mask_class_remap)
            if self.dataset_type == 1
            else None,
        )
        self._audit_thread.log_signal.connect(self.log)
        self._audit_thread.log_signal.connect(
            lambda m: print(f"[AUDIT][THREAD] {m}", flush=True)
        )
        self._audit_thread.progress.connect(self._on_audit_progress)
        self._audit_thread.finished_signal.connect(self._on_audit_finished)
        self._start_audit_worker(self._audit_thread)

    def stop_loss_inspection(self):
        if self._audit_thread and self._audit_thread.isRunning():
            self._audit_thread.cancel()
            self._audit_log("Аудит: запрошена остановка.", tag="AUDIT", level="WARN")

    def _on_audit_progress(self, c, t):
        if stale(self):
            return
        self.audit_summary_label.setText(f"⏳ Аудит: {c}/{t}…")

    def _on_audit_finished(self, result):
        if stale(self, result):
            return
        self.btn_run_audit.setEnabled(True)
        self.btn_stop_audit.setEnabled(False)
        err = result.get("error")
        if err:
            self.audit_summary_label.setText(f"❌ {err}")
            self._audit_log(f"Аудит ошибка: {err}", tag="AUDIT", level="ERROR")
            return

        if result.get("cancelled"):
            self.audit_summary_label.setText("Аудит отменён. Результат не обновлён.")
            return

        suspicious = result.get("suspicious", [])
        all_results = result.get("all_results", suspicious)
        processed = result.get("processed", 0)
        total = result.get("total", 0)
        errors = result.get("errors", 0)

        self._last_audit_results = suspicious
        self._last_audit_all_results = all_results
        self._last_audit_meta = {
            "model": self._audit_model_path,
            "iou_thr": float(self.audit_iou_spin.value()),
            "conf_thr": float(self.audit_conf_spin.value()),
            "imgsz": int(self.audit_imgsz_spin.value()),
            "processed": processed,
            "total": total,
            "errors": errors,
            "suspicious_count": len(suspicious),
            "audited_count": len(all_results),
        }

        self._rebuild_audit_table()

        self._audit_log(
            f"Аудит завершён: processed={processed}/{total}, "
            f"errors={errors}, аудировано={len(all_results)}, "
            f"подозрительных={len(suspicious)}",
            tag="AUDIT",
            level="OK",
        )

        for item in suspicious[:5]:
            self._audit_log(
                f"  {os.path.basename(item['img_path'])}: "
                f"gt={item['gt_count']}, pred={item['pred_count']}, "
                f"iou={item['mean_iou']:.3f}",
                tag="AUDIT",
            )

    def _open_audit_row(self, row, col):
        items = getattr(self, "_last_audit_display", None) or []
        if row < 0 or row >= len(items):
            return
        self._open_audit_comparison(items[row])

    def _rebuild_audit_table(self):
        """Перестраивает таблицу аудита согласно выбранному фильтру."""
        mode = "suspicious"
        try:
            mode = self.audit_filter_combo.currentData() or "suspicious"
        except Exception:
            pass

        if mode == "all":
            items = self._last_audit_all_results or []
        else:
            items = self._last_audit_results or []

        self._last_audit_display = list(items)

        self.audit_table.setRowCount(len(items))
        for row, item in enumerate(items):
            is_susp = bool(item.get("is_suspicious", True))
            name = os.path.basename(item.get("img_path", ""))
            gt = item.get("gt_count", 0)
            pred = item.get("pred_count", 0)
            miou = item.get("mean_iou", 0.0)
            min_iou = item.get("min_iou", 0.0)
            reason = item.get("reason", "—")

            tooltip = (
                f"{name}\n"
                f"Причина: {reason}\n"
                f"GT: {gt}, Pred: {pred}\n"
                f"Mean IoU: {miou:.3f}, Min IoU: {min_iou:.3f}\n"
                f"Подозрительная: {'ДА' if is_susp else 'нет'}"
            )

            cells = (name, str(gt), str(pred), f"{miou:.3f}")
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                it.setToolTip(tooltip)
                if is_susp:
                    it.setBackground(QColor(255, 220, 220))
                self.audit_table.setItem(row, c, it)

        meta = self._last_audit_meta or {}
        if meta:
            processed = meta.get("processed", 0)
            total = meta.get("total", 0)
            errors = meta.get("errors", 0)
            audited = len(self._last_audit_all_results or [])
            susp = len(self._last_audit_results or [])
            filter_label = (
                "все аудированные" if mode == "all" else "только подозрительные"
            )
            self.audit_summary_label.setText(
                f"Обработано: {processed}/{total}. "
                f"Аудировано: {audited}. "
                f"Подозрительных: {susp}. Ошибок: {errors}. "
                f"Показано: {len(items)} "
                f"(фильтр: {filter_label}). "
                f"Двойной клик — открыть двойное окно сравнения "
                f"(GT ↔ Pred)."
            )

    def _get_audit_model(self):
        path = self._audit_model_path
        if not path:
            return None
        if self._audit_model_cache is not None and self._audit_model_cache_path == path:
            return self._audit_model_cache
        try:
            from runtime_policy import configure_ultralytics

            configure_ultralytics()
            from runtime_torch import require_torch

            require_torch()
            from ultralytics import YOLO

            m = YOLO(path)
            if self._device.startswith("cuda"):
                try:
                    m.to(self._device)
                except Exception as e:
                    self._audit_log(
                        f"[AUDIT] Не удалось переместить модель на "
                        f"{self._device} ({e}) — остаёмся на CPU",
                        tag="AUDIT",
                        level="WARN",
                    )
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
            self._audit_model_cache = m
            self._audit_model_cache_path = path
            self._audit_log(
                f"[AUDIT] Модель загружена в кэш: "
                f"{os.path.basename(path)} (device: {self._device})",
                tag="AUDIT",
            )
            return self._audit_model_cache
        except Exception as e:
            self._audit_log(
                f"[AUDIT] Не удалось загрузить модель: {e}", tag="AUDIT", level="ERROR"
            )
            return None

    def _open_folder_in_explorer(self, path):
        if not path:
            QMessageBox.information(
                self, "Нет данных", "Сначала выполните действие (экспорт / отчёт)."
            )
            return
        if not os.path.isdir(path):
            QMessageBox.warning(
                self, "Папка не найдена", f"Путь не существует:\n{path}"
            )
            return
        try:
            if _sys.platform.startswith("win"):
                os.startfile(path)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            self._audit_log(f"Открываю папку: {path}", tag="FS", level="OK")
        except Exception as e:
            QMessageBox.warning(self, "Не удалось открыть папку", f"{path}\n{e}")

    def open_last_export_folder(self):
        self._open_folder_in_explorer(self._last_export_dir)

    def open_report_folder(self):
        self._open_folder_in_explorer(self._last_report_dir)

    def _open_audit_comparison(self, item):
        img_path = item["img_path"]
        lbl_path = item.get("lbl_path")
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Нет файла", img_path)
            return

        try:
            base = read_image(img_path)
        except OSError:
            base = None
        if base is None:
            QMessageBox.warning(self, "Не удалось прочитать", img_path)
            return

        results = None
        model = self._get_audit_model()
        if model is not None:
            try:
                results = model(
                    img_path,
                    conf=float(self.audit_conf_spin.value()),
                    imgsz=int(self.audit_imgsz_spin.value()),
                    device=self._device,
                    verbose=False,
                )
            except Exception as e:
                self._audit_log(
                    f"[AUDIT] Ошибка инференса: {e}", tag="AUDIT", level="ERROR"
                )

        h, w = base.shape[:2]
        img_gt = base.copy()
        img_pred = base.copy()

        try:
            if self.dataset_type == 0 and lbl_path and os.path.exists(lbl_path):
                anns = load_pair_annotations(
                    img_path, lbl_path, format_hint=self.input_format
                )
                for ann in anns:
                    if not ann:
                        continue
                    if ann[0] == "detect":
                        _, cls, cx, cy, bw, bh = ann
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(
                            img_gt,
                            f"c{cls}",
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 0),
                            1,
                        )
                    elif ann[0] in ("segment", "obb"):
                        pts = ann[2]
                        if not pts or len(pts) < 6:
                            continue
                        arr = np.array(
                            [
                                [int(pts[i] * w), int(pts[i + 1] * h)]
                                for i in range(0, len(pts), 2)
                            ],
                            dtype=np.int32,
                        )
                        cv2.polylines(img_gt, [arr], True, (0, 200, 0), 2)
            elif self.dataset_type == 1 and lbl_path and os.path.exists(lbl_path):
                mask = read_index_mask(lbl_path)
                if mask is not None and mask.ndim == 2:
                    overlay = np.zeros_like(img_gt)
                    overlay[mask > 0] = (0, 200, 0)
                    img_gt = cv2.addWeighted(img_gt, 1.0, overlay, 0.35, 0)
                    contours, _ = cv2.findContours(
                        (mask > 0).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(img_gt, contours, -1, (0, 200, 0), 2)
        except Exception as e:
            self._audit_log(f"[AUDIT] Ошибка отрисовки GT: {e}", tag="AUDIT")

        try:
            if results:
                r = results[0]
                if getattr(r, "masks", None) is not None:
                    try:
                        m_arr = r.masks.data.cpu().numpy()
                        for m in m_arr:
                            m_resized = cv2.resize(
                                m, (w, h), interpolation=cv2.INTER_NEAREST
                            )
                            overlay = np.zeros_like(img_pred)
                            overlay[m_resized > 0.5] = (0, 0, 220)
                            img_pred = cv2.addWeighted(img_pred, 1.0, overlay, 0.35, 0)
                    except Exception:
                        pass

                boxes_obj = getattr(r, "boxes", None)
                obb_obj = getattr(r, "obb", None)

                draw_boxes = None
                if boxes_obj is not None and len(boxes_obj):
                    draw_boxes = boxes_obj
                elif obb_obj is not None and len(obb_obj):
                    draw_boxes = obb_obj

                if draw_boxes is not None:
                    for box in draw_boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 220), 2)
                        cv2.putText(
                            img_pred,
                            f"c{cls} {conf:.2f}",
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 220),
                            1,
                        )
        except Exception as e:
            self._audit_log(f"[AUDIT] Ошибка отрисовки Pred: {e}", tag="AUDIT")

        _, buf_gt = cv2.imencode(".png", img_gt)
        _, buf_pr = cv2.imencode(".png", img_pred)
        pix_gt = QPixmap()
        pix_gt.loadFromData(buf_gt.tobytes(), "PNG")
        pix_pr = QPixmap()
        pix_pr.loadFromData(buf_pr.tobytes(), "PNG")

        dlg = QDialog(self)
        dlg.setWindowTitle(
            f"Аудит: {os.path.basename(img_path)} | "
            f"GT {item['gt_count']} / Pred {item['pred_count']} | "
            f"IoU mean={item['mean_iou']:.3f} "
            f"min={item['min_iou']:.3f}"
        )
        dlg.resize(1500, 800)
        v = QVBoxLayout(dlg)

        info = QLabel(
            f"<b>Файл:</b> {img_path}<br>"
            f"<b>Метка:</b> {lbl_path or '—'}<br>"
            f"<b>Причина:</b> {item.get('reason', '—')}<br>"
            f"<b>GT:</b> {item['gt_count']}, "
            f"<b>Pred:</b> {item['pred_count']}, "
            f"<b>Mean IoU:</b> {item['mean_iou']:.3f}, "
            f"<b>Min IoU:</b> {item['min_iou']:.3f}, "
            f"<b>Device:</b> {self._device}"
        )
        info.setWordWrap(True)
        v.addWidget(info)

        row_h = QHBoxLayout()
        panels = (
            ("🟢 GT (оригинал)", pix_gt, "#1a7f1a"),
            ("🔴 Предсказание модели", pix_pr, "#a51b1b"),
        )
        for title, pix, color in panels:
            col = QVBoxLayout()
            header = QLabel(f"<b style='color:{color}'>{title}</b>")
            header.setAlignment(Qt.AlignCenter)
            col.addWidget(header)

            lbl = QLabel()
            if not pix.isNull():
                scaled = pix.scaled(
                    720, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                lbl.setPixmap(scaled)
            else:
                lbl.setText("(не удалось отрисовать)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid #888; background:#111; padding:2px;")
            lbl.setMinimumSize(300, 300)
            col.addWidget(lbl, 1)
            row_h.addLayout(col)

        v.addLayout(row_h, 1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_close)
        v.addLayout(btn_box)

        dlg.exec_()

    def browse_val_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "data.yaml тестового датасета",
            "",
            "YAML (*.yaml *.yml);;Все файлы (*)",
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Папка тестового датасета")
        if path:
            self.val_data_edit.setText(path)

    def _resolve_val_data_path(self, path):
        if not path:
            return None
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for name in ("data.yaml", "data.yml"):
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    return cand
            return None
        return None

    def run_yolo_validation(self):
        if not self._audit_model_path:
            QMessageBox.warning(self, "Модель не выбрана", "Загрузите YOLO-модель.")
            return
        raw = self.val_data_edit.text().strip()
        if not raw:
            QMessageBox.warning(
                self, "Нет данных", "Укажите путь к data.yaml или папке."
            )
            return
        if not os.path.exists(raw):
            QMessageBox.warning(self, "Не найдено", f"Путь не существует:\n{raw}")
            return
        data_path = self._resolve_val_data_path(raw)
        if data_path is None:
            QMessageBox.warning(
                self, "data.yaml не найден", f"В папке нет data.yaml:\n{raw}"
            )
            return
        if self._val_thread and self._val_thread.isRunning():
            QMessageBox.information(self, "Идёт валидация", "Дождитесь завершения.")
            return

        split = self.val_split_combo.currentText()
        imgsz = int(self.val_imgsz_spin.value())
        conf = float(self.val_conf_spin.value())
        iou = float(self.val_iou_spin.value())

        self._audit_log(
            f"Старт валидации: model={self._audit_model_path}, "
            f"data={data_path}, split={split}, imgsz={imgsz}, "
            f"conf={conf}, iou={iou}, device={self._device}",
            tag="VAL",
        )

        self.val_table.setRowCount(0)
        self.val_summary_label.setText("Валидация запущена…")
        self.btn_run_validation.setEnabled(False)
        self.btn_stop_validation.setEnabled(True)

        self._val_thread = YoloValidationWorker(
            model_path=self._audit_model_path,
            data_path=data_path,
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self._device,
        )
        self._val_thread.log_signal.connect(self.log)
        self._val_thread.log_signal.connect(
            lambda m: print(f"[VAL][THREAD] {m}", flush=True)
        )
        self._val_thread.finished_signal.connect(self._on_validation_finished)
        self._start_audit_worker(self._val_thread)

    def stop_yolo_validation(self):
        if self._val_thread and self._val_thread.isRunning():
            self._val_thread.request_stop()
            self._audit_log("Валидация: запрошена остановка.", tag="VAL", level="WARN")

    def _on_validation_finished(self, result):
        if stale(self, result):
            return
        self.btn_run_validation.setEnabled(True)
        self.btn_stop_validation.setEnabled(False)
        err = result.get("error")
        if err:
            self.val_summary_label.setText(f"❌ {err}")
            self._audit_log(f"Валидация ошибка: {err}", tag="VAL", level="ERROR")
            return

        m = result.get("metrics") or {}
        summary = m.get("summary", {})
        per_class = m.get("per_class", [])
        self._last_validation = m

        def _fmt(v, nd=4):
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return "—"

        self.val_summary_label.setText(
            f"<b>Precision:</b> {_fmt(summary.get('Precision'))} | "
            f"<b>Recall:</b> {_fmt(summary.get('Recall'))} | "
            f"<b>F1:</b> {_fmt(summary.get('F1'))} | "
            f"<b>mAP50:</b> {_fmt(summary.get('mAP50'))} | "
            f"<b>mAP50-95:</b> {_fmt(summary.get('mAP50-95'))} | "
            f"<b>mAP75:</b> {_fmt(summary.get('mAP75'))}"
        )

        self.val_table.setRowCount(len(per_class))
        for r, row in enumerate(per_class):
            for c, key in enumerate(("class", "p", "r", "f1", "ap50", "ap")):
                it = QTableWidgetItem(str(row.get(key, "")))
                it.setToolTip(
                    f"{row.get('class', '')}\n"
                    f"Precision: {row.get('p', '')}\n"
                    f"Recall:    {row.get('r', '')}\n"
                    f"F1:        {row.get('f1', '')}\n"
                    f"mAP50:     {row.get('ap50', '')}\n"
                    f"mAP50-95:  {row.get('ap', '')}"
                )
                self.val_table.setItem(r, c, it)

        self._audit_log(
            f"Валидация завершена: "
            f"P={_fmt(summary.get('Precision'))}, "
            f"R={_fmt(summary.get('Recall'))}, "
            f"F1={_fmt(summary.get('F1'))}, "
            f"mAP50={_fmt(summary.get('mAP50'))}, "
            f"mAP50-95={_fmt(summary.get('mAP50-95'))}",
            tag="VAL",
            level="OK",
        )
