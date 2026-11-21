"""Dataset analysis: duplicates Qt controller."""

import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import TrainingImageLabel
from utils_new.dataset_audit.objects import load_pair_annotations
from utils_new.dataset_files_new import read_image, read_index_mask


class DatasetDuplicatesController:
    def run_duplicate_analysis(self):
        self.scans.start("duplicates")

    def present_duplicates(self, dups):
        exact = dups.get("exact", [])
        near = dups.get("near", [])
        leak_report = dups.get("leak", {"groups": [], "leaked_count": 0})
        leaked = leak_report.get("leaked_count", 0)

        self._last_dup = {
            "phash_available": dups.get("phash_available", True),
            "errors": dups.get("errors", []),
            "exact": exact,
            "near": near,
            "leak": leak_report,
            "cancelled": dups.get("cancelled", False),
        }

        self._audit_log(
            f"Дубликаты: exact={len(exact)} групп, "
            f"near={len(near)} групп, утечка={leaked}",
            tag="DUP",
            level="ERROR" if leaked else "OK",
        )

        self._clear_duplicate_highlights()
        self._highlight_duplicates_in_file_list(dups, leak_report)

        self._rebuild_dup_table()

        lines = [f"Точных групп: {len(exact)}, почти-групп: {len(near)}."]
        if leaked:
            lines.append(f"🚨 Утечка в {leaked} группах — проверьте сплиты!")
        else:
            lines.append("Среди проверенных групп утечек не найдено.")
        if not dups.get("phash_available", True):
            lines.append(
                "Поиск похожих снимков недоступен: не установлен ImageHash. Проверены точные повторы и YAML split."
            )
        if dups.get("errors"):
            lines.append(f"Не проверено файлов: {len(dups['errors'])}.")
        if exact or near:
            lines.append(
                "Найденные изображения подсвечены в списке пар слева. "
                "Клик по строке таблицы выделяет пары группы."
            )
        self.dup_summary_label.setText("<br>".join(lines))

        for rg in leak_report["groups"]:
            if rg["leak"]:
                names = ", ".join(os.path.basename(p) for p in rg["paths"][:3])
                self._audit_log(f"  утечка {rg['splits']}: {names}", tag="DUP")

    def stop_duplicate_analysis(self):
        self.scans.cancel()

    def _clear_duplicate_highlights(self):
        if not hasattr(self, "file_list"):
            return
        self.file_list.blockSignals(True)
        try:
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                it.setData(Qt.BackgroundRole, None)
                it.setToolTip("Ctrl+C — копировать.\nДвойной клик — открыть пару.")
        finally:
            self.file_list.blockSignals(False)

    def _highlight_duplicates_in_file_list(self, dups, leak_report):
        if not hasattr(self, "file_list"):
            return

        exact_paths = set()
        for g in dups.get("exact", []):
            exact_paths.update(g)

        near_paths = set()
        for g in dups.get("near", []):
            near_paths.update(g)
        near_paths -= exact_paths

        leak_paths = set()
        for g in leak_report.get("groups", []):
            if g.get("leak"):
                leak_paths.update(g["paths"])

        self.file_list.blockSignals(True)
        try:
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                pair = it.data(Qt.UserRole + 1)
                if not pair:
                    continue
                p = pair[0]
                if p in leak_paths:
                    it.setBackground(QColor(255, 180, 180))
                    it.setToolTip(
                        (it.toolTip() or "")
                        + "\n🚨 Утечка между сплитами (train/val/test)"
                    )
                elif p in exact_paths:
                    it.setBackground(QColor(255, 210, 210))
                    it.setToolTip((it.toolTip() or "") + "\n🔴 Точный дубликат (MD5)")
                elif p in near_paths:
                    it.setBackground(QColor(255, 235, 200))
                    it.setToolTip(
                        (it.toolTip() or "") + "\n🟠 Похожее изображение (pHash ≤ 8)"
                    )
        finally:
            self.file_list.blockSignals(False)

    def _on_dup_row_selected(self):
        """Клик по строке таблицы дубликатов → выделить её пары."""
        if not hasattr(self, "file_list"):
            return
        try:
            sel_model = self.dup_table.selectionModel()
            if sel_model is None:
                return
            rows = sel_model.selectedRows()
        except Exception:
            return

        if not rows:
            self.file_list.blockSignals(True)
            try:
                self.file_list.clearSelection()
            finally:
                self.file_list.blockSignals(False)
            return

        row = rows[0].row()
        groups = getattr(self, "_dup_groups_by_row", [])
        if row < 0 or row >= len(groups):
            return
        paths = set(groups[row])
        if not paths:
            return

        self.file_list.blockSignals(True)
        try:
            self.file_list.clearSelection()
            first = None
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it is None:
                    continue
                pair = it.data(Qt.UserRole + 1)
                if pair and pair[0] in paths:
                    it.setSelected(True)
                    if first is None:
                        first = it
            if first is not None:
                self.file_list.scrollToItem(first, QAbstractItemView.PositionAtCenter)
        finally:
            self.file_list.blockSignals(False)

        self._audit_log(
            f"[DUP] Выбрана группа #{row + 1} ({len(paths)} пар).", tag="DUP"
        )

    def _on_dup_row_double_clicked(self, row, col):
        """Двойной клик по строке таблицы дубликатов →
        открывает ОДНО окно со всеми парами группы."""
        groups = getattr(self, "_dup_groups_by_row", [])
        if row < 0 or row >= len(groups):
            return
        paths = list(groups[row])
        if not paths:
            return
        by_path = {p[0]: p for p in self.pairs}
        pairs_to_open = [by_path[path] for path in paths if path in by_path]
        if not pairs_to_open:
            self._audit_log(
                f"[DUP] Группа #{row + 1}: пары не найдены в self.pairs",
                tag="DUP",
                level="WARN",
            )
            return
        self._open_dup_group_validation(pairs_to_open, row + 1)

    def _on_delete_pair_from_dup(self, card_widget, img_path, lbl_path):
        """Удаление пары прямо из окна просмотра группы дубликатов."""
        pair = (img_path, lbl_path) if lbl_path else (img_path,)
        if self._delete_pair_files(pair):
            card_widget.setVisible(False)

    def _open_dup_group_validation(self, pairs, group_num):
        """Одно окно с прокруткой, показывающее все пары группы
        (изображение + наложенная GT-разметка)."""

        n = len(pairs)
        if n == 0:
            return

        if n > 100:
            reply = QMessageBox.question(
                self,
                "Большая группа",
                f"В группе {n} пар — рендер может занять время и память.\nПродолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._audit_log(f"[DUP] Открытие окна группы #{group_num} ({n} пар)", tag="DUP")

        prog = QProgressDialog(f"Подготовка {n} изображений…", "Отмена", 0, n, self)
        prog.setWindowTitle("Группа дубликатов")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)
        prog.setValue(0)

        rendered = []
        cancelled = False

        for idx, pair in enumerate(pairs, 1):
            if prog.wasCanceled():
                cancelled = True
                break
            prog.setValue(idx - 1)
            prog.setLabelText(f"Подготовка {idx}/{n}: {os.path.basename(pair[0])}")
            QApplication.processEvents()

            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None

            if not os.path.exists(img_path):
                rendered.append((idx, img_path, lbl_path, None, "файл не найден"))
                continue

            try:
                base = read_image(img_path)
            except Exception as e:
                rendered.append((idx, img_path, lbl_path, None, f"ошибка чтения: {e}"))
                continue

            if base is None:
                rendered.append((idx, img_path, lbl_path, None, "не удалось прочитать"))
                continue

            h, w = base.shape[:2]
            img = base.copy()

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
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                            cv2.putText(
                                img,
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
                            cv2.polylines(img, [arr], True, (0, 200, 0), 2)
                elif self.dataset_type == 1 and lbl_path and os.path.exists(lbl_path):
                    mask = read_index_mask(lbl_path)
                    if mask is not None and mask.ndim == 2:
                        overlay = np.zeros_like(img)
                        overlay[mask > 0] = (0, 200, 0)
                        img = cv2.addWeighted(img, 1.0, overlay, 0.35, 0)
                        contours, _ = cv2.findContours(
                            (mask > 0).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(img, contours, -1, (0, 200, 0), 2)
            except Exception as e:
                self._audit_log(
                    f"[DUP] Ошибка отрисовки GT {img_path}: {e}",
                    tag="DUP",
                    level="WARN",
                )

            try:
                _, buf = cv2.imencode(".png", img)
                pix = QPixmap()
                pix.loadFromData(buf.tobytes(), "PNG")
                rendered.append((idx, img_path, lbl_path, pix, None))
            except Exception as e:
                rendered.append((idx, img_path, lbl_path, None, f"кодирование: {e}"))

        prog.setValue(n)

        if not rendered:
            if cancelled:
                self._audit_log(
                    "[DUP] Рендеринг отменён пользователем", tag="DUP", level="WARN"
                )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Группа дубликатов #{group_num} — {len(rendered)} пар")
        dlg.resize(1400, 900)
        v = QVBoxLayout(dlg)

        header = QLabel(
            f"<b>Группа дубликатов #{group_num}</b> | "
            f"Пар: <b>{len(rendered)}</b>. "
            f"Прокручивайте вниз, чтобы увидеть все."
        )
        header.setWordWrap(True)
        v.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(14)

        for idx, img_path, lbl_path, pix, err in rendered:
            card = QWidget()
            card.setObjectName("pair_card")
            card.setStyleSheet(
                "QWidget#pair_card {"
                " background-color: #fafafa;"
                " border: 1px solid #ddd;"
                " border-radius: 4px; }"
            )
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(8, 8, 8, 8)
            card_lay.setSpacing(6)

            name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path) if lbl_path else "—"
            info = QLabel(
                f"<b>#{idx}</b> — <code>{name}</code><br>"
                f"<span style='color:#666;font-size:11px'>"
                f"Метка: {lbl_name} | Путь: {img_path}</span>"
            )
            info.setWordWrap(True)
            info.setTextInteractionFlags(Qt.TextSelectableByMouse)
            card_lay.addWidget(info)

            if err:
                err_lbl = QLabel(f"❌ {err}")
                err_lbl.setStyleSheet(
                    "color: #b00000; font-weight: bold; padding: 8px;"
                )
                err_lbl.setWordWrap(True)
                card_lay.addWidget(err_lbl)
            elif pix is not None and not pix.isNull():
                if TrainingImageLabel is not None:
                    img_lbl = TrainingImageLabel()
                    img_lbl.set_image(pix)
                    img_lbl.setMinimumHeight(360)
                    card_lay.addWidget(img_lbl)
                else:
                    img_lbl = QLabel()
                    scaled = pix.scaled(
                        1100, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    img_lbl.setPixmap(scaled)
                    img_lbl.setAlignment(Qt.AlignCenter)
                    card_lay.addWidget(img_lbl)

            del_row = QHBoxLayout()
            del_row.addStretch()
            btn_del = QPushButton("🗑 Удалить пару")
            btn_del.setStyleSheet(
                "background-color: #d32f2f; color: white; "
                "font-weight: bold; padding: 4px 14px;"
            )
            btn_del.setToolTip("Удалить изображение и файл разметки/маски с диска.")
            btn_del.clicked.connect(
                lambda _checked=False, c=card, ip=img_path, lp=lbl_path: (
                    self._on_delete_pair_from_dup(c, ip, lp)
                )
            )
            del_row.addWidget(btn_del)
            card_lay.addLayout(del_row)

            lay.addWidget(card)

        lay.addStretch()
        scroll.setWidget(container)
        v.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        v.addLayout(btn_row)

        dlg.exec_()
