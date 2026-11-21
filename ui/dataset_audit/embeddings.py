"""Dataset analysis: embeddings Qt controller."""

import os
from collections import Counter

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ui.dataset_audit.lifecycle import stale


class DatasetEmbeddingController:
    def run_embedding_clustering(self):
        pairs = self._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(self, "Нет данных", "Нет пар для анализа.")
            return
        if self._cluster_thread and self._cluster_thread.isRunning():
            QMessageBox.information(self, "Идёт кластеризация", "Дождитесь.")
            return

        from ui.dataset_audit.workers import EmbeddingClusterWorker
        from utils_new.dataset_audit.checks import CLIP_AVAILABLE

        if not CLIP_AVAILABLE:
            QMessageBox.warning(
                self, "CLIP не установлен", "pip install open_clip_torch"
            )
            return

        self.btn_run_cluster.setEnabled(False)
        self.btn_stop_cluster.setEnabled(True)
        self.cluster_summary_label.setText("Извлечение эмбеддингов…")
        self.cluster_ax.clear()
        self.cluster_ax.text(0.5, 0.5, "Обработка…", ha="center", va="center")
        self.cluster_canvas.draw()
        self._audit_log(
            f"Старт кластеризации: pairs={len(pairs)}, device={self._device}",
            tag="CLUSTER",
        )

        self._cluster_thread = EmbeddingClusterWorker(
            pairs, batch_size=16, device=self._device
        )
        self._cluster_thread.log_signal.connect(self.log)
        self._cluster_thread.log_signal.connect(
            lambda m: print(f"[CLUSTER][THREAD] {m}", flush=True)
        )
        self._cluster_thread.progress.connect(self._on_cluster_progress)
        self._cluster_thread.finished_signal.connect(self._on_cluster_finished)
        self._start_audit_worker(self._cluster_thread)

    def stop_embedding_clustering(self):
        if self._cluster_thread and self._cluster_thread.isRunning():
            self._cluster_thread.cancel()
            self._audit_log(
                "Кластеризация: запрошена остановка.", tag="CLUSTER", level="WARN"
            )

    def _on_cluster_progress(self, c, t):
        if stale(self):
            return
        self.cluster_summary_label.setText(f"⏳ CLIP: {c}/{t}…")

    def _on_cluster_finished(self, result):
        if stale(self, result):
            return
        from matplotlib import pyplot as plt

        self.btn_run_cluster.setEnabled(True)
        self.btn_stop_cluster.setEnabled(False)

        err = result.get("error")
        if err:
            self.cluster_summary_label.setText(f"❌ {err}")
            self._audit_log(
                f"Кластеризация ошибка: {err}", tag="CLUSTER", level="ERROR"
            )
            return

        if result.get("cancelled"):
            self.cluster_summary_label.setText("Кластеризация отменена.")
            return
        paths = result["paths"]
        emb_2d = np.array(result["embeddings_2d"])
        clusters = np.array(result["clusters"])

        self._last_cluster = {
            "count": len(paths),
            "n_clusters": len(set(clusters.tolist()) - {-1}),
            "noise": int((clusters == -1).sum()),
            "sizes": dict(Counter(clusters.tolist())),
        }
        self._last_cluster_data = {
            "embeddings_2d": emb_2d.tolist(),
            "paths": paths,
            "clusters": clusters.tolist(),
        }

        self.cluster_ax.clear()
        unique = sorted(set(clusters.tolist()))
        cmap = plt.get_cmap("tab20")
        for i, cl in enumerate(unique):
            mask = clusters == cl
            color = "lightgray" if cl == -1 else cmap(i % 20)
            label = "Шум" if cl == -1 else f"Кластер {cl}"
            self.cluster_ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[color],
                label=label,
                s=20,
                alpha=0.7,
            )
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = int((clusters == -1).sum())
        self.cluster_ax.set_title(f"UMAP + HDBSCAN (кластеров: {n_clusters})")
        self.cluster_ax.set_xlabel("UMAP-1")
        self.cluster_ax.set_ylabel("UMAP-2")
        if len(unique) <= 20:
            self.cluster_ax.legend(fontsize=6, loc="best")
        self.cluster_figure.tight_layout()
        self.cluster_canvas.draw()

        self.cluster_summary_label.setText(
            f"Изображений: {len(paths)}. Кластеров: {n_clusters}. "
            f"Шум: {n_noise}. Клик по точке — открыть пару. "
            f"Панель инструментов — зум/pan/сохранение. "
            f"Device: {self._device}."
        )
        self._audit_log(
            f"Кластеризация готова: images={len(paths)}, "
            f"clusters={n_clusters}, noise={n_noise}",
            tag="CLUSTER",
            level="OK",
        )

    def _on_cluster_click(self, event):
        if not self._last_cluster_data:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        tb = getattr(self, "cluster_toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return

        emb = np.asarray(self._last_cluster_data["embeddings_2d"])
        paths = self._last_cluster_data["paths"]
        if not len(emb):
            return

        d = np.sqrt((emb[:, 0] - event.xdata) ** 2 + (emb[:, 1] - event.ydata) ** 2)
        idx = int(np.argmin(d))
        try:
            threshold = float(self.cluster_click_radius_spin.value())
        except AttributeError:
            threshold = 0.5

        if d[idx] > threshold:
            self._audit_log(
                f"[CLUSTER] Клик мимо (d={d[idx]:.2f} > {threshold:.2f})",
                tag="CLUSTER",
                level="WARN",
            )
            return
        if idx < 0 or idx >= len(paths):
            return
        target = paths[idx]

        for p in self.pairs:
            if p[0] == target:
                self._audit_log(
                    f"[CLUSTER] Открытие: {os.path.basename(target)}", tag="CLUSTER"
                )
                self._open_pair_validation(p)
                return
