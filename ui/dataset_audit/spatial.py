"""Dataset analysis: spatial Qt controller."""


class DatasetSpatialController:
    def run_heatmap_analysis(self):
        self.scans.start("heatmap")

    def present_heatmap(self, result):
        for error in result.get("errors", []):
            self._audit_log(str(error), tag="HEATMAP", level="WARN")
        self._last_heatmap = result
        heatmap = result["heatmap"]
        total_objs = result["total_objects"]
        grid = result["grid_size"]

        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except Exception:
                pass
            self._heatmap_colorbar = None

        self.heatmap_ax.clear()
        if total_objs == 0:
            self.heatmap_ax.text(0.5, 0.5, "Нет объектов", ha="center", va="center")
            self.heatmap_ax.set_title("Тепловая карта: пусто", fontsize=9)
            self._audit_log(
                "Тепловая карта: нет объектов.", tag="HEATMAP", level="WARN"
            )
        else:
            im = self.heatmap_ax.imshow(
                heatmap,
                cmap="hot",
                interpolation="nearest",
                extent=[0, 100, 100, 0],
                aspect="auto",
            )
            self.heatmap_ax.set_xlabel("X (% ширины, 0 = левый край)", fontsize=8)
            self.heatmap_ax.set_ylabel("Y (% высоты, 0 = верх)", fontsize=8)
            self.heatmap_ax.set_title(
                f"Распределение центров объектов "
                f"(всего {total_objs}, сетка {grid}×{grid})",
                fontsize=9,
            )
            cbar = self.heatmap_figure.colorbar(
                im, ax=self.heatmap_ax, fraction=0.046, pad=0.04
            )
            cbar.set_label(
                "Число объектов в ячейке", rotation=270, labelpad=16, fontsize=8
            )
            self._heatmap_colorbar = cbar
            self._audit_log(
                f"Тепловая карта: объектов={total_objs}, grid={grid}x{grid}",
                tag="HEATMAP",
                level="OK",
            )

        try:
            self.heatmap_figure.tight_layout()
        except Exception:
            pass
        self.heatmap_canvas.draw()
        self.heatmap_summary_label.setText(
            f"Объектов: {total_objs}. Сетка {grid}×{grid}. "
            f"Оси — в % от размера изображения; цвет — число "
            f"центров объектов в ячейке."
        )

    def stop_heatmap_analysis(self):
        self.scans.cancel()
