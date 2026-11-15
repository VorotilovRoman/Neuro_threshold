"""Qt tables and plots for dataset statistics."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem


def get_included_class_ids(class_table):
    included = set()
    for row in range(class_table.rowCount()):
        cb = class_table.cellWidget(row, 4)
        if cb and cb.isChecked():
            try:
                included.add(int(class_table.item(row, 1).text()))
            except (ValueError, TypeError, AttributeError):
                continue
    return included


def get_class_names_by_orig_id(class_table):
    names = {}
    for row in range(class_table.rowCount()):
        try:
            orig = int(class_table.item(row, 1).text())
        except (ValueError, TypeError, AttributeError):
            continue
        name_item = class_table.item(row, 2)
        names[orig] = name_item.text().strip() if name_item else str(orig)
    return names


def get_class_names_included(class_table):
    names = {}
    for row in range(class_table.rowCount()):
        cb = class_table.cellWidget(row, 4)
        if not (cb and cb.isChecked()):
            continue
        try:
            orig_id = int(class_table.item(row, 1).text())
        except (ValueError, TypeError, AttributeError):
            continue
        name_item = class_table.item(row, 2)
        name = name_item.text().strip() if name_item else str(orig_id)
        names[orig_id] = name or str(orig_id)
    return names


def render_metrics(
    metrics_summary_label,
    class_stats_table,
    ax_all,
    ax_classes,
    canvas,
    filtered_pairs,
    all_pairs,
    class_counts,
    total_objects,
    object_areas,
    class_areas,
    name_by_orig,
    ax_pie=None,
    ax_bar=None,
    ax_area_pct=None,
    object_areas_pct=None,
):
    metrics_summary_label.setText(
        f"<b>Снимков после фильтрации:</b> {len(filtered_pairs)} из {len(all_pairs)}<br><b>Всего объектов (по выбранным классам):</b> {total_objects}<br><b>Уникальных классов:</b> {len(class_counts)}"
    )
    class_stats_table.setRowCount(0)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_stats_table.setRowCount(len(sorted_classes))
    for row, (cls_id, count) in enumerate(sorted_classes):
        name = name_by_orig.get(cls_id, str(cls_id))
        item0 = QTableWidgetItem(f"{cls_id}: {name}")
        item0.setFlags(item0.flags() & ~Qt.ItemIsEditable)
        item1 = QTableWidgetItem(str(count))
        item1.setFlags(item1.flags() & ~Qt.ItemIsEditable)
        item2 = QTableWidgetItem(
            f"{count / total_objects * 100:.1f}%" if total_objects else "0%"
        )
        item2.setFlags(item2.flags() & ~Qt.ItemIsEditable)
        class_stats_table.setItem(row, 0, item0)
        class_stats_table.setItem(row, 1, item1)
        class_stats_table.setItem(row, 2, item2)
    ax_all.clear()
    if object_areas:
        ax_all.hist(object_areas, bins=50, color="skyblue", edgecolor="black", log=True)
        ax_all.set_title("Площади: все объекты (px²)")
        ax_all.set_xlabel("Площадь (px²)")
        ax_all.set_ylabel("Кол-во (Log)")
    else:
        ax_all.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        ax_all.set_title("Площади: все объекты (px²)")
    if ax_area_pct is not None:
        ax_area_pct.clear()
        if object_areas_pct:
            ax_area_pct.hist(
                object_areas_pct,
                bins=50,
                color="lightcoral",
                edgecolor="black",
                log=True,
            )
            ax_area_pct.set_title("Площади: все объекты (%)")
            ax_area_pct.set_xlabel("Площадь (% от изображения)")
            ax_area_pct.set_ylabel("Кол-во (Log)")
        else:
            ax_area_pct.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax_area_pct.set_title("Площади: все объекты (%)")
    ax_classes.clear()
    if class_areas:
        top = sorted(class_areas.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        for i, (cls_id, areas) in enumerate(top):
            name = name_by_orig.get(cls_id, str(cls_id))
            ax_classes.hist(
                areas,
                bins=40,
                alpha=0.55,
                color=colors[i % len(colors)],
                edgecolor="black",
                log=True,
                label=f"{cls_id}: {name} (n={len(areas)})",
            )
        ax_classes.legend(fontsize=6, loc="upper right")
        ax_classes.set_title("Площади по классам (топ-10, px²)")
        ax_classes.set_xlabel("Площадь (px²)")
        ax_classes.set_ylabel("Log")
    else:
        ax_classes.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        ax_classes.set_title("Площади по классам (px²)")
    try:
        ax_all.figure.tight_layout()
    except Exception:
        pass
    canvas.draw()
    if ax_pie is None or ax_bar is None:
        return
    ax_pie.clear()
    ax_bar.clear()
    if not class_counts:
        ax_pie.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        ax_pie.set_title("Распределение по классам", fontsize=9)
        ax_bar.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        ax_bar.set_title("Кол-во по классам", fontsize=9)
        try:
            ax_pie.figure.tight_layout()
            ax_pie.figure.canvas.draw()
        except Exception:
            pass
        return
    if len(sorted_classes) > 10:
        top_pie = sorted_classes[:9]
        other_count = sum((c for _, c in sorted_classes[9:]))
        labels_pie = [f"{cid}: {name_by_orig.get(cid, cid)}" for cid, _ in top_pie] + [
            f"Прочие ({len(sorted_classes) - 9} кл.)"
        ]
        sizes_pie = [c for _, c in top_pie] + [other_count]
    else:
        labels_pie = [
            f"{cid}: {name_by_orig.get(cid, cid)}" for cid, _ in sorted_classes
        ]
        sizes_pie = [c for _, c in sorted_classes]
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(sizes_pie))]
    except Exception:
        colors = None
    try:
        ax_pie.pie(
            sizes_pie,
            labels=labels_pie,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 6},
            colors=colors,
        )
        ax_pie.set_title(
            f"Распределение по классам (объектов: {total_objects})", fontsize=9
        )
    except Exception:
        ax_pie.text(0.5, 0.5, "Ошибка отрисовки", ha="center", va="center")
        ax_pie.set_title("Распределение по классам", fontsize=9)
    try:
        names_bar = [
            f"{cid}: {name_by_orig.get(cid, cid)}" for cid, _ in sorted_classes
        ]
        counts_bar = [c for _, c in sorted_classes]
        y_pos = list(range(len(counts_bar)))
        ax_bar.barh(y_pos, counts_bar, color="skyblue", edgecolor="black")
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names_bar, fontsize=6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Кол-во объектов", fontsize=7)
        ax_bar.set_title(f"Кол-во по классам (классов: {len(counts_bar)})", fontsize=9)
        for i, c in enumerate(counts_bar):
            ax_bar.text(c, i, f"  {c}", va="center", fontsize=6)
    except Exception:
        ax_bar.text(0.5, 0.5, "Ошибка отрисовки", ha="center", va="center")
        ax_bar.set_title("Кол-во по классам", fontsize=9)
    try:
        ax_pie.figure.tight_layout()
        ax_pie.figure.canvas.draw()
    except Exception:
        pass
