"""Numerical audit/embedding/report/validation jobs. No Qt; callbacks and dict results."""
from pathlib import Path

import os
import traceback
import numpy as np
from PIL import Image
from utils_new.dataset_audit.objects import read_objects
from utils_new.dataset_audit.matching import match_detections, project_embeddings
from utils_new.import_export.atomic import write_text_atomic
from utils_new.dataset_audit.checks import detect_device, _safe_progress, CLIP_AVAILABLE, _CLIP_BACKEND, UMAP_AVAILABLE, HDBSCAN_AVAILABLE


class LossInspectionJob:
    def __init__(
        self,
        pairs,
        model_path,
        dataset_type=0,
        iou_threshold=0.3,
        conf_threshold=0.25,
        imgsz=640,
        device=None,
        input_format="auto",
        mask_class_map=None,
    ):
        self.pairs = tuple(tuple(pair) for pair in pairs)
        self.model_path = model_path
        self.mask_class_map = mask_class_map
        self.dataset_type = dataset_type
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device or detect_device()
        self.input_format = input_format
        self._cancel = False
        self.log = lambda message: None
        self.progress = lambda *args: None

    def cancel(self):
        self._cancel = True

    def run(self):
        self.log(f"[LossInspection] старт (device: {self.device})")
        if self.dataset_type == 2:
            return {
                "error": "Аудит модели требует аннотаций или масок.",
                "suspicious": [],
            }
        if self._cancel:
            return {
                "error": None,
                "cancelled": True,
                "suspicious": [],
                "all_results": [],
            }
        try:
            from runtime_policy import configure_ultralytics

            configure_ultralytics()
            from runtime_torch import require_torch

            require_torch()
            from utils_new.import_export.model_load import load_yolo as YOLO
        except (ImportError, OSError, RuntimeError) as exc:
            return {"error": f"PyTorch/Ultralytics недоступен: {exc}", "suspicious": []}
        try:
            model = YOLO(self.model_path)
            if self.device.startswith("cuda"):
                try:
                    model.to(self.device)
                except Exception as e:
                    self.log(
                        f"[LossInspection] Не удалось на {self.device} ({e}), fallback на cpu"
                    )
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                    self.device = "cpu"
            self.log(f"[LossInspection] модель загружена (device: {self.device})")
        except Exception as e:
            return {"error": f"Не удалось загрузить модель: {e}", "suspicious": []}
        suspicious = []
        all_results = []
        total = len(self.pairs)
        processed = 0
        errors = 0
        zero_pred_logged = 0
        for i, pair in enumerate(self.pairs):
            if self._cancel:
                self.log("[LossInspection] отменено")
                break
            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None
            if not lbl_path or not os.path.exists(lbl_path):
                errors += 1
                self.log(f"Нет разметки для аудита: {img_path}")
                _safe_progress(self.progress, i, total, step=10)
                continue
            try:
                results = model(
                    img_path,
                    conf=self.conf_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                if not results:
                    raise ValueError("Модель не вернула результат.")
                r = results[0]
                boxes = getattr(r, "boxes", None)
                obb = getattr(r, "obb", None)
                pred_xyxy = None
                pred_conf = None
                pred_classes = []
                if boxes is not None and len(boxes) > 0:
                    try:
                        pred_xyxy = boxes.xyxy.cpu().numpy()
                        pred_conf = boxes.conf.cpu().numpy()
                        pred_classes = boxes.cls.cpu().numpy()
                    except Exception as e:
                        self.log(f"[LossInspection] boxes extract: {e}")
                if (
                    (pred_xyxy is None or len(pred_xyxy) == 0)
                    and obb is not None
                    and (len(obb) > 0)
                ):
                    try:
                        pred_xyxy = obb.xyxy.cpu().numpy()
                        pred_conf = obb.conf.cpu().numpy()
                        pred_classes = obb.cls.cpu().numpy()
                    except Exception as e:
                        self.log(f"[LossInspection] obb extract: {e}")
                pred_count = len(pred_xyxy) if pred_xyxy is not None else 0
                if pred_count == 0 and zero_pred_logged < 5:
                    masks = getattr(r, "masks", None)
                    self.log(
                        f"[LossInspection] {os.path.basename(img_path)}: 0 предсказаний — boxes={(len(boxes) if boxes is not None else None)}, obb={(len(obb) if obb is not None else None)}, masks={(len(masks) if masks is not None else None)}, conf_thr={self.conf_threshold}, imgsz={self.imgsz}"
                    )
                    zero_pred_logged += 1
                h, w = r.orig_shape if hasattr(r, "orig_shape") else (0, 0)
                if h == 0 or w == 0:
                    from utils_new.import_export.images import open_pillow_image
                    with open_pillow_image(img_path) as im:
                        w, h = im.size
                size, gt_objects = read_objects(
                    pair, self.dataset_type, self.input_format
                )
                gt_xyxy = []
                gt_classes = []
                for obj in gt_objects:
                    cls = obj.class_id
                    if self.dataset_type == 1:
                        if self.mask_class_map is not None:
                            if cls not in self.mask_class_map:
                                continue
                            cls = self.mask_class_map[cls]
                        else:
                            cls -= 1
                    x0, y0, x1, y1 = obj.bounds
                    gt_xyxy.append([x0 * w, y0 * h, x1 * w, y1 * h])
                    gt_classes.append(cls)

                gt_count = len(gt_xyxy)
                matched = match_detections(
                    gt_xyxy,
                    gt_classes,
                    [] if pred_xyxy is None else pred_xyxy,
                    pred_classes,
                    self.iou_threshold,
                )
                mean_iou = (
                    float(np.mean(matched["ious"]))
                    if gt_count
                    else (1.0 if not pred_count else 0.0)
                )
                min_iou = (
                    float(np.min(matched["ious"]))
                    if gt_count
                    else (1.0 if not pred_count else 0.0)
                )
                is_suspicious = bool(
                    matched["false_positives"] or matched["false_negatives"]
                )
                reason = (
                    f"FN={matched['false_negatives']}, FP={matched['false_positives']}, mean IoU={mean_iou:.3f}"
                    + (" — OK" if not is_suspicious else "")
                )
                item = {
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "reason": reason,
                    "pred_count": pred_count,
                    "gt_count": gt_count,
                    "mean_iou": mean_iou,
                    "min_iou": min_iou,
                    "min_conf": float(pred_conf.min())
                    if pred_conf is not None and len(pred_conf)
                    else 0.0,
                    "false_positives": matched["false_positives"],
                    "false_negatives": matched["false_negatives"],
                    "is_suspicious": is_suspicious,
                }
                processed += 1
                all_results.append(item)
                if is_suspicious:
                    suspicious.append(item)
            except Exception as e:
                errors += 1
                self.log(f"[LossInspection] ошибка {img_path}: {e}")
            _safe_progress(self.progress, i, total, step=10)
        self.log(
            f"[LossInspection] готово: processed={processed}/{total}, errors={errors}, аудировано={len(all_results)}, подозрительных={len(suspicious)}"
        )
        return {
            "error": (
                "Не удалось проверить ни одного изображения; подробности в журнале."
                if errors and not all_results and not self._cancel
                else None
            ),
            "suspicious": suspicious,
            "all_results": all_results,
            "processed": processed,
            "total": total,
            "errors": errors,
            "cancelled": self._cancel,
        }


class EmbeddingClusterJob:
    def __init__(self, pairs, batch_size=16, device=None):
        self.pairs = tuple(tuple(pair) for pair in pairs)
        self.batch_size = batch_size
        self.device = device or detect_device()
        self._cancel = False
        self.log = lambda message: None
        self.progress = lambda *args: None

    def cancel(self):
        self._cancel = True

    def run(self):
        self.log(f"[Cluster] старт (backend: {_CLIP_BACKEND}, device: {self.device})")
        if self._cancel:
            return {"error": None, "cancelled": True, "clusters": []}
        if not CLIP_AVAILABLE:
            return {"error": "CLIP не установлен.", "clusters": []}
        try:
            from runtime_torch import require_torch

            require_torch()
            if _CLIP_BACKEND == "open_clip":
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
            else:
                from transformers import CLIPModel, CLIPProcessor

                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            try:
                model = model.to(self.device).eval()
            except Exception as e:
                self.log(
                    f"[Cluster] Не удалось на {self.device} ({e}), fallback на cpu"
                )
                try:
                    model = model.to("cpu")
                except Exception:
                    pass
                self.device = "cpu"
                model = model.eval()
            self.log(f"[Cluster] CLIP загружен (device: {self.device})")
        except Exception as e:
            return {"error": f"Не удалось загрузить CLIP: {e}", "clusters": []}
        embeddings = []
        paths = []
        batch_imgs = []
        total = len(self.pairs)
        from runtime_torch import require_torch

        torch = require_torch()
        from PIL import Image as _Img

        def flush_batch(batch_paths):
            if not batch_paths:
                return
            try:
                if _CLIP_BACKEND == "open_clip":
                    imgs = []
                    for p in batch_paths:
                        with _Img.open(p) as im:
                            imgs.append(preprocess(im.convert("RGB")))
                    batch = torch.stack(imgs).to(self.device)
                    with torch.no_grad():
                        feats = model.encode_image(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.extend(feats.cpu().numpy())
                else:
                    imgs = []
                    for p in batch_paths:
                        with _Img.open(p) as im:
                            imgs.append(
                                processor(
                                    images=im.convert("RGB"), return_tensors="pt"
                                )["pixel_values"][0]
                            )
                    batch = torch.stack(imgs).to(self.device)
                    with torch.no_grad():
                        feats = model.get_image_features(pixel_values=batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.extend(feats.cpu().numpy())
                paths.extend(batch_paths)
            except Exception as e:
                self.log(f"[Cluster] ошибка батча: {e}")

        for i, pair in enumerate(self.pairs):
            if self._cancel:
                self.log("[Cluster] отменено")
                break
            if os.path.exists(pair[0]):
                batch_imgs.append(pair[0])
            if len(batch_imgs) >= self.batch_size or i == total - 1:
                flush_batch(batch_imgs)
                batch_imgs = []
            _safe_progress(self.progress, i, total, step=10)
        if self._cancel:
            return {"error": None, "cancelled": True, "clusters": []}
        if not embeddings:
            return {"error": "Не удалось извлечь ни одного эмбеддинга.", "clusters": []}
        emb = np.array(embeddings)
        if self._cancel:
            return {
                "error": None,
                "cancelled": True,
                "clusters": [],
                "paths": [],
                "embeddings_2d": [],
            }
        emb_2d = project_embeddings(emb, prefer_umap=UMAP_AVAILABLE, log=self.log)
        if HDBSCAN_AVAILABLE:
            try:
                import hdbscan

                labels = hdbscan.HDBSCAN(
                    min_cluster_size=5, min_samples=3, metric="euclidean"
                ).fit_predict(emb_2d)
                clusters = labels.tolist()
                self.log("[Cluster] HDBSCAN выполнен")
            except Exception as e:
                self.log(f"[Cluster] HDBSCAN ошибка: {e}")
                clusters = [-1] * len(emb_2d)
        else:
            from sklearn.cluster import DBSCAN

            clusters = DBSCAN(eps=0.5, min_samples=3).fit_predict(emb_2d).tolist()
            self.log("[Cluster] DBSCAN вместо HDBSCAN")
        return {
            "error": None,
            "paths": paths,
            "embeddings_2d": emb_2d.tolist(),
            "clusters": clusters,
        }


class ReportJob:
    def __init__(self, data, save_path, sections):
        self.data = data
        self.save_path = save_path
        self.sections = sections
        self._cancel = False
        self.log = lambda message: None
        self.progress = lambda *args: None

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from datetime import datetime

            d = self.data
            s = self.sections
            imgs = d.get("images", {})
            L = []
            L.append("# Dataset Analysis Report\n")
            L.append(
                f"**Сгенерирован:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            gi = d.get("general", {})
            L.append("## 1. Общая информация\n")
            for k, v in gi.items():
                L.append(f"- **{k}:** {v}")
            L.append("")
            fp = d.get("filter_params")
            if fp:
                L.append("## 2. Параметры фильтрации\n")
                L.append("| Параметр | Мин | Макс |")
                L.append("|---|---:|---:|")
                L.append(
                    f"| Ширина изобр. (px) | {fp.get('min_w', 0)} | {fp.get('max_w', 0)} |"
                )
                L.append(
                    f"| Высота изобр. (px) | {fp.get('min_h', 0)} | {fp.get('max_h', 0)} |"
                )
                min_b = fp.get("min_brightness", -1)
                max_b = fp.get("max_brightness", -1)
                min_b_str = "—" if min_b is None or min_b < 0 else str(min_b)
                max_b_str = "—" if max_b is None or max_b < 0 else str(max_b)
                L.append(f"| Яркость изобр. (0–255) | {min_b_str} | {max_b_str} |")
                L.append(
                    f"| Кол-во объектов | {fp.get('min_objs', 0)} | {fp.get('max_objs', 0)} |"
                )
                L.append(
                    f"| Площадь объекта (px²) | {fp.get('min_area', 0)} | {fp.get('max_area', 0)} |"
                )
                L.append(
                    f"| Площадь объекта (%) | {fp.get('min_area_pct', 0.0):.2f} | {fp.get('max_area_pct', 0.0):.2f} |"
                )
                L.append(
                    f"| Соотношение сторон W/H | {fp.get('min_aspect', 0.0):.2f} | {fp.get('max_aspect', 0.0):.2f} |"
                )
                L.append("")
                L.append(
                    "_0 в полях «мин/макс» = без ограничения. Яркость −1 = «не задано» (0 — валидное значение: очень тёмные)._"
                )
                L.append("")
                L.append(f"- **include_empty_annotations:** {fp.get('include_empty')}")
                L.append(
                    f"- **exclude_classes_from_files:** {fp.get('exclude_classes_from_files')}"
                )
                L.append(f"- **All class IDs:** {fp.get('all_class_ids')}")
                L.append(f"- **Included class IDs:** {fp.get('included_class_ids')}")
                L.append(f"- **Excluded class IDs:** {fp.get('excluded_class_ids')}")
                L.append("")
            fs = d.get("filter_stats")
            if fs:
                L.append("### Итог фильтрации\n")
                L.append(f"- **Пар прошло фильтр:** {fs['filtered_pairs']}")
                if fs.get("empty_images"):
                    L.append(f"- **Из них пустых снимков:** {fs['empty_images']}")
                L.append(f"- **Объектов в наборе:** {fs['total_objects']}")
                L.append(f"- **Классов с объектами:** {fs['classes_with_objects']}")
                L.append("")
            cs = d.get("class_stats", [])
            if cs:
                L.append("## 3. Классы и распределение площадей\n")
                L.append("| Класс | Объектов | Доля |")
                L.append("|---|---:|---:|")
                for row in cs:
                    L.append(f"| {row[0]} | {row[1]} | {row[2]} |")
                L.append("")
            if imgs.get("areas"):
                L.append(
                    "**Распределение площадей:** слева — в px², в центре — в % от площади изображения, справа — по классам (топ-10).\n"
                )
                L.append(
                    f"![Площади: px² / % от изображения / по классам]({imgs['areas']})\n"
                )
            if imgs.get("classes"):
                L.append(f"![Распределение по классам]({imgs['classes']})\n")
            imb = d.get("imbalance")
            if imb:
                L.append("## 4. Дисбаланс классов\n")
                L.append(f"- **Imbalance ratio:** {imb['imbalance_ratio']:.2f}:1")
                L.append(f"- **Gini:** {imb['gini']:.4f}")
                L.append(f"- **Entropy:** {imb['entropy']:.4f}")
                L.append(f"- **Majority:** {imb['majority_class']}")
                L.append(f"- **Minority:** {imb['minority_class']}")
                L.append(f"- **Оценка:** {imb['warning']}")
                L.append("")
            if s.get("quality"):
                L.append("## 5. Качество изображений\n")
                q = d.get("quality")
                if not q:
                    L.append("_Не выполнялось._\n")
                else:
                    if q['median_w']>0 and q['median_h']>0:
                        L.append(f"- **Median resolution:** {q['median_w']:.0f}×{q['median_h']:.0f}")
                        L.append(f"- **Ориентир масштаба (не оценка VRAM):** {q['recommended_imgsz']}")
                    else:
                        L.append("- Нет читаемых снимков: размеры и рекомендуемый масштаб не определены.")
                    br = q.get("brightness")
                    if br:
                        L.append(f"- **Mean brightness:** {br['mean_brightness']:.1f}")
                        L.append(f"- **Тёмные (порог {q.get('thresholds',{}).get('dark',40)}):** {br['dark_count']}")
                        L.append(f"- **Светлые (порог {q.get('thresholds',{}).get('bright',215)}):** {br['bright_count']}")
                    L.append("")
                    L.append(f"- **Кандидатов / не читается:** {q.get('candidates',0)} / {q.get('unreadable',0)}")
                    L.append(f"- **Пороги:** {q.get('thresholds',{})}")
                    for row in q.get('review_rows',[])[:100]:
                        L.append(f"- `{Path(row['path']).name}`: {', '.join(row['issues'])}")
                    if imgs.get("quality"):
                        L.append(f"![Качество]({imgs['quality']})\n")
            if s.get("dup"):
                L.append("## 6. Дубликаты и утечка\n")
                dp = d.get("dup")
                if not dp:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(
                        f"- **Точные (MD5):** {dp['exact_count']} групп ({dp['exact_images']} изображений)"
                    )
                    L.append(
                        f"- **Почти (pHash):** {dp['near_count']} групп ({dp['near_images']} изображений)"
                    )
                    L.append(f"- **Утечка:** {dp['leaked_count']} групп")
                    if dp.get("leaked_groups"):
                        L.append("\n**Утёкшие группы:**\n")
                        for rg in dp["leaked_groups"]:
                            L.append(f"- {rg}")
                    L.append("")
            if s.get("heatmap"):
                L.append("## 7. Тепловая карта\n")
                hm = d.get("heatmap")
                if not hm:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Объектов:** {hm['total_objects']}")
                    L.append(f"- **Сетка:** {hm['grid']}×{hm['grid']}")
                    L.append("- **Топ-3 зоны:**")
                    for zone in hm.get("top_zones", []):
                        L.append(f"  - {zone}")
                    L.append("")
                    if imgs.get("heatmap"):
                        L.append(f"![Тепловая карта]({imgs['heatmap']})\n")
            if s.get("audit"):
                L.append("## 8. Аудит модели (Loss Inspection)\n")
                m = d.get("audit_meta")
                if not m:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Model:** `{m['model']}`")
                    L.append(f"- **IoU threshold:** {m['iou_thr']}")
                    L.append(f"- **Conf threshold:** {m['conf_thr']}")
                    L.append(f"- **imgsz:** {m.get('imgsz', '—')}")
                    L.append(f"- **Processed:** {m['processed']}/{m['total']}")
                    L.append(f"- **Errors:** {m['errors']}")
                    L.append(f"- **Suspicious:** {m['suspicious_count']}")
                    rows = d.get("audit_rows", [])
                    if rows:
                        L.append("\n| Изображение | GT | Pred | Mean IoU |")
                        L.append("|---|---:|---:|---:|")
                        for r in rows:
                            L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} |")
                    L.append("")
            if s.get("validation"):
                L.append("## 9. Валидация модели (YOLO val)\n")
                v = d.get("validation")
                if not v:
                    L.append("_Не выполнялось._\n")
                else:
                    sm = v.get("summary", {})

                    def _fmt(x):
                        try:
                            return f"{float(x):.4f}"
                        except (TypeError, ValueError):
                            return "—"

                    L.append(f"- **Precision:** {_fmt(sm.get('Precision'))}")
                    L.append(f"- **Recall:** {_fmt(sm.get('Recall'))}")
                    L.append(f"- **F1:** {_fmt(sm.get('F1'))}")
                    L.append(f"- **mAP50:** {_fmt(sm.get('mAP50'))}")
                    L.append(f"- **mAP50-95:** {_fmt(sm.get('mAP50-95'))}")
                    L.append("")
                    per_class = v.get("per_class", [])
                    if per_class:
                        L.append("| Класс | P | R | F1 | mAP50 | mAP50-95 |")
                        L.append("|---|---:|---:|---:|---:|---:|")
                        for r in per_class:
                            L.append(
                                f"| {r.get('class', '')} | {r.get('p', '')} | {r.get('r', '')} | {r.get('f1', '')} | {r.get('ap50', '')} | {r.get('ap', '')} |"
                            )
                        L.append("")
            if s.get("cluster"):
                L.append("## 10. Кластеризация\n")
                c = d.get("cluster")
                if not c:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Изображений:** {c['count']}")
                    L.append(f"- **Кластеров:** {c['n_clusters']}")
                    L.append(f"- **Шум:** {c['noise']}")
                    if c.get("sizes"):
                        L.append("\n| Кластер | Изображений |")
                        L.append("|---|---:|")
                        for cl, sz in c["sizes"][:15]:
                            name = "шум" if cl == -1 else str(cl)
                            L.append(f"| {name} | {sz} |")
                    L.append("")
                    if imgs.get("cluster"):
                        L.append(f"![Кластеры]({imgs['cluster']})\n")
            if s.get("training_results"):
                L.append("## 11. Результаты обучения модели\n")
                tr = d.get("training_results")
                if not tr or tr.get("error"):
                    err = tr.get("error") if tr else "—"
                    L.append(f"_Не выполнялось или ошибка: {err}_\n")
                else:
                    L.append(f"- **Folder:** `{tr['folder']}`")
                    bm = tr.get("best_model")
                    bmm = tr.get("best_model_meta")
                    L.append(
                        f"- **Best model:** `{(os.path.basename(bm) if bm else '—')}`"
                    )
                    L.append(
                        f"- **Best meta:** `{(os.path.basename(bmm) if bmm else '—')}`"
                    )
                    L.append("")
                    if imgs.get("training_results_png"):
                        L.append(
                            f"### results.png\n![results.png]({imgs['training_results_png']})\n"
                        )
                    if imgs.get("training_labels"):
                        L.append(
                            f"### labels.jpg\n![labels.jpg]({imgs['training_labels']})\n"
                        )
                    if imgs.get("training_confusion"):
                        L.append(
                            f"### confusion_matrix_normalized.png\n![confusion_matrix_normalized.png]({imgs['training_confusion']})\n"
                        )
                    if tr.get("curves"):
                        L.append("### Дополнительные графики\n")
                        for name in sorted(tr["curves"].keys()):
                            rel = imgs.get(f"train_curve_{name}")
                            if rel:
                                L.append(f"![{name}]({rel})\n")
                            else:
                                L.append(f"- `{name}`\n")
                        L.append("")
                    if tr.get("batches"):
                        L.append("### Батчи обучения / валидации\n")
                        L.append("| Файл |")
                        L.append("|---|")
                        for name, _ in tr["batches"][:50]:
                            L.append(f"| {name} |")
                        if len(tr["batches"]) > 50:
                            L.append(f"| … ещё {len(tr['batches']) - 50} |")
                        L.append("")
                    if tr.get("metrics"):
                        L.append("### Метрики и аргументы\n")
                        csv_path = tr["metrics"].get("results.csv")
                        if csv_path and os.path.isfile(csv_path):
                            try:
                                with open(csv_path, "r", encoding="utf-8") as f:
                                    csv_content = f.read()
                                L.append(
                                    "<details>\n<summary>📊 results.csv (полное содержимое)</summary>\n"
                                )
                                L.append("")
                                L.append("```csv")
                                L.append(csv_content.rstrip())
                                L.append("```")
                                L.append("")
                                L.append("</details>\n")
                            except Exception as e:
                                L.append(f"- `results.csv` (не удалось прочитать: {e})")
                        else:
                            L.append("- `results.csv` (не найден)")
                        yaml_path = tr["metrics"].get("args.yaml")
                        if yaml_path and os.path.isfile(yaml_path):
                            try:
                                with open(yaml_path, "r", encoding="utf-8") as f:
                                    yaml_content = f.read()
                                L.append(
                                    "<details>\n<summary>⚙️ args.yaml (параметры обучения)</summary>\n"
                                )
                                L.append("")
                                L.append("```yaml")
                                L.append(yaml_content.rstrip())
                                L.append("```")
                                L.append("")
                                L.append("</details>\n")
                            except Exception as e:
                                L.append(f"- `args.yaml` (не удалось прочитать: {e})")
                        else:
                            L.append("- `args.yaml` (не найден)")
                        L.append("")
                    if tr.get("weights"):
                        L.append("### Веса\n")
                        for name in sorted(tr["weights"].keys()):
                            L.append(f"- `weights/{name}`")
                        L.append("")
                    meta_path = tr.get("best_model_meta")
                    if meta_path and os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta_content = f.read()
                            L.append(
                                "<details>\n<summary>🧠 best_meta.json (метаданные модели)</summary>\n"
                            )
                            L.append("")
                            L.append("```json")
                            L.append(meta_content.rstrip())
                            L.append("```")
                            L.append("")
                            L.append("</details>\n")
                        except Exception as e:
                            L.append(f"- `best_meta.json` (не удалось прочитать: {e})")
                    elif tr.get("best_model_meta"):
                        L.append("- `best_meta.json` (не найден)")
                    if imgs.get("best_model"):
                        L.append(f"- **Скопирован best.pt:** `{imgs['best_model']}`")
                    if imgs.get("best_model_meta"):
                        L.append(
                            f"- **Скопирован best_meta.json:** `{imgs['best_model_meta']}`"
                        )
                    L.append("")
            L.append("---\n")
            L.append("_Отчёт сформирован автоматически вкладкой «Анализ датасета»._\n")
            if self._cancel:
                return {"error": None, "cancelled": True, "path": None}
            write_text_atomic(self.save_path, "\n".join(L))
            return {"error": None, "path": self.save_path}
        except Exception as e:
            return {"error": str(e), "path": None}


class YoloValidationJob:
    def __init__(
        self,
        model_path,
        data_path,
        split="val",
        imgsz=640,
        conf=0.001,
        iou=0.6,
        device=None,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.split = split
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device or detect_device()
        self._stop = False
        self.log = lambda message: None
        self.progress = lambda *args: None

    def request_stop(self):
        self._stop = True

    def _extract_metrics(self, metrics):
        """Извлекает summary + per-class из объекта DetMetrics."""
        box = getattr(metrics, "box", None)
        names = getattr(metrics, "names", {}) or {}
        summary = {}
        per_class = []
        if box is None:
            return {"summary": summary, "per_class": per_class}

        def _safe(obj, attr, default=0.0):
            try:
                v = getattr(obj, attr, None)
                if v is None:
                    return default
                if hasattr(v, "__len__") and (not isinstance(v, str)):
                    return float(np.mean(v)) if len(v) > 0 else default
                return float(v)
            except Exception:
                return default

        summary["Precision"] = _safe(box, "mp")
        summary["Recall"] = _safe(box, "mr")
        summary["mAP50"] = _safe(box, "map50")
        summary["mAP50-95"] = _safe(box, "map")
        summary["mAP75"] = _safe(box, "map75")
        try:
            f1_arr = getattr(box, "f1", None)
            if f1_arr is not None and len(f1_arr) > 0:
                summary["F1"] = float(np.mean(f1_arr))
        except Exception:
            pass
        if "F1" not in summary:
            p, r = (summary["Precision"], summary["Recall"])
            summary["F1"] = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
        try:
            ap_class_index = getattr(box, "ap_class_index", None)
            if ap_class_index is None:
                return {"summary": summary, "per_class": per_class}
            p_arr = getattr(box, "p", None)
            r_arr = getattr(box, "r", None)
            f1_arr = getattr(box, "f1", None)
            ap50_arr = getattr(box, "ap50", None)
            ap_arr = getattr(box, "ap", None)

            def _v(arr, i, fmt="{:.4f}"):
                if arr is None or i >= len(arr):
                    return ""
                try:
                    return fmt.format(float(arr[i]))
                except Exception:
                    return ""

            for i, ci in enumerate(ap_class_index):
                ci = int(ci)
                if isinstance(names, dict):
                    name = names.get(ci, str(ci))
                elif isinstance(names, (list, tuple)) and ci < len(names):
                    name = names[ci]
                else:
                    name = str(ci)
                per_class.append(
                    {
                        "class": f"{ci}: {name}",
                        "p": _v(p_arr, i),
                        "r": _v(r_arr, i),
                        "f1": _v(f1_arr, i),
                        "ap50": _v(ap50_arr, i),
                        "ap": _v(ap_arr, i),
                    }
                )
        except Exception as e:
            self.log(f"[VAL] Ошибка per-class: {e}")
        return {"summary": summary, "per_class": per_class}

    def run(self):
        try:
            from runtime_policy import configure_ultralytics

            configure_ultralytics()
            from utils_new.import_export.dataset_sources import enable_ultralytics_sources, metadata
            if metadata(self.data_path):
                enable_ultralytics_sources()
            from runtime_torch import require_torch

            require_torch()
            from utils_new.import_export.model_load import load_yolo as YOLO
            from ultralytics.utils import LOGGER
        except (ImportError, OSError, RuntimeError) as exc:
            return {"error": f"PyTorch/Ultralytics недоступен: {exc}", "metrics": None}
        import logging

        class _CallbackLoggerHandler(logging.Handler):
            """Пересылает записи Ultralytics-LOGGER в callback задачи."""

            def __init__(self, emit_fn):
                super().__init__()
                self._emit = emit_fn

            def emit(self, record):
                try:
                    msg = self.format(record).strip()
                    if msg:
                        self._emit(msg)
                except Exception:
                    pass

        handler = _CallbackLoggerHandler(lambda s: self.log(f"[VAL] {s}"))
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)
        try:
            self.log(f"[VAL] Загрузка модели: {self.model_path}")
            model = YOLO(self.model_path)
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"data.yaml не найден: {self.data_path}")
            self.log(
                f"[VAL] Модель загружена. Старт валидации: data={self.data_path}, split={self.split}, imgsz={self.imgsz}, conf={self.conf}, iou={self.iou}, device={self.device}"
            )
            try:

                def _stop_cb(*args, **kwargs):
                    if self._stop:
                        raise SystemExit("__VAL_CANCEL__")

                model.add_callback("on_val_batch_end", _stop_cb)
            except Exception as e:
                self.log(f"[VAL] Не удалось добавить callback отмены: {e}")
            try:
                metrics = model.val(
                    data=self.data_path,
                    split=self.split,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=True,
                )
            except Exception as e:
                msg = str(e).lower()
                if self.device != "cpu" and any(
                    (
                        k in msg
                        for k in ("cuda", "out of memory", "device", "cublas", "cudnn")
                    )
                ):
                    self.log(f"[VAL] Ошибка на {self.device} ({e}), повтор на cpu")
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                    self.device = "cpu"
                    metrics = model.val(
                        data=self.data_path,
                        split=self.split,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device="cpu",
                        verbose=True,
                    )
                else:
                    raise
        except SystemExit as e:
            if str(e) == "__VAL_CANCEL__":
                return {"error": "Валидация отменена пользователем", "metrics": None}
            else:
                return {"error": f"Валидация прервана: {e}", "metrics": None}
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"[VAL][ERROR] {e}")
            self.log(tb)
            return {"error": str(e), "metrics": None}
        finally:
            try:
                LOGGER.removeHandler(handler)
            except Exception:
                pass
        if self._stop:
            return {"error": "Валидация отменена пользователем", "metrics": None}
        try:
            data = self._extract_metrics(metrics)
            return {"error": None, "metrics": data}
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"[VAL][ERROR] extract: {e}")
            self.log(tb)
            return {"error": str(e), "metrics": None}
