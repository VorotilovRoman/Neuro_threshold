# utils/dataset_audit_logic.py
"""
Продвинутый аудит датасета:
  • точные и почти-дубликаты (MD5 + pHash);
  • утечка между train/val/test;
  • тепловая карта распределения объектов (сетка 8…128);
  • Loss Inspection (YOLO: подозрительная разметка);
  • кластеризация эмбеддингов (CLIP + UMAP + HDBSCAN);
  • фоновая сборка Markdown-отчёта.

Базовая статистика (площади, классы, дисбаланс, разрешение, яркость) —
в utils/dataset_analysis_logic.py.
"""
from import_libs_internal import *
from PIL import Image  # noqa: F401

import hashlib
import os
from collections import defaultdict, Counter

import numpy as np
import cv2


try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
    _CLIP_BACKEND = 'open_clip'
except ImportError:
    try:
        from transformers import CLIPModel, CLIPProcessor
        CLIP_AVAILABLE = True
        _CLIP_BACKEND = 'transformers'
    except ImportError:
        CLIP_AVAILABLE = False
        _CLIP_BACKEND = None

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ============================================================
# Утилиты (локальные — анализ не зависит от audit, audit не зависит от анализа)
# ============================================================
def _safe_progress(progress_cb, i, total, step=25):
    if progress_cb and ((i + 1) % step == 0 or i == total - 1):
        progress_cb(i + 1, total)


def _check_cancel(cancel_check):
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _detect_device():
    """Автоопределение устройства: 'cuda:0', если доступно, иначе 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 'cuda:0'
    except Exception:
        pass
    return 'cpu'


# ============================================================
# 1. Дубликаты (точные + почти)
# ============================================================
class DuplicateDetector:
    @staticmethod
    def _file_md5(path, chunk_size=1 << 20):
        h = hashlib.md5()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _phash(path, hash_size=8):
        if not IMAGEHASH_AVAILABLE:
            return None
        try:
            with Image.open(path) as img:
                return imagehash.phash(img.convert('RGB'),
                                        hash_size=hash_size)
        except Exception:
            return None

    @staticmethod
    def find_duplicates(pairs, phash_threshold=8, progress_cb=None,
                        cancel_check=None):
        total = len(pairs)
        md5_map = defaultdict(list)
        phash_map = {}
        cancelled = False

        for i, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            img_path = pair[0]
            if os.path.exists(img_path):
                md5 = DuplicateDetector._file_md5(img_path)
                if md5:
                    md5_map[md5].append(img_path)
                ph = DuplicateDetector._phash(img_path)
                if ph is not None:
                    phash_map[img_path] = ph
            _safe_progress(progress_cb, i, total)

        exact_groups = [g for g in md5_map.values() if len(g) > 1]

        near_groups = []
        if phash_map and not cancelled:
            paths = list(phash_map.keys())
            used = set()
            for i, p1 in enumerate(paths):
                if _check_cancel(cancel_check):
                    cancelled = True
                    break
                if p1 in used:
                    continue
                group = [p1]
                h1 = phash_map[p1]
                for p2 in paths[i + 1:]:
                    if p2 in used:
                        continue
                    if (h1 - phash_map[p2]) <= phash_threshold:
                        group.append(p2)
                        used.add(p2)
                if len(group) > 1:
                    near_groups.append(group)
                    used.update(group)

        return {
            'exact': exact_groups,
            'near': near_groups,
            'cancelled': cancelled,
        }


# ============================================================
# 2. Утечка между сплитами
# ============================================================
class LeakageChecker:
    SPLIT_NAMES = {'train', 'val', 'valid', 'validation', 'test'}

    @staticmethod
    def _infer_split(path):
        parts = path.replace('\\', '/').lower().split('/')
        for part in reversed(parts):
            for name in LeakageChecker.SPLIT_NAMES:
                if name in part:
                    if name in ('val', 'valid', 'validation'):
                        return 'val'
                    return name
        return 'unknown'

    @staticmethod
    def check(duplicate_groups, pairs):
        split_of = {pair[0]: LeakageChecker._infer_split(pair[0])
                    for pair in pairs}

        result_groups = []
        leaked_count = 0
        for group in duplicate_groups:
            splits = Counter(split_of.get(p, 'unknown') for p in group)
            known = {s for s in splits if s != 'unknown'}
            leak = len(known) >= 2
            if leak:
                leaked_count += 1
            result_groups.append({
                'paths': list(group),
                'splits': dict(splits),
                'leak': leak,
            })

        return {'groups': result_groups, 'leaked_count': leaked_count}


# ============================================================
# 3. Тепловая карта
# ============================================================
class ObjectDistributionHeatmap:
    @staticmethod
    def compute(pairs, dataset_type=0, grid_size=16,
                progress_cb=None, cancel_check=None):
        grid_size = max(2, int(grid_size))
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        total_objs = 0
        total = len(pairs)
        cancelled = False

        for i, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                if w <= 0 or h <= 0:
                    _safe_progress(progress_cb, i, total)
                    continue

                anns = []
                if dataset_type == 0 and lbl_path:
                    try:
                        anns = load_annotations(lbl_path, 1, 1)
                    except Exception:
                        anns = []
                elif dataset_type == 1 and lbl_path:
                    mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        for v in np.unique(mask):
                            if v == 0:
                                continue
                            ys, xs = np.where(mask == v)
                            if len(xs) == 0:
                                continue
                            anns.append(('mask', v,
                                         xs.mean() / mask.shape[1],
                                         ys.mean() / mask.shape[0],
                                         0, 0))

                for ann in anns:
                    typ = ann[0]
                    if typ == 'detect':
                        cx, cy = ann[2], ann[3]
                    elif typ in ('obb', 'segment'):
                        pts = ann[2]
                        xs = pts[0::2]
                        ys = pts[1::2]
                        cx = sum(xs) / len(xs) if xs else 0
                        cy = sum(ys) / len(ys) if ys else 0
                    else:
                        cx, cy = ann[2], ann[3]

                    gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                    gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                    heatmap[gy, gx] += 1
                    total_objs += 1
            except Exception:
                pass
            _safe_progress(progress_cb, i, total)

        return {
            'heatmap': heatmap,
            'total_objects': total_objs,
            'grid_size': grid_size,
            'cancelled': cancelled,
        }


# ============================================================
# 4. Loss Inspection (YOLO)
# ============================================================
class LossInspectionThread(QThread):
    progress = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, pairs, model_path, dataset_type=0,
                 iou_threshold=0.3, conf_threshold=0.25, imgsz=640,
                 device=None, input_format='auto', parent=None):
        super().__init__(parent)
        self.pairs = list(pairs)
        self.model_path = model_path
        self.dataset_type = dataset_type
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device or _detect_device()
        self.input_format = input_format
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @staticmethod
    def _iou_xyxy(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        u = aa + bb - inter
        return inter / u if u > 0 else 0.0

    def run(self):
        self.log_signal.emit(
            f"[LossInspection] старт (device: {self.device})")
        try:
            from ultralytics import YOLO
        except ImportError:
            self.finished_signal.emit({
                'error': 'ultralytics не установлен', 'suspicious': []})
            return

        try:
            model = YOLO(self.model_path)
            # Пытаемся перевести на GPU, иначе остаёмся на CPU
            if self.device.startswith('cuda'):
                try:
                    model.to(self.device)
                except Exception as e:
                    self.log_signal.emit(
                        f"[LossInspection] Не удалось на {self.device} "
                        f"({e}), fallback на cpu")
                    self.device = 'cpu'
            self.log_signal.emit(
                f"[LossInspection] модель загружена "
                f"(device: {self.device})")
        except Exception as e:
            self.finished_signal.emit({
                'error': f'Не удалось загрузить модель: {e}',
                'suspicious': []})
            return

        suspicious = []
        total = len(self.pairs)
        processed = 0
        errors = 0

        for i, pair in enumerate(self.pairs):
            if self._cancel:
                self.log_signal.emit("[LossInspection] отменено")
                break

            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None

            if not lbl_path or not os.path.exists(lbl_path):
                _safe_progress(self.progress.emit, i, total, step=10)
                continue

            try:
                results = model(img_path,
                                conf=self.conf_threshold,
                                imgsz=self.imgsz,
                                device=self.device,
                                verbose=False)
                if not results:
                    _safe_progress(self.progress.emit, i, total, step=10)
                    continue

                r = results[0]
                gt_anns = load_annotations(lbl_path, 1, 1,
                                            format_hint=self.input_format)
                processed += 1

                boxes = getattr(r, 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    if gt_anns:
                        suspicious.append({
                            'img_path': img_path,
                            'lbl_path': lbl_path,
                            'reason':
                                'Модель не нашла объектов, а GT не пуст',
                            'pred_count': 0,
                            'gt_count': len(gt_anns),
                            'mean_iou': 0.0,
                            'min_iou': 0.0,
                            'min_conf': 0.0,
                        })
                    _safe_progress(self.progress.emit, i, total, step=10)
                    continue

                pred_xyxy = boxes.xyxy.cpu().numpy()
                pred_conf = boxes.conf.cpu().numpy()

                h, w = (r.orig_shape
                        if hasattr(r, 'orig_shape') else (0, 0))
                if h == 0 or w == 0:
                    with Image.open(img_path) as im:
                        w, h = im.size

                gt_xyxy = []
                for ann in gt_anns:
                    if ann[0] == 'detect':
                        _, cls, cx, cy, bw, bh = ann
                        gt_xyxy.append([
                            (cx - bw / 2) * w,
                            (cy - bh / 2) * h,
                            (cx + bw / 2) * w,
                            (cy + bh / 2) * h,
                        ])

                if not gt_xyxy:
                    _safe_progress(self.progress.emit, i, total, step=10)
                    continue

                ious = []
                for g in gt_xyxy:
                    best = max(
                        (self._iou_xyxy(g, p) for p in pred_xyxy),
                        default=0.0)
                    ious.append(best)

                mean_iou = float(np.mean(ious))
                min_iou = float(np.min(ious))

                if mean_iou < self.iou_threshold or min_iou < 0.05:
                    suspicious.append({
                        'img_path': img_path,
                        'lbl_path': lbl_path,
                        'reason': (f'mean IoU={mean_iou:.3f} '
                                   f'(min={min_iou:.3f})'),
                        'pred_count': len(pred_xyxy),
                        'gt_count': len(gt_xyxy),
                        'mean_iou': mean_iou,
                        'min_iou': min_iou,
                        'min_conf': (float(pred_conf.min())
                                     if len(pred_conf) else 0.0),
                    })
            except Exception as e:
                errors += 1
                self.log_signal.emit(
                    f"[LossInspection] ошибка {img_path}: {e}")

            _safe_progress(self.progress.emit, i, total, step=10)

        self.log_signal.emit(
            f"[LossInspection] готово: processed={processed}/{total}, "
            f"errors={errors}, подозрительных={len(suspicious)}")
        self.finished_signal.emit({
            'error': None,
            'suspicious': suspicious,
            'processed': processed,
            'total': total,
            'errors': errors,
        })


# ============================================================
# 5. Кластеризация эмбеддингов (CLIP + UMAP + HDBSCAN)
# ============================================================
class EmbeddingClusterThread(QThread):
    progress = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, pairs, batch_size=16, device=None, parent=None):
        super().__init__(parent)
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.device = device or _detect_device()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        self.log_signal.emit(
            f"[Cluster] старт (backend: {_CLIP_BACKEND}, "
            f"device: {self.device})")
        if not CLIP_AVAILABLE:
            self.finished_signal.emit({
                'error': 'CLIP не установлен.',
                'clusters': []})
            return

        try:
            if _CLIP_BACKEND == 'open_clip':
                model, _, preprocess = \
                    open_clip.create_model_and_transforms(
                        'ViT-B-32', pretrained='laion2b_s34b_b79k')
            else:
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32")

            # Пытаемся перевести на GPU, иначе fallback на CPU
            try:
                model = model.to(self.device).eval()
            except Exception as e:
                self.log_signal.emit(
                    f"[Cluster] Не удалось на {self.device} ({e}), "
                    f"fallback на cpu")
                self.device = 'cpu'
                model = model.to(self.device).eval()

            self.log_signal.emit(
                f"[Cluster] CLIP загружен (device: {self.device})")
        except Exception as e:
            self.finished_signal.emit({
                'error': f'Не удалось загрузить CLIP: {e}',
                'clusters': []})
            return

        embeddings = []
        paths = []
        batch_imgs = []
        total = len(self.pairs)

        import torch
        from PIL import Image as _Img

        def flush_batch(batch_paths):
            if not batch_paths:
                return
            try:
                if _CLIP_BACKEND == 'open_clip':
                    imgs = []
                    for p in batch_paths:
                        with _Img.open(p) as im:
                            imgs.append(preprocess(im.convert('RGB')))
                    batch = torch.stack(imgs).to(self.device)
                    with torch.no_grad():
                        feats = model.encode_image(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.extend(feats.cpu().numpy())
                else:
                    imgs = []
                    for p in batch_paths:
                        with _Img.open(p) as im:
                            imgs.append(processor(
                                images=im.convert('RGB'),
                                return_tensors='pt'
                            )['pixel_values'][0])
                    batch = torch.stack(imgs).to(self.device)
                    with torch.no_grad():
                        feats = model.get_image_features(pixel_values=batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.extend(feats.cpu().numpy())
                paths.extend(batch_paths)
            except Exception as e:
                self.log_signal.emit(f"[Cluster] ошибка батча: {e}")

        for i, pair in enumerate(self.pairs):
            if self._cancel:
                self.log_signal.emit("[Cluster] отменено")
                break
            if os.path.exists(pair[0]):
                batch_imgs.append(pair[0])
            if len(batch_imgs) >= self.batch_size or i == total - 1:
                flush_batch(batch_imgs)
                batch_imgs = []
            _safe_progress(self.progress.emit, i, total, step=10)

        if not embeddings:
            self.finished_signal.emit({
                'error': 'Не удалось извлечь ни одного эмбеддинга.',
                'clusters': []})
            return

        emb = np.array(embeddings)

        if UMAP_AVAILABLE:
            try:
                emb_2d = umap.UMAP(
                    n_neighbors=15, min_dist=0.1,
                    n_components=2, metric='cosine',
                    random_state=42).fit_transform(emb)
                self.log_signal.emit("[Cluster] UMAP выполнен")
            except Exception as e:
                self.log_signal.emit(
                    f"[Cluster] UMAP ошибка: {e}, PCA")
                from sklearn.decomposition import PCA
                emb_2d = PCA(n_components=2).fit_transform(emb)
        else:
            from sklearn.decomposition import PCA
            emb_2d = PCA(n_components=2).fit_transform(emb)
            self.log_signal.emit("[Cluster] PCA вместо UMAP")

        if HDBSCAN_AVAILABLE:
            try:
                labels = hdbscan.HDBSCAN(
                    min_cluster_size=5, min_samples=3,
                    metric='euclidean').fit_predict(emb_2d)
                clusters = labels.tolist()
                self.log_signal.emit("[Cluster] HDBSCAN выполнен")
            except Exception as e:
                self.log_signal.emit(f"[Cluster] HDBSCAN ошибка: {e}")
                clusters = [-1] * len(emb_2d)
        else:
            from sklearn.cluster import DBSCAN
            clusters = DBSCAN(
                eps=0.5, min_samples=3).fit_predict(emb_2d).tolist()
            self.log_signal.emit("[Cluster] DBSCAN вместо HDBSCAN")

        self.finished_signal.emit({
            'error': None,
            'paths': paths,
            'embeddings_2d': emb_2d.tolist(),
            'clusters': clusters,
        })


# ============================================================
# 6. Фоновая сборка Markdown-отчёта
# ============================================================
class ReportThread(QThread):
    finished_signal = pyqtSignal(dict)

    def __init__(self, data, save_path, sections, parent=None):
        super().__init__(parent)
        self.data = data
        self.save_path = save_path
        self.sections = sections

    def run(self):
        try:
            from datetime import datetime

            d = self.data
            s = self.sections
            imgs = d.get('images', {})
            L = []
            L.append("# Dataset Analysis Report\n")
            L.append(f"**Сгенерирован:** "
                     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 1. Общее
            gi = d.get('general', {})
            L.append("## 1. Общая информация\n")
            for k, v in gi.items():
                L.append(f"- **{k}:** {v}")
            L.append("")

            # 2. Параметры фильтрации
            fp = d.get('filter_params')
            if fp:
                L.append("## 2. Параметры фильтрации\n")
                L.append(f"- **min_w:** {fp.get('min_w')} px")
                L.append(f"- **min_h:** {fp.get('min_h')} px")
                L.append(f"- **min_objs:** {fp.get('min_objs')}")
                L.append(f"- **max_objs:** {fp.get('max_objs')} "
                         f"(0 = без ограничений)")
                L.append(f"- **min_area:** {fp.get('min_area')} px²")
                L.append(f"- **max_area:** {fp.get('max_area')} px² "
                         f"(0 = без ограничений)")
                L.append(f"- **include_empty_annotations:** "
                         f"{fp.get('include_empty')}")
                L.append(f"- **exclude_classes_from_files:** "
                         f"{fp.get('exclude_classes_from_files')}")
                L.append(f"- **All class IDs:** "
                         f"{fp.get('all_class_ids')}")
                L.append(f"- **Included class IDs:** "
                         f"{fp.get('included_class_ids')}")
                L.append(f"- **Excluded class IDs:** "
                         f"{fp.get('excluded_class_ids')}")
                L.append("")

            # 3. Классы и площади
            cs = d.get('class_stats', [])
            if cs:
                L.append("## 3. Классы и распределение площадей\n")
                L.append("| Класс | Объектов | Доля |")
                L.append("|---|---:|---:|")
                for row in cs:
                    L.append(f"| {row[0]} | {row[1]} | {row[2]} |")
                L.append("")
            if imgs.get('areas'):
                L.append(f"![Распределение площадей]({imgs['areas']})\n")
            if imgs.get('classes'):
                L.append(f"![Распределение по классам]({imgs['classes']})\n")

            # 4. Дисбаланс
            imb = d.get('imbalance')
            if imb:
                L.append("## 4. Дисбаланс классов\n")
                L.append(f"- **Imbalance ratio:** "
                         f"{imb['imbalance_ratio']:.2f}:1")
                L.append(f"- **Gini:** {imb['gini']:.4f}")
                L.append(f"- **Entropy:** {imb['entropy']:.4f}")
                L.append(f"- **Majority:** {imb['majority_class']}")
                L.append(f"- **Minority:** {imb['minority_class']}")
                L.append(f"- **Оценка:** {imb['warning']}")
                L.append("")

            # 5. Качество изображений
            if s.get('quality'):
                L.append("## 5. Качество изображений\n")
                q = d.get('quality')
                if not q:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Median resolution:** "
                             f"{q['median_w']:.0f}×{q['median_h']:.0f}")
                    L.append(f"- **Recommended imgsz:** "
                             f"{q['recommended_imgsz']}")
                    br = q.get('brightness')
                    if br:
                        L.append(f"- **Mean brightness:** "
                                 f"{br['mean_brightness']:.1f}")
                        L.append(f"- **Dark images (<40):** "
                                 f"{br['dark_count']}")
                        L.append(f"- **Bright images (>215):** "
                                 f"{br['bright_count']}")
                    L.append("")
                    if imgs.get('quality'):
                        L.append(f"![Качество]({imgs['quality']})\n")

            # 6. Дубликаты и утечка
            if s.get('dup'):
                L.append("## 6. Дубликаты и утечка\n")
                dp = d.get('dup')
                if not dp:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Точные (MD5):** "
                             f"{dp['exact_count']} групп "
                             f"({dp['exact_images']} изображений)")
                    L.append(f"- **Почти (pHash):** "
                             f"{dp['near_count']} групп "
                             f"({dp['near_images']} изображений)")
                    L.append(f"- **Утечка:** {dp['leaked_count']} групп")
                    if dp.get('leaked_groups'):
                        L.append("\n**Утёкшие группы:**\n")
                        for rg in dp['leaked_groups']:
                            L.append(f"- {rg}")
                    L.append("")

            # 7. Тепловая карта
            if s.get('heatmap'):
                L.append("## 7. Тепловая карта\n")
                hm = d.get('heatmap')
                if not hm:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Объектов:** {hm['total_objects']}")
                    L.append(f"- **Сетка:** {hm['grid']}×{hm['grid']}")
                    L.append("- **Топ-3 зоны:**")
                    for zone in hm.get('top_zones', []):
                        L.append(f"  - {zone}")
                    L.append("")
                    if imgs.get('heatmap'):
                        L.append(f"![Тепловая карта]"
                                 f"({imgs['heatmap']})\n")

            # 8. Аудит модели
            if s.get('audit'):
                L.append("## 8. Аудит модели (Loss Inspection)\n")
                m = d.get('audit_meta')
                if not m:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Model:** `{m['model']}`")
                    L.append(f"- **IoU threshold:** {m['iou_thr']}")
                    L.append(f"- **Conf threshold:** {m['conf_thr']}")
                    L.append(f"- **imgsz:** {m.get('imgsz', '—')}")
                    L.append(f"- **Processed:** "
                             f"{m['processed']}/{m['total']}")
                    L.append(f"- **Errors:** {m['errors']}")
                    L.append(f"- **Suspicious:** "
                             f"{m['suspicious_count']}")
                    rows = d.get('audit_rows', [])
                    if rows:
                        L.append("\n| Изображение | GT | Pred | Mean IoU |")
                        L.append("|---|---:|---:|---:|")
                        for r in rows:
                            L.append(f"| {r[0]} | {r[1]} | {r[2]} | "
                                     f"{r[3]:.3f} |")
                    L.append("")

            # 9. Кластеризация
            if s.get('cluster'):
                L.append("## 9. Кластеризация\n")
                c = d.get('cluster')
                if not c:
                    L.append("_Не выполнялось._\n")
                else:
                    L.append(f"- **Изображений:** {c['count']}")
                    L.append(f"- **Кластеров:** {c['n_clusters']}")
                    L.append(f"- **Шум:** {c['noise']}")
                    if c.get('sizes'):
                        L.append("\n| Кластер | Изображений |")
                        L.append("|---|---:|")
                        for cl, sz in c['sizes'][:15]:
                            name = "шум" if cl == -1 else str(cl)
                            L.append(f"| {name} | {sz} |")
                    L.append("")
                    if imgs.get('cluster'):
                        L.append(f"![Кластеры]"
                                 f"({imgs['cluster']})\n")

            L.append("---\n")
            L.append("_Отчёт сформирован автоматически вкладкой "
                     "«Анализ датасета»._\n")

            with open(self.save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(L))

            self.finished_signal.emit({
                'error': None,
                'path': self.save_path,
            })
        except Exception as e:
            self.finished_signal.emit({
                'error': str(e),
                'path': None,
            })