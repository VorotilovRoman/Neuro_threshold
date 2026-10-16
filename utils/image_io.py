# utils/image_io.py
from import_libs_external import *

import math

# ============================================================
# Жёсткие пределы
# ============================================================
HARD_MAX_SIDE = 16384
MAX_QPIXMAP_SIDE = 8192


# ============================================================
# OBB: конвертация между форматами
# ============================================================
def _xywhr_to_corners(cx, cy, w, h, angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = w / 2.0
    dy = h / 2.0
    v1x, v1y = dx * cos_a, dx * sin_a
    v2x, v2y = -dy * sin_a, dy * cos_a
    return [
        cx + v1x + v2x, cy + v1y + v2y,
        cx + v1x - v2x, cy + v1y - v2y,
        cx - v1x - v2x, cy - v1y - v2y,
        cx - v1x + v2x, cy - v1y + v2y,
    ]


def _corners_to_xywhr(corners):
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        edges.append((p2[0] - p1[0], p2[1] - p1[1]))

    lengths = [math.hypot(dx, dy) for dx, dy in edges]
    angles = [math.atan2(dy, dx) for dx, dy in edges]

    longest_idx = max(range(4), key=lambda i: lengths[i])
    w = lengths[longest_idx]
    angle_rad = angles[longest_idx]

    perp1 = (longest_idx + 1) % 4
    perp2 = (longest_idx + 3) % 4
    h = (lengths[perp1] + lengths[perp2]) / 2.0

    if h > w:
        w, h = h, w
        angle_rad += math.pi / 2.0

    angle_rad = angle_rad % math.pi
    if angle_rad >= 3 * math.pi / 4:
        angle_rad -= math.pi

    return cx, cy, w, h, angle_rad


def canonicalize_obb_corners(norm_pts):
    if len(norm_pts) != 8:
        return list(norm_pts)
    try:
        pts = [(float(norm_pts[i]), float(norm_pts[i + 1]))
               for i in range(0, 8, 2)]
    except (TypeError, ValueError):
        return list(norm_pts)
    if len(pts) != 4:
        return list(norm_pts)

    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    def _dist(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    if _dist(pts[0], pts[1]) < _dist(pts[1], pts[2]) - 1e-12:
        pts = pts[1:] + pts[:1]

    mid_a_y = (pts[0][1] + pts[1][1]) / 2.0
    mid_b_y = (pts[2][1] + pts[3][1]) / 2.0
    if mid_a_y < mid_b_y - 1e-12:
        pts = [pts[2], pts[3], pts[0], pts[1]]

    p0, p1, p2, p3 = pts
    v1x, v1y = p1[0] - p0[0], p1[1] - p0[1]
    v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
    if v1x * v2y - v1y * v2x < 0:
        pts = [p0, p3, p2, p1]

    out = []
    for (x, y) in pts:
        out.append(float(x))
        out.append(float(y))
    return out


# ---------- Чтение изображений ----------
def _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE):
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pil_img = Image.open(image_path)
        try:
            pil_img.draft("RGB", (max_side, max_side))
        except Exception:
            pass
        img = np.array(pil_img)
        if img.ndim == 2:
            return img
        if img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"[read] PIL draft failed for {image_path}: {e}")
        return None


def _read_with_tifffile(image_path):
    try:
        import tifffile
    except ImportError:
        return None
    try:
        arr = tifffile.imread(image_path)
        if arr is None:
            return None
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            if arr.shape[-1] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception as e:
        print(f"[read] tifffile failed for {image_path}: {e}")
        return None


def read_image_with_fallback(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    is_tiff = ext in ('.tif', '.tiff')

    need_draft = False
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as probe:
            w, h = probe.size
            if max(w, h) > HARD_MAX_SIDE:
                need_draft = True
    except Exception:
        pass

    if is_tiff:
        if need_draft:
            print(f"[read] {os.path.basename(image_path)}: очень большой TIFF, "
                  f"читаю через PIL draft")
            img = _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)
            if img is not None:
                return img

        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            if img.ndim == 2:
                return img
            if img.shape[-1] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            if img.shape[-1] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"[read] PIL failed for TIFF {image_path}: {e}")

        img = _read_with_tifffile(image_path)
        if img is not None:
            return img

        img = cv2.imread(image_path)
        if img is not None:
            return img
        return None

    if need_draft:
        print(f"[read] {os.path.basename(image_path)}: очень большой, "
              f"читаю через PIL draft")
        img = _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)
        if img is not None:
            return img

    img = cv2.imread(image_path)
    if img is not None:
        return img

    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
        if img.ndim == 2:
            return img
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def read_mask_with_fallback(mask_path):
    """
    Читает маску, сохраняя её dtype (uint8/uint16).
    НЕ приводит к uint8.
    """
    ext = os.path.splitext(mask_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            pil_img = Image.open(mask_path)
            arr = np.array(pil_img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr.squeeze()
        except Exception:
            pass
        try:
            import tifffile
            arr = tifffile.imread(mask_path)
            if arr is not None:
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                return arr.squeeze()
        except Exception:
            pass

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.squeeze()


def resize_to_max_side(img, max_side=1024):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
        else:
            return img.astype(np.uint8)
    return img.astype(np.uint8)


def convert_to_grayscale(img):
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def numpy_to_qpixmap(img_bgr, max_side=MAX_QPIXMAP_SIDE):
    if img_bgr is None:
        return QPixmap()

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

    if not img_bgr.flags['C_CONTIGUOUS']:
        img_bgr = np.ascontiguousarray(img_bgr)

    if img_bgr.ndim == 2:
        h, w = img_bgr.shape
        qimage = QImage(img_bgr.data, w, h, w, QImage.Format_Grayscale8)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        h, w, ch = img_bgr.shape
        qimage = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        h, w, ch = img_bgr.shape
        qimage = QImage(img_bgr.data, w, h, ch * w, QImage.Format_ARGB32)
    else:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


# ============================================================
# Аннотации: внутренний формат и (де)сериализация
# ============================================================
FORMAT_HINT_AUTO         = 'auto'
FORMAT_HINT_MIXED        = 'mixed'
FORMAT_HINT_DETECT       = 'detect'
FORMAT_HINT_OBB_CORNERS  = 'obb_corners'
FORMAT_HINT_OBB_XYWHR    = 'obb_xywhr'
FORMAT_HINT_SEGMENT      = 'segment'

_VALID_FORMAT_HINTS = {
    FORMAT_HINT_AUTO, FORMAT_HINT_MIXED, FORMAT_HINT_DETECT,
    FORMAT_HINT_OBB_CORNERS, FORMAT_HINT_OBB_XYWHR, FORMAT_HINT_SEGMENT,
}


def _classify_line(n_parts, format_hint):
    if format_hint not in _VALID_FORMAT_HINTS:
        format_hint = FORMAT_HINT_AUTO

    if format_hint == FORMAT_HINT_DETECT:
        return 'detect' if n_parts == 5 else None
    if format_hint == FORMAT_HINT_OBB_CORNERS:
        return 'obb_corners' if n_parts == 9 else None
    if format_hint == FORMAT_HINT_OBB_XYWHR:
        return 'obb_xywhr' if n_parts == 6 else None
    if format_hint == FORMAT_HINT_SEGMENT:
        if n_parts >= 7 and (n_parts - 1) % 2 == 0:
            return 'segment'
        return None

    if n_parts == 5:
        return 'detect'
    if n_parts == 6:
        return 'obb_xywhr'
    if n_parts == 9:
        return 'obb_corners'
    if n_parts >= 7 and (n_parts - 1) % 2 == 0:
        return 'segment'
    return None


def load_annotations(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO):
    annotations = []
    if not os.path.exists(txt_path):
        return annotations

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 1:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    print(f"Warning: invalid class id in {txt_path} line {line_num}")
                    continue

                coords = []
                invalid = False
                for p in parts[1:]:
                    try:
                        coords.append(float(p))
                    except ValueError:
                        invalid = True
                        break
                if invalid or len(coords) == 0:
                    print(f"Warning: invalid coordinates in {txt_path} line {line_num}")
                    continue

                n = len(parts)
                kind = _classify_line(n, format_hint)
                if kind is None:
                    print(f"Warning: line {line_num} in {txt_path} "
                          f"({n} tokens) does not match format_hint="
                          f"{format_hint!r}; skipped")
                    continue

                if kind == 'detect':
                    cx, cy, w, h = coords
                    annotations.append(('detect', cls, cx, cy, w, h))
                elif kind == 'obb_xywhr':
                    cx, cy, w, h, angle = coords
                    corners = _xywhr_to_corners(cx, cy, w, h, angle)
                    corners = canonicalize_obb_corners(corners)
                    annotations.append(('obb', cls, corners))
                elif kind == 'obb_corners':
                    corners = canonicalize_obb_corners(coords)
                    annotations.append(('obb', cls, corners))
                elif kind == 'segment':
                    annotations.append(('segment', cls, coords))

    except Exception as e:
        print(f"Error loading annotations {txt_path}: {e}")
    return annotations


def save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False):
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                if ann[0] == 'detect':
                    _, cls, cx, cy, w, h = ann
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                elif ann[0] == 'obb':
                    _, cls, points = ann
                    if obb_as_xywhr:
                        cx, cy, w, h, angle = _corners_to_xywhr(points)
                        f.write(f"{cls} {cx:.6f} {cy:.6f} "
                                f"{w:.6f} {h:.6f} {angle:.6f}\n")
                    else:
                        line = f"{cls} " + " ".join(f"{p:.6f}" for p in points)
                        f.write(line + "\n")
                elif ann[0] == 'segment':
                    _, cls, points = ann
                    line = f"{cls} " + " ".join(f"{p:.6f}" for p in points)
                    f.write(line + "\n")
                else:
                    print(f"Warning: unknown annotation type {ann[0]} skipped")
        return True
    except Exception as e:
        print(f"Error saving annotations {txt_path}: {e}")
        return False


def load_annotations_obb(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO):
    all_anns = load_annotations(txt_path, img_w, img_h, format_hint=format_hint)
    return [(cls, pts) for typ, cls, pts in all_anns if typ == 'obb']


def load_annotations_for_image(img_path, img_w, img_h,
                                format_hint=FORMAT_HINT_AUTO):
    base = os.path.splitext(img_path)[0]
    txt_path = base + ".txt"
    return load_annotations(txt_path, img_w, img_h, format_hint=format_hint)


# ---------- Универсальная загрузка изображений ----------
def load_images_universal(source, require_annotations=False, resize_enabled=True,
                          max_side=1024, progress_callback=None, parent=None,
                          format_hint=FORMAT_HINT_AUTO):
    if isinstance(source, str) and os.path.isdir(source):
        all_files = []
        for f in os.listdir(source):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
                all_files.append(os.path.join(source, f))
    elif isinstance(source, list):
        all_files = source
    else:
        raise ValueError("source must be a folder path or a list of file paths")

    total = len(all_files)
    image_paths = []
    images = []
    gray_images = []
    annotations_list = []

    use_internal_progress = (progress_callback is None and parent is not None)
    progress = None
    if use_internal_progress:
        progress = QProgressDialog("Загрузка изображений...", "Отмена", 0, total, parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

    for idx, path in enumerate(all_files):
        if use_internal_progress:
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"Загрузка {os.path.basename(path)}...")
        elif progress_callback is not None:
            if not progress_callback(idx, total):
                break

        txt_path = os.path.splitext(path)[0] + ".txt"
        if require_annotations and not os.path.exists(txt_path):
            continue

        img_original = read_image_with_fallback(path)
        if img_original is None:
            continue

        img_original = normalize_to_uint8(img_original)

        h0, w0 = img_original.shape[:2]
        if max(h0, w0) > HARD_MAX_SIDE:
            print(f"[load] {os.path.basename(path)}: {w0}x{h0} "
                  f"> {HARD_MAX_SIDE}, downscaling")
            img_original = resize_to_max_side(img_original, max_side=HARD_MAX_SIDE)

        if resize_enabled:
            img = resize_to_max_side(img_original, max_side=max_side)
        else:
            img = img_original

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            img = img[:, :, :3]

        gray = convert_to_grayscale(img)
        ann = load_annotations(txt_path, img.shape[1], img.shape[0],
                               format_hint=format_hint)

        image_paths.append(path)
        images.append(img)
        gray_images.append(gray)
        annotations_list.append(ann)

    if use_internal_progress:
        progress.close()

    return image_paths, images, gray_images, annotations_list


# ============================================================
# Загрузка датасета из YAML (YOLO)
# ============================================================
def load_dataset_from_yaml(yaml_path, resize_enabled=True, max_side=1024,
                            progress_callback=None, parent=None,
                            format_hint=FORMAT_HINT_AUTO):
    """
    Загружает YOLO-датасет из data.yaml.
    Возвращает (image_paths, images, gray_images, annotations_list).
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', os.path.dirname(yaml_path))
    base_path = os.path.abspath(base_path)

    items = []

    for split in ['train', 'val', 'test']:
        split_val = config.get(split)
        if not split_val:
            continue
        if isinstance(split_val, str):
            split_dirs = [split_val]
        elif isinstance(split_val, list):
            split_dirs = split_val
        else:
            continue

        for rel_dir in split_dirs:
            img_dir = os.path.join(base_path, rel_dir)
            if not os.path.isdir(img_dir):
                continue

            for fname in os.listdir(img_dir):
                if not fname.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                    continue
                img_path = os.path.join(img_dir, fname)

                label_path = None
                stem = os.path.splitext(fname)[0]
                possible_label_dirs = [
                    os.path.join(base_path, 'labels', split),
                    os.path.join(base_path, 'labels', os.path.basename(rel_dir)),
                    os.path.join(base_path, 'labels'),
                    os.path.join(os.path.dirname(img_dir), 'labels'),
                ]
                for lbl_dir in possible_label_dirs:
                    candidate = os.path.join(lbl_dir, stem + '.txt')
                    if os.path.exists(candidate):
                        label_path = candidate
                        break

                items.append((img_path, label_path))

    if not items:
        return [], [], [], []

    img_paths, label_paths = zip(*items) if items else ([], [])

    paths, imgs, grays, _ = load_images_universal(
        source=list(img_paths),
        require_annotations=False,
        resize_enabled=resize_enabled,
        max_side=max_side,
        progress_callback=progress_callback,
        parent=parent,
        format_hint=format_hint,
    )

    if not paths:
        return [], [], [], []

    annotations_list = []
    for i, path in enumerate(paths):
        label_path = None
        for orig_path, lbl in zip(img_paths, label_paths):
            if os.path.samefile(path, orig_path):
                label_path = lbl
                break
        if label_path and os.path.exists(label_path):
            img_h, img_w = imgs[i].shape[:2]
            ann = load_annotations(label_path, img_w, img_h,
                                    format_hint=format_hint)
        else:
            ann = []
        annotations_list.append(ann)

    return paths, imgs, grays, annotations_list


# ============================================================
# Загрузка датасета из YAML с масками (U-Net)
# ============================================================
def load_dataset_from_yaml_with_masks(yaml_path, resize_enabled=True,
                                       max_side=1024, progress_callback=None,
                                       parent=None):
    """
    Загружает YAML с масками.
    Возвращает:
        image_paths, images, gray_images, annotations_list,
        mask_paths_list, mask_images_list
    Маски читаются через read_mask_with_fallback — dtype сохраняется.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', '')
    if not base_path:
        base_path = os.path.dirname(yaml_path)
    base_path = os.path.abspath(base_path)

    items = []  # (img_path, label_path, mask_path)

    for split in ['train', 'val', 'test']:
        split_val = config.get(split)
        if not split_val:
            continue
        if isinstance(split_val, str):
            split_dirs = [split_val]
        elif isinstance(split_val, list):
            split_dirs = split_val
        else:
            continue

        for rel_dir in split_dirs:
            img_dir = os.path.join(base_path, rel_dir)
            if not os.path.isdir(img_dir):
                continue

            for fname in os.listdir(img_dir):
                if not fname.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                    continue
                img_path = os.path.join(img_dir, fname)

                stem = os.path.splitext(fname)[0]
                label_path = None
                for lbl_sub in [split, '']:
                    lbl_dir = (os.path.join(base_path, 'labels', lbl_sub)
                               if lbl_sub else os.path.join(base_path, 'labels'))
                    candidate = os.path.join(lbl_dir, stem + '.txt')
                    if os.path.exists(candidate):
                        label_path = candidate
                        break

                mask_path = None
                for mask_sub in [split, '']:
                    mask_dir = (os.path.join(base_path, 'masks', mask_sub)
                                if mask_sub else os.path.join(base_path, 'masks'))
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp',
                                '.tif', '.tiff']:
                        candidate = os.path.join(mask_dir, stem + ext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                    if mask_path:
                        break

                items.append((img_path, label_path, mask_path))

    if not items:
        return [], [], [], [], [], []

    img_paths, label_paths, mask_paths = zip(*items) if items else ([], [], [])
    img_paths = list(img_paths)
    label_paths = list(label_paths)
    mask_paths = list(mask_paths)

    paths, imgs, grays, _ = load_images_universal(
        source=img_paths,
        require_annotations=False,
        resize_enabled=resize_enabled,
        max_side=max_side,
        progress_callback=progress_callback,
        parent=parent,
    )

    if not paths:
        return [], [], [], [], [], []

    img_to_label = {os.path.normpath(p): lp
                    for p, lp in zip(img_paths, label_paths)}
    img_to_mask = {os.path.normpath(p): mp
                   for p, mp in zip(img_paths, mask_paths)}

    annotations_list = []
    mask_images_list = []
    for idx, path in enumerate(paths):
        label_path = img_to_label.get(os.path.normpath(path))
        if label_path and os.path.exists(label_path):
            h, w = imgs[idx].shape[:2]
            ann = load_annotations(label_path, w, h)
        else:
            ann = []
        annotations_list.append(ann)

        mask_path = img_to_mask.get(os.path.normpath(path))
        mask_img = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask_img = read_mask_with_fallback(mask_path)
            except Exception as e:
                print(f"Warning: could not load mask {mask_path}: {e}")
        mask_images_list.append(mask_img)

    return paths, imgs, grays, annotations_list, mask_paths, mask_images_list


# ---------- Прочие утилиты ----------
def save_coordinates(main_window):
    if not main_window.display_images:
        QMessageBox.warning(main_window, "No Image", "No image loaded.")
        return
    img_path = main_window.image_paths[main_window.current_index]
    txt_path = os.path.splitext(img_path)[0] + ".txt"
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for obj in main_window.current_objects_full:
                if len(obj) == 4:
                    x, y, w, h = obj
                    img_w = main_window.display_images[main_window.current_index].shape[1]
                    img_h = main_window.display_images[main_window.current_index].shape[0]
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    f.write(f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                elif len(obj) == 5:
                    cx, cy, w, h, angle = obj
                    img_w = main_window.display_images[main_window.current_index].shape[1]
                    img_h = main_window.display_images[main_window.current_index].shape[0]
                    cx_norm = cx / img_w
                    cy_norm = cy / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        main_window.log(f"Saved {len(main_window.current_objects_full)} objects to {txt_path}")
        QMessageBox.information(main_window, "Save", f"Coordinates saved to {txt_path}")
    except Exception as e:
        main_window.log(f"Error saving: {e}")
        QMessageBox.critical(main_window, "Error", f"Failed to save: {e}")


def find_image_label_pairs(images_folder, labels_folder, img_extensions=None):
    if img_extensions is None:
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    img_files = {}
    for f in os.listdir(images_folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in img_extensions:
            img_files[name] = os.path.join(images_folder, f)
    pairs = []
    for name, img_path in img_files.items():
        label_path = os.path.join(labels_folder, name + '.txt')
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    return pairs


def collect_unique_classes_from_labels(label_paths, format_hint=FORMAT_HINT_AUTO):
    unique_classes = set()
    for label_path in label_paths:
        anns = load_annotations(label_path, img_w=1, img_h=1,
                                format_hint=format_hint)
        for ann in anns:
            unique_classes.add(ann[1])
    return sorted(unique_classes)


def read_image_with_fallback_find(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = normalize_to_uint8(img)
        img = resize_to_max_side(img, max_side=640)
        return img

    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = normalize_to_uint8(img)
        img = resize_to_max_side(img, max_side=640)
        return img
    except Exception:
        pass

    try:
        import tifffile
        img = tifffile.imread(image_path)
        if img is not None:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = normalize_to_uint8(img)
            img = resize_to_max_side(img, max_side=640)
            return img
    except ImportError:
        pass
    except Exception:
        pass

    return None


def convert_segment_masks_to_yolo_seg_manual(masks_dir: str, output_dir: str,
                                              pixel_to_class: dict,
                                              mask_offset: int = 0):
    """
    Конвертация масок в YOLO-seg .txt (lossy, findContours).
    pixel_to_class: {pixel_value: yolo_class_id}
    mask_offset: прибавляется к pixel_value для поиска в pixel_to_class.
                 (0 — pixel_value как есть; 1 — U-Net-стиль.)
    """
    from pathlib import Path

    masks_path = Path(masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for mask_file in masks_path.glob("*"):
        if mask_file.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp',
                                             '.tif', '.tiff'):
            continue

        mask = read_mask_with_fallback(str(mask_file))
        if mask is None:
            print(f"  Cannot read {mask_file}")
            continue
        if mask.ndim != 2:
            print(f"  Unexpected shape {mask.shape} for {mask_file}, skipping")
            continue

        h, w = mask.shape
        yolo_lines = []

        for pixel_val, class_id in pixel_to_class.items():
            # Если задан offset — ищем по pixel_val + offset,
            # иначе — по pixel_val напрямую.
            search_val = pixel_val + int(mask_offset) if mask_offset else pixel_val
            binary = (mask == search_val).astype(np.uint8)
            if binary.sum() == 0:
                continue
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                contour = contour.squeeze().astype(np.float32)
                if contour.ndim != 2:
                    continue
                contour[:, 0] /= w
                contour[:, 1] /= h
                flat = contour.reshape(-1).tolist()
                coord_str = " ".join(f"{x:.6f}" for x in flat)
                yolo_lines.append(f"{class_id} {coord_str}")

        if yolo_lines:
            txt_path = output_path / f"{mask_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))


# ============================================================
# Ленивая загрузка
# ============================================================
def load_one_image_item(image_path, resize_enabled=True, max_side=1024,
                        safe_max_side=None, load_annotations_flag=True,
                        format_hint=FORMAT_HINT_AUTO):
    img_original = read_image_with_fallback(image_path)
    if img_original is None:
        return None
    img_original = normalize_to_uint8(img_original)

    h0, w0 = img_original.shape[:2]
    if max(h0, w0) > HARD_MAX_SIDE:
        img_original = resize_to_max_side(img_original, max_side=HARD_MAX_SIDE)

    if resize_enabled:
        img = resize_to_max_side(img_original, max_side=max_side)
    else:
        img = img_original

    if safe_max_side is not None:
        img = resize_to_max_side(img, max_side=safe_max_side)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        img = img[:, :, :3]

    gray = convert_to_grayscale(img)

    anns = []
    if load_annotations_flag:
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(txt_path):
            anns = load_annotations(txt_path, img.shape[1], img.shape[0],
                                    format_hint=format_hint)

    return {"image": img, "gray": gray, "annotations": anns, "path": image_path}


class LazyImageCache:
    def __init__(self, paths, loader, window=8, prefetch_ahead=4,
                 max_resident=16, max_workers=2):
        self.paths = list(paths)
        self.loader = loader
        self.window = window
        self.prefetch_ahead = prefetch_ahead
        self.max_resident = max_resident
        self._cache = OrderedDict()
        self._errors = {}
        self._lock = threading.RLock()
        self._in_flight = set()
        self._events = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="img-load")
        self._shutdown = False

    def __len__(self):
        return len(self.paths)

    def _trim_locked(self):
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)

    def _load_sync(self, idx):
        try:
            item = self.loader(self.paths[idx])
        except Exception as e:
            with self._lock:
                self._errors[idx] = e
            raise
        with self._lock:
            if item is not None:
                self._cache[idx] = item
                self._cache.move_to_end(idx)
                self._trim_locked()
                self._errors.pop(idx, None)
            else:
                self._errors[idx] = IOError(f"Loader вернул None для {self.paths[idx]}")
        return item

    def is_loaded(self, idx):
        with self._lock:
            return idx in self._cache

    def get_error(self, idx):
        with self._lock:
            return self._errors.get(idx)

    def get(self, idx):
        if idx < 0 or idx >= len(self.paths):
            return None
        with self._lock:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return self._cache[idx]
            ev = self._events.get(idx)
        if ev is not None:
            ev.wait()
            with self._lock:
                if idx in self._cache:
                    self._cache.move_to_end(idx)
                    return self._cache[idx]
                if idx in self._errors:
                    return None
        return self._load_sync(idx)

    def _load_bg(self, idx):
        if self._shutdown:
            return
        item = None
        error = None
        try:
            item = self.loader(self.paths[idx])
            if item is None:
                error = IOError(f"Loader вернул None для {self.paths[idx]}")
        except Exception as e:
            error = e
            print(f"[lazy] background error loading {self.paths[idx]}: {e}")
        if self._shutdown:
            return
        with self._lock:
            self._in_flight.discard(idx)
            if item is not None:
                self._cache[idx] = item
                self._cache.move_to_end(idx)
                self._trim_locked()
                self._errors.pop(idx, None)
            elif error is not None:
                self._errors[idx] = error
            ev = self._events.pop(idx, None)
        if ev is not None:
            ev.set()

    def prefetch(self, indices):
        if self._shutdown:
            return
        for idx in indices:
            if idx < 0 or idx >= len(self.paths):
                continue
            with self._lock:
                if idx in self._cache or idx in self._in_flight:
                    continue
                if idx in self._errors:
                    continue
                self._in_flight.add(idx)
                self._events[idx] = threading.Event()
            self._pool.submit(self._load_bg, idx)

    def trim_around(self, center, keep_behind=3, keep_ahead=None):
        if keep_ahead is None:
            keep_ahead = self.prefetch_ahead + 4
        with self._lock:
            lo = max(0, center - keep_behind)
            hi = center + keep_ahead
            to_remove = [i for i in self._cache.keys() if not (lo <= i <= hi)]
            for i in to_remove:
                del self._cache[i]
            err_remove = [i for i in self._errors.keys() if not (lo <= i <= hi)]
            for i in err_remove:
                del self._errors[i]

    def shutdown(self):
        self._shutdown = True
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._pool.shutdown(wait=False)
        with self._lock:
            self._cache.clear()
            self._errors.clear()
            self._in_flight.clear()
            for ev in self._events.values():
                ev.set()
            self._events.clear()


# Алиасы обратной совместимости
load_images = load_images_universal
load_images_from_paths = load_images_universal
load_folder = load_images_universal