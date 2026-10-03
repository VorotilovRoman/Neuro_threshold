from import_libs_external import *

import math

# ============================================================
# Жёсткие пределы — защита от падения OpenCV/Qt/libtiff
# на очень больших изображениях (0xC0000409 и т.п.).
# Применяются ВСЕГДА, независимо от галочки «resize».
# ============================================================
HARD_MAX_SIDE = 16384        # абсолютный предел стороны (px) при чтении

# Практический предел для QPixmap/GDI+ на Windows. При превышении
# QPixmap.fromImage может уронить процесс нативным исключением
# (0xC0000409), которое нельзя поймать try/except.
MAX_QPIXMAP_SIDE = 8192


# ============================================================
# OBB: конвертация между двумя форматами представления
# ============================================================
# Внутри приложения OBB всегда хранится как 4 угла (8 нормализованных
# чисел) — так проще рисовать (cv2.polylines), делать hit-testing
# (point-in-polygon) и редактировать (перетаскивание вершин).
#
# Файл на диске может быть в одном из двух форматов:
#   1) class_index x1 y1 x2 y2 x3 y3 x4 y4        (9 токенов)
#   2) class_index cx cy w h angle_rad            (6 токенов, xywhr)
#
# Канонизация xywhr (соглашение Ultralytics YOLO):
#   • w — длинная сторона (long-edge form), w >= h;
#   • angle — радианы, отсчитывается от +x до направления стороны w,
#     значение нормализовано в [-π/4, 3π/4);
#   • угол и его поворот на 180° эквивалентны (бокс без направления).
# ============================================================

def _xywhr_to_corners(cx, cy, w, h, angle_rad):
    """
    (cx, cy, w, h, angle_rad) → 8 нормализованных чисел (4 угла).

    Порядок углов — обход по периметру (как отдаёт cv2.boxPoints):
        p1 = ctr + v1 + v2
        p2 = ctr + v1 - v2
        p3 = ctr - v1 - v2
        p4 = ctr - v1 + v2
    где
        v1 = (w/2 * cos θ,  w/2 * sin θ)   — вдоль длинной стороны w
        v2 = (-h/2 * sin θ, h/2 * cos θ)   — вдоль короткой стороны h
    """
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
    """
    8 нормализованных чисел (4 угла) → (cx, cy, w, h, angle_rad).

    Приводит бокс к канонической long-edge форме YOLO:
        w >= h, angle ∈ [-π/4, 3π/4).
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    # Рёбра (p_{i+1} - p_i)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        edges.append((p2[0] - p1[0], p2[1] - p1[1]))

    lengths = [math.hypot(dx, dy) for dx, dy in edges]
    angles = [math.atan2(dy, dx) for dx, dy in edges]

    # Длинное ребро — это w и ориентация бокса
    longest_idx = max(range(4), key=lambda i: lengths[i])
    w = lengths[longest_idx]
    angle_rad = angles[longest_idx]

    # Короткая сторона — усредняем две перпендикулярные стороны
    perp1 = (longest_idx + 1) % 4
    perp2 = (longest_idx + 3) % 4
    h = (lengths[perp1] + lengths[perp2]) / 2.0

    # Канонизация: w >= h
    if h > w:
        w, h = h, w
        angle_rad += math.pi / 2.0

    # Нормализация угла в [-π/4, 3π/4)
    angle_rad = angle_rad % math.pi
    if angle_rad >= 3 * math.pi / 4:
        angle_rad -= math.pi

    return cx, cy, w, h, angle_rad


def canonicalize_obb_corners(norm_pts):
    """
    Приводит 8 нормализованных чисел (4 угла OBB) к каноническому порядку,
    совпадающему с smart_view._obb_to_points:

        p0 — «нижний» угол (максимум y; при равенстве — максимум x),
        p1 — следующий по периметру,
        p2, p3 — далее по периметру.

    Правила:
      1) сортировка по полярному углу от центра даёт периметрический обход;
      2) первое ребро — самое длинное (w >= h);
      3) середина первого ребра имеет не меньший y, чем середина
         противоположного (сторона p0—p1 «ниже»);
      4) обход — с положительным векторным произведением
         cross(p1-p0, p2-p1) > 0.

    Функция устойчива к любому входному порядку, включая «бабочку».

    Применяется везде, где OBB приходит из внешнего источника:
      • load_annotations (для обоих форматов: 4 угла и xywhr);
      • YOLO auto-label в layout_dataset.
    """
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

    # 1) Периметрический обход.
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    def _dist(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    # 2) Длинное ребро — первым.
    if _dist(pts[0], pts[1]) < _dist(pts[1], pts[2]) - 1e-12:
        pts = pts[1:] + pts[:1]

    # 3) Первое ребро — «нижнее» (больший y у середины).
    mid_a_y = (pts[0][1] + pts[1][1]) / 2.0
    mid_b_y = (pts[2][1] + pts[3][1]) / 2.0
    if mid_a_y < mid_b_y - 1e-12:
        pts = [pts[2], pts[3], pts[0], pts[1]]

    # 4) Ориентация обхода (на случай, если сортировка дала CCW).
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

# ---------- Базовые функции работы с изображениями ----------
def _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE):
    """
    Читает изображение через PIL с уменьшением на этапе декодирования.
    Для TIFF `draft()` заставляет libtiff декодировать сразу в уменьшенном
    масштабе — полное изображение в память не попадает.
    """
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pil_img = Image.open(image_path)
        try:
            pil_img.draft("RGB", (max_side, max_side))
        except Exception:
            pass  # draft поддерживается не всеми форматами
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
    """Читает TIFF через tifffile — устойчиво к приватным тегам libtiff."""
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
    """
    Загружает изображение с защитой от падения нативных библиотек.

    Порядок для TIFF:
        1) PIL (с draft для очень больших);
        2) tifffile — устойчив к приватным тегам libtiff
           (напр. tag 65100 0xfe4c), которые роняют OpenCV;
        3) OpenCV — в самом конце.

    Для остальных форматов:
        1) при очень большом размере — PIL draft;
        2) cv2.imread;
        3) PIL без draft.
    """
    ext = os.path.splitext(image_path)[1].lower()
    is_tiff = ext in ('.tif', '.tiff')

    # --- 1. Смотрим размеры БЕЗ полного декодирования (только заголовок) ---
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

    # --- 2. TIFF: PIL → tifffile → (только потом) OpenCV ---
    if is_tiff:
        if need_draft:
            print(f"[read] {os.path.basename(image_path)}: очень большой TIFF, "
                  f"читаю через PIL draft")
            img = _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)
            if img is not None:
                return img

        # Обычный TIFF — сначала PIL (без draft)
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

        # TIFF через tifffile — обходит кривой libtiff-парсер OpenCV
        img = _read_with_tifffile(image_path)
        if img is not None:
            return img

        # Если ни PIL, ни tifffile не справились — пробуем OpenCV
        # (последний шанс, может упасть нативным исключением).
        img = cv2.imread(image_path)
        if img is not None:
            return img
        return None

    # --- 3. Не-TIFF: очень большой файл — грузим через PIL draft сразу ---
    if need_draft:
        print(f"[read] {os.path.basename(image_path)}: очень большой, "
              f"читаю через PIL draft")
        img = _read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)
        if img is not None:
            return img

    # --- 4. Обычный путь через OpenCV ---
    img = cv2.imread(image_path)
    if img is not None:
        return img

    # --- 5. Fallback на PIL без draft ---
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


def resize_to_max_side(img, max_side=1024):
    """Уменьшает изображение, чтобы длинная сторона стала max_side (пропорционально)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_to_uint8(img):
    """Приводит изображение к типу uint8 (0..255)."""
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
    """Преобразует BGR или RGB изображение в оттенки серого."""
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
    """
    Преобразует numpy-изображение (BGR или grayscale) в QPixmap.

    Аварийный даунскейл: QPixmap.fromImage на Windows может уронить
    процесс нативным исключением (0xC0000409) при больших размерах
    (>~8K по стороне и/или нескольких ГБ суммарной памяти на pixmap).
    Ограничиваем сторону до max_side ДО создания QImage.
    """
    if img_bgr is None:
        return QPixmap()

    # --- Аварийный даунскейл для защиты от крэша QPixmap.fromImage ---
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
        # BGR888 доступен с Qt 5.14 — экономим целую копию cvtColor.
        qimage = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        h, w, ch = img_bgr.shape
        qimage = QImage(img_bgr.data, w, h, ch * w, QImage.Format_ARGB32)
    else:
        # На всякий случай — fallback на прежний путь.
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # QPixmap.fromImage копирует в нативный формат, поэтому
    # numpy-буфер должен быть жив только на время вызова — так и есть.
    return QPixmap.fromImage(qimage)


# ============================================================
# Работа с аннотациями
# ============================================================
# Внутренний формат (единый для всего приложения):
#     ('detect',  cls:int, cx, cy, w, h)                  # 4 норм. числа
#     ('obb',     cls:int, [x1,y1,x2,y2,x3,y3,x4,y4])     # 8 норм. чисел
#     ('segment', cls:int, [x1,y1,x2,y2,...])             # чётное ≥6
#
# На диске OBB может быть в одном из двух вариантов:
#     class_index x1 y1 x2 y2 x3 y3 x4 y4        (как есть)
#     class_index cx cy w h angle_rad            (xywhr, конвертируется
#                                                 в 4 угла при загрузке)
# ============================================================

def load_annotations(txt_path, img_w, img_h):
    """
    Загружает аннотации из YOLO .txt файла.

    Возвращает список кортежей:
        ('detect', class_id, cx, cy, w, h)              # 5 чисел
        ('obb', class_id, [x1,y1,x2,y2,x3,y3,x4,y4])    # 9 чисел (4 угла)
        ('obb', class_id, [x1,y1,x2,y2,x3,y3,x4,y4])    # 6 чисел (xywhr) —
                                                        # конвертируется в 4 угла
        ('segment', class_id, [x1,y1,x2,y2,...])        # ≥6 чётное число координат

    Параметры img_w, img_h сохранены для обратной совместимости сигнатуры;
    координаты в файле уже нормализованы и не требуют домножения.
    """
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

                # --- Разбор по числу токенов ---
                if n == 5:
                    # class + 4 числа → detect (cx cy w h)
                    cx, cy, w, h = coords
                    annotations.append(('detect', cls, cx, cy, w, h))

                elif n == 6:
                    # class + 5 чисел → OBB в формате xywhr.
                    # Конвертируем в 4 угла: внутри приложения всегда 4 угла.
                    cx, cy, w, h, angle = coords
                    corners = _xywhr_to_corners(cx, cy, w, h, angle)
                    corners = canonicalize_obb_corners(corners)   # ← добавить
                    annotations.append(('obb', cls, corners))

                elif n == 9:
                    # class + 8 чисел → OBB в виде 4 углов (текущий формат)
                    coords = canonicalize_obb_corners(coords)      # ← добавить
                    annotations.append(('obb', cls, coords))

                elif n >= 7 and (n - 1) % 2 == 0:
                    # polygon (segment): чётное число координат ≥ 6
                    annotations.append(('segment', cls, coords))

                else:
                    print(f"Warning: unknown annotation format in "
                          f"{txt_path} line {line_num} (length {n})")
    except Exception as e:
        print(f"Error loading annotations {txt_path}: {e}")
    return annotations


def save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False):
    """
    Сохраняет аннотации в YOLO формате.

    Поддерживает типы 'detect', 'obb', 'segment'.

    Параметр obb_as_xywhr управляет форматом записи OBB:
        False (по умолчанию) — 4 угла:
            class_index x1 y1 x2 y2 x3 y3 x4 y4
        True — канонический xywhr:
            class_index cx cy w h angle_rad
    Внутреннее представление всегда 4 угла — конвертация выполняется
    на границе записи.

    Параметры img_w, img_h сохранены для обратной совместимости сигнатуры;
    координаты уже нормализованы.
    """
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


def load_annotations_obb(txt_path, img_w, img_h):
    """
    Загружает только OBB-аннотации из .txt файла.
    Всегда возвращает 4 угла (8 нормализованных чисел) независимо от того,
    в каком формате файл был на диске (4 угла или xywhr).
    """
    all_anns = load_annotations(txt_path, img_w, img_h)
    return [(cls, pts) for typ, cls, pts in all_anns if typ == 'obb']


def load_annotations_for_image(img_path, img_w, img_h):
    """Загружает аннотации, если существует соответствующий .txt файл."""
    base = os.path.splitext(img_path)[0]
    txt_path = base + ".txt"
    return load_annotations(txt_path, img_w, img_h)


# ---------- Универсальная загрузка изображений ----------
def load_images_universal(source, require_annotations=False, resize_enabled=True,
                          max_side=1024, progress_callback=None, parent=None):
    """
    Универсальная загрузка изображений и аннотаций.
    Returns: (image_paths, images, gray_images, annotations_list)
    """
    # Определяем список файлов
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

        # --- HARD LIMIT: до любых cvtColor/resize ---
        h0, w0 = img_original.shape[:2]
        if max(h0, w0) > HARD_MAX_SIDE:
            print(f"[load] {os.path.basename(path)}: {w0}x{h0} "
                  f"> {HARD_MAX_SIDE}, downscaling")
            img_original = resize_to_max_side(img_original, max_side=HARD_MAX_SIDE)

        if resize_enabled:
            img = resize_to_max_side(img_original, max_side=max_side)
        else:
            img = img_original

        # Приводим к BGR (3 канала)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            img = img[:, :, :3]

        gray = convert_to_grayscale(img)
        ann = load_annotations(txt_path, img.shape[1], img.shape[0])

        image_paths.append(path)
        images.append(img)
        gray_images.append(gray)
        annotations_list.append(ann)

    if use_internal_progress:
        progress.close()

    return image_paths, images, gray_images, annotations_list


# ========== Загрузка датасета из YAML ==========
def load_dataset_from_yaml(yaml_path, resize_enabled=True, max_side=1024, progress_callback=None, parent=None):
    """
    Загружает датасет из YAML-файла YOLO (train/val/test).
    Объединяет все найденные изображения из всех секций.

    Параметры:
        yaml_path: путь к .yaml/.yml файлу
        resize_enabled: булево, уменьшать ли большие изображения
        max_side: максимальная сторона при ресайзе
        progress_callback: функция (idx, total) -> bool для отмены
        parent: QWidget для прогресс-диалога

    Возвращает:
        (image_paths, images, gray_images, annotations_list) как в load_images_universal
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', os.path.dirname(yaml_path))
    base_path = os.path.abspath(base_path)

    items = []  # (img_path, label_path)

    for split in ['train', 'val', 'test']:
        split_val = config.get(split)
        if not split_val:
            continue

        # split_val может быть строкой (путь к папке с изображениями) или списком таких строк
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

            # Получаем все изображения в папке
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                    continue
                img_path = os.path.join(img_dir, fname)

                # Поиск label-файла (стандартное расположение YOLO)
                label_path = None
                stem = os.path.splitext(fname)[0]
                # Пробуем стандартные пути: labels/внутри_сплита, labels/сама_папка, labels/на_уровне_с_изображениями
                possible_label_dirs = [
                    os.path.join(base_path, 'labels', split),           # labels/train
                    os.path.join(base_path, 'labels', os.path.basename(rel_dir)),  # labels/images_subfolder
                    os.path.join(base_path, 'labels'),                  # labels/
                    os.path.join(os.path.dirname(img_dir), 'labels'),   # папка labels рядом с images
                ]
                for lbl_dir in possible_label_dirs:
                    candidate = os.path.join(lbl_dir, stem + '.txt')
                    if os.path.exists(candidate):
                        label_path = candidate
                        break

                items.append((img_path, label_path))

    if not items:
        return [], [], [], []

    # Извлекаем списки
    img_paths, label_paths = zip(*items) if items else ([], [])

    # Загружаем изображения с помощью load_images_universal, но аннотации будем загружать отдельно
    paths, imgs, grays, _ = load_images_universal(
        source=list(img_paths),
        require_annotations=False,
        resize_enabled=resize_enabled,
        max_side=max_side,
        progress_callback=progress_callback,
        parent=parent
    )

    if not paths:
        return [], [], [], []

    # Собираем аннотации (используем label_paths, которые нашли выше)
    annotations_list = []
    for i, path in enumerate(paths):
        # Находим соответствующий label_path
        label_path = None
        for orig_path, lbl in zip(img_paths, label_paths):
            if os.path.samefile(path, orig_path):
                label_path = lbl
                break
        if label_path and os.path.exists(label_path):
            img_h, img_w = imgs[i].shape[:2]
            ann = load_annotations(label_path, img_w, img_h)
        else:
            ann = []
        annotations_list.append(ann)

    return paths, imgs, grays, annotations_list


# ---------- Остальные утилиты ----------
def save_coordinates(main_window):
    """Сохраняет координаты выделенных объектов в .txt файл (только detect)."""
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
    """Возвращает список пар (image_path, label_path)."""
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


def collect_unique_classes_from_labels(label_paths):
    """Возвращает отсортированный список уникальных class id из списка label-файлов."""
    unique_classes = set()
    for label_path in label_paths:
        anns = load_annotations(label_path, img_w=1, img_h=1)
        for ann in anns:
            if ann[0] == 'detect':
                _, cls, _, _, _, _ = ann
                unique_classes.add(cls)
            elif ann[0] in ('obb', 'segment'):
                _, cls, _ = ann
                unique_classes.add(cls)
    return sorted(unique_classes)


def read_image_with_fallback_find(image_path):
    """Загружает изображение с помощью OpenCV, при ошибке пробует через PIL и tifffile."""
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


def convert_segment_masks_to_yolo_seg_manual(masks_dir: str, output_dir: str, pixel_to_class: dict):
    """Convert segmentation masks to YOLO format."""
    from pathlib import Path

    masks_path = Path(masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for mask_file in masks_path.glob("*"):
        if mask_file.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
            continue

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Cannot read {mask_file}")
            continue

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.squeeze()
        if mask.ndim != 2:
            print(f"  Unexpected shape {mask.shape} for {mask_file}, skipping")
            continue

        h, w = mask.shape
        yolo_lines = []

        for pixel_val, class_id in pixel_to_class.items():
            binary = (mask == pixel_val).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def load_dataset_from_yaml_with_masks(yaml_path, resize_enabled=True, max_side=1024,
                                       progress_callback=None, parent=None):
    """
    Загружает датасет из YAML-файла YOLO, включая маски (если они есть).
    Возвращает:
        image_paths, images, gray_images, annotations_list, mask_paths_list, mask_images_list
    mask_paths_list — список путей к файлам масок (если маска не найдена, элемент None)
    mask_images_list — список загруженных масок (numpy array) или None
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
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                    continue
                img_path = os.path.join(img_dir, fname)

                # Поиск аннотации
                stem = os.path.splitext(fname)[0]
                label_path = None
                for lbl_sub in [split, '']:
                    lbl_dir = os.path.join(base_path, 'labels', lbl_sub) if lbl_sub else os.path.join(base_path, 'labels')
                    candidate = os.path.join(lbl_dir, stem + '.txt')
                    if os.path.exists(candidate):
                        label_path = candidate
                        break

                # Поиск маски
                mask_path = None
                for mask_sub in [split, '']:
                    mask_dir = os.path.join(base_path, 'masks', mask_sub) if mask_sub else os.path.join(base_path, 'masks')
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
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

    # Загружаем изображения (без аннотаций, так как мы подставим свои)
    paths, imgs, grays, _ = load_images_universal(
        source=img_paths,
        require_annotations=False,
        resize_enabled=resize_enabled,
        max_side=max_side,
        progress_callback=progress_callback,
        parent=parent
    )

    if not paths:
        return [], [], [], [], [], []

    # Сопоставляем пути изображений с label_paths и mask_paths
    img_to_label = {os.path.normpath(p): lp for p, lp in zip(img_paths, label_paths)}
    img_to_mask = {os.path.normpath(p): mp for p, mp in zip(img_paths, mask_paths)}

    annotations_list = []
    mask_images_list = []
    for path in paths:
        label_path = img_to_label.get(os.path.normpath(path))
        if label_path and os.path.exists(label_path):
            h, w = imgs[annotations_list.__len__()].shape[:2]
            ann = load_annotations(label_path, w, h)
        else:
            ann = []
        annotations_list.append(ann)

        mask_path = img_to_mask.get(os.path.normpath(path))
        mask_img = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None and len(mask_img.shape) == 3:
                    mask_img = mask_img[:, :, 0]
            except Exception as e:
                print(f"Warning: could not load mask {mask_path}: {e}")
        mask_images_list.append(mask_img)

    return paths, imgs, grays, annotations_list, mask_paths, mask_images_list


# ============================================================
# Ленивая загрузка изображений (LRU-кэш с фоновым префетчем)
# ============================================================
def load_one_image_item(image_path, resize_enabled=True, max_side=1024,
                        safe_max_side=None, load_annotations_flag=True):
    """
    Загружает одно изображение и подготавливает его для отображения/инференса.

    Параметры:
        image_path            — путь к файлу;
        resize_enabled        — уменьшать ли длинную сторону до max_side;
        max_side              — значение для обычного ресайза (по галочке);
        safe_max_side         — жёсткий предел стороны, применяется всегда
                                (аналог MAX_SAFE_SIDE в окне вкладки);
        load_annotations_flag — читать ли .txt рядом с изображением.

    Возвращает dict:
        {'image': BGR uint8, 'gray': gray uint8, 'annotations': [...], 'path': str}
    или None, если изображение не читается.
    """
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
            anns = load_annotations(txt_path, img.shape[1], img.shape[0])

    return {"image": img, "gray": gray, "annotations": anns, "path": image_path}


class LazyImageCache:
    """
    LRU-кэш изображений с фоновой предзагрузкой окна вокруг текущего индекса.

    • Хранит не более max_resident готовых элементов.
    • Для текущего индекса get() синхронно возвращает готовый элемент
      (или ждёт уже запущенную фоновую задачу для того же индекса).
    • prefetch() ставит загрузку в ThreadPoolExecutor — UI не блокируется.
    • trim_around() удаляет из кэша всё, что дальше заданного окна.
    • Ошибки загрузки не глотаются молча: они сохраняются в _errors[idx]
      и поднимаются как исключение из get() либо доступны через get_error().

    Потокобезопасен: доступ к внутренним структурам защищён RLock.
    """
    def __init__(self, paths, loader, window=8, prefetch_ahead=4,
                 max_resident=16, max_workers=2):
        self.paths = list(paths)
        self.loader = loader
        self.window = window
        self.prefetch_ahead = prefetch_ahead
        self.max_resident = max_resident
        self._cache = OrderedDict()            # idx -> item
        self._errors = {}                      # idx -> Exception
        self._lock = threading.RLock()
        self._in_flight = set()                # idx, для которых запущена загрузка
        self._events = {}                      # idx -> threading.Event
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="img-load")
        self._shutdown = False

    # --- базовое ---
    def __len__(self):
        return len(self.paths)

    def _trim_locked(self):
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)

    def _load_sync(self, idx):
        """Синхронная загрузка. Ошибка сохраняется и пере-поднимается."""
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
                # Лоадер вернул None — трактуем как ошибку чтения.
                self._errors[idx] = IOError(f"Loader вернул None для {self.paths[idx]}")
        return item

    def is_loaded(self, idx):
        with self._lock:
            return idx in self._cache

    def get_error(self, idx):
        """Последняя ошибка загрузки для индекса (или None)."""
        with self._lock:
            return self._errors.get(idx)

    def get(self, idx):
        """
        Возвращает элемент по индексу; при отсутствии — загружает синхронно.

        Если загрузка падает, исключение поднимается наружу. Если фоновая
        задача уже упала — возвращает None, но исходная ошибка доступна
        через get_error(idx).
        """
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
                    # Фоновая задача уже провалилась — повторно не пытаемся,
                    # возвращаем None (ошибка доступна через get_error).
                    return None
        return self._load_sync(idx)

    # --- фоновый префетч ---
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
        """Ставит в очередь фоновую загрузку указанных индексов."""
        if self._shutdown:
            return
        for idx in indices:
            if idx < 0 or idx >= len(self.paths):
                continue
            with self._lock:
                if idx in self._cache or idx in self._in_flight:
                    continue
                # Не пытаемся снова, если уже была ошибка — иначе будем
                # бесконечно перезапускать сбойную загрузку.
                if idx in self._errors:
                    continue
                self._in_flight.add(idx)
                self._events[idx] = threading.Event()
            self._pool.submit(self._load_bg, idx)

    def trim_around(self, center, keep_behind=3, keep_ahead=None):
        """Удаляет из кэша всё, что дальше заданного окна вокруг center."""
        if keep_ahead is None:
            keep_ahead = self.prefetch_ahead + 4
        with self._lock:
            lo = max(0, center - keep_behind)
            hi = center + keep_ahead
            to_remove = [i for i in self._cache.keys() if not (lo <= i <= hi)]
            for i in to_remove:
                del self._cache[i]
            # Заодно чистим устаревшие ошибки вне окна
            err_remove = [i for i in self._errors.keys() if not (lo <= i <= hi)]
            for i in err_remove:
                del self._errors[i]

    # --- остановка ---
    def shutdown(self):
        """Останавливает пул потоков и очищает кэш."""
        self._shutdown = True
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9 — нет cancel_futures
            self._pool.shutdown(wait=False)
        with self._lock:
            self._cache.clear()
            self._errors.clear()
            self._in_flight.clear()
            for ev in self._events.values():
                ev.set()
            self._events.clear()


# Алиасы для обратной совместимости
load_images = load_images_universal
load_images_from_paths = load_images_universal
load_folder = load_images_universal