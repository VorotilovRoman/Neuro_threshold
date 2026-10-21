# utils/dataset_validation.py
from import_libs_internal import *
from utils.smart_view import SmartGraphicsView

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# Допустимые значения input_format (совпадают с image_io.FORMAT_HINT_*)
INPUT_FORMAT_AUTO        = 'auto'
INPUT_FORMAT_MIXED       = 'mixed'
INPUT_FORMAT_DETECT      = 'detect'
INPUT_FORMAT_OBB_CORNERS = 'obb_corners'
INPUT_FORMAT_OBB_XYWHR   = 'obb_xywhr'
INPUT_FORMAT_SEGMENT     = 'segment'

_VALID_INPUT_FORMATS = {
    INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED, INPUT_FORMAT_DETECT,
    INPUT_FORMAT_OBB_CORNERS, INPUT_FORMAT_OBB_XYWHR, INPUT_FORMAT_SEGMENT,
}


class ImageStats:
    @staticmethod
    def empty_color_stats():
        return {'gray_real': 0, 'gray_visual': 0, 'color': 0, 'unknown': 0}

    @staticmethod
    def empty_bit_stats():
        return {'uint8_png': 0, 'uint8_other': 0, 'uint16': 0,
                'int16': 0, 'float': 0, 'other': 0, 'unknown': 0}

    @staticmethod
    def detect_color_type(img):
        if img is None:
            return 'unknown'
        try:
            if len(img.shape) == 2:
                return 'gray_real'
            if img.shape[2] == 1:
                return 'gray_real'
            bgr = img[:, :, :3] if img.shape[2] >= 3 else img
            b = bgr[:, :, 0]; g = bgr[:, :, 1]; r = bgr[:, :, 2]
            if np.array_equal(b, g) and np.array_equal(g, r):
                return 'gray_visual'
            return 'color'
        except Exception:
            return 'unknown'

    @staticmethod
    def detect_bit_type(img, img_path):
        if img is None:
            return 'unknown'
        try:
            ext = os.path.splitext(str(img_path))[1].lower()
            is_png = ext == '.png'
            dt = img.dtype
            if dt == np.uint8:
                return 'uint8_png' if is_png else 'uint8_other'
            if dt == np.uint16:
                return 'uint16'
            if dt == np.int16:
                return 'int16'
            if dt in (np.float32, np.float64):
                return 'float'
            return 'other'
        except Exception:
            return 'unknown'

    @staticmethod
    def to_uint8(img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img / 256.0).clip(0, 255).astype(np.uint8)
        if img.dtype in (np.float32, np.float64) and img.size > 0 \
                and img.max() <= 1.0 + 1e-6:
            return (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def to_display_bgr(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def to_gray(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 1:
            return img[:, :, 0]
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_bgr3(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def format_stats_text(color_stats, bit_stats, has_pairs):
        if not has_pairs:
            return "", False
        gr = color_stats.get('gray_real', 0)
        gv = color_stats.get('gray_visual', 0)
        c = color_stats.get('color', 0)
        u = color_stats.get('unknown', 0)
        p8 = bit_stats.get('uint8_png', 0)
        o8 = bit_stats.get('uint8_other', 0)
        b16 = bit_stats.get('uint16', 0)
        i16 = bit_stats.get('int16', 0)
        fl = bit_stats.get('float', 0)
        oth = bit_stats.get('other', 0)

        line1 = (f"⚫ Реально ЧБ: {gr}   |   "
                 f"🌫 Визуально ЧБ: {gv}   |   "
                 f"🎨 Реально цветных: {c}")
        if u:
            line1 += f"   |   ❓ Неизвестно: {u}"

        non_8bit = b16 + i16 + fl + oth
        line2 = f"🔵 8-bit PNG: {p8}   |   🟡 8-bit non-PNG: {o8}"
        if non_8bit:
            parts = []
            if b16: parts.append(f"16-bit: {b16}")
            if i16: parts.append(f"int16: {i16}")
            if fl: parts.append(f"float: {fl}")
            if oth: parts.append(f"other: {oth}")
            line2 += "   |   ⚠️ " + ", ".join(parts)
        return line1 + "\n" + line2, True


class PairValidator:
    """
    dataset_type:
        0 — labels (.txt);
        1 — masks (.png);
        2 — images-only.

    input_format (только для dataset_type=0):
        'auto'/'mixed' — эвристика;
        'detect' / 'obb_corners' / 'obb_xywhr' / 'segment' — жёстко.
    """

    def __init__(self, dataset_type=0, input_format=INPUT_FORMAT_AUTO):
        self.dataset_type = dataset_type
        self.input_format = input_format if input_format in _VALID_INPUT_FORMATS \
            else INPUT_FORMAT_AUTO
        self._cache = {}

    def set_dataset_type(self, dataset_type):
        self.dataset_type = dataset_type
        self._cache.clear()

    def set_input_format(self, fmt):
        self.input_format = fmt if fmt in _VALID_INPUT_FORMATS else INPUT_FORMAT_AUTO
        self._cache.clear()

    def clear_cache(self):
        self._cache.clear()

    def validate_full(self, pair):
        key = (pair[0], pair[1] if len(pair) > 1 else None)
        if key in self._cache:
            return self._cache[key]
        result = self._validate_impl(pair)
        self._cache[key] = result
        return result

    def validate(self, pair):
        ok, err, _, _ = self.validate_full(pair)
        return ok, err

    def validate_all(self, pairs, progress_cb=None):
        self._cache.clear()
        total = len(pairs)
        if total == 0:
            return 0, 0
        valid = 0; invalid = 0
        step = max(1, total // 100)
        for idx, pair in enumerate(pairs):
            ok, _ = self.validate(pair)
            if ok: valid += 1
            else: invalid += 1
            if progress_cb and ((idx + 1) % step == 0 or idx == total - 1):
                progress_cb(idx + 1, total)
        return valid, invalid

    def recompute_stats(self, pairs):
        color_stats = ImageStats.empty_color_stats()
        bit_stats = ImageStats.empty_bit_stats()
        for pair in pairs:
            _, _, ctype, btype = self.validate_full(pair)
            color_stats[ctype] = color_stats.get(ctype, 0) + 1
            bit_stats[btype] = bit_stats.get(btype, 0) + 1
        return color_stats, bit_stats

    def _validate_impl(self, pair):
        img_path = pair[0]
        aux_path = pair[1] if len(pair) > 1 else None

        if not os.path.exists(img_path):
            return False, "Image file not found", 'unknown', 'unknown'
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Cannot decode image", 'unknown', 'unknown'
            img_h, img_w = img.shape[:2]
            if img_h == 0 or img_w == 0:
                return False, "Zero-size image", 'unknown', 'unknown'
            color_type = ImageStats.detect_color_type(img)
            bit_type = ImageStats.detect_bit_type(img, img_path)
        except Exception as e:
            return False, f"Image read error: {e}", 'unknown', 'unknown'

        if aux_path is None:
            return True, "", color_type, bit_type

        if self.dataset_type == 1:
            if not os.path.exists(aux_path):
                return False, "Mask file not found", color_type, bit_type
            try:
                mask = read_mask_with_fallback(str(aux_path))
                if mask is None:
                    return False, "Cannot decode mask", color_type, bit_type
                if mask.ndim != 2:
                    return (False,
                            f"Mask expected 2D, got shape {mask.shape}",
                            color_type, bit_type)
                if mask.dtype not in (np.uint8, np.uint16):
                    return (False,
                            f"Mask dtype {mask.dtype} (expected uint8 or uint16)",
                            color_type, bit_type)
                if mask.shape[:2] != (img_h, img_w):
                    return (False,
                            f"Mask size {mask.shape[1]}x{mask.shape[0]} != "
                            f"image {img_w}x{img_h}",
                            color_type, bit_type)
            except Exception as e:
                return False, f"Mask read error: {e}", color_type, bit_type
            return True, "", color_type, bit_type

        if not os.path.exists(aux_path):
            return False, "Label file not found", color_type, bit_type
        try:
            with open(aux_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
        except Exception as e:
            return False, f"Label read error: {e}", color_type, bit_type

        lines = []
        for ln, raw in enumerate(raw_lines, 1):
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            lines.append((ln, s))

        if not lines:
            return True, "", color_type, bit_type

        types_in_file = set()
        for ln, line in lines:
            parts = line.split()
            if len(parts) < 5:
                return (False, f"Line {ln}: too few values ({len(parts)})",
                        color_type, bit_type)
            try:
                cls = int(parts[0])
                if cls < 0:
                    return (False, f"Line {ln}: negative class id {cls}",
                            color_type, bit_type)
            except ValueError:
                return (False, f"Line {ln}: invalid class id '{parts[0]}'",
                        color_type, bit_type)
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError as e:
                return (False, f"Line {ln}: invalid coordinate ({e})",
                        color_type, bit_type)

            def in_range(v):
                return 0.0 <= v <= 1.0

            n = len(parts)

            # Учитываем input_format: жёсткое правило или эвристика.
            fmt = self.input_format
            if fmt in (INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED):
                if n == 5:
                    kind = 'detect'
                elif n == 6:
                    kind = 'obb_xywhr'
                elif n == 9:
                    kind = 'obb_corners'
                elif n >= 7 and (n - 1) % 2 == 0:
                    kind = 'segment'
                else:
                    return (False,
                            f"Line {ln}: unknown format ({n} values)",
                            color_type, bit_type)
            elif fmt == INPUT_FORMAT_DETECT:
                if n != 5:
                    return (False,
                            f"Line {ln}: expected detect (5 values), got {n}",
                            color_type, bit_type)
                kind = 'detect'
            elif fmt == INPUT_FORMAT_OBB_CORNERS:
                if n != 9:
                    return (False,
                            f"Line {ln}: expected obb_corners (9 values), got {n}",
                            color_type, bit_type)
                kind = 'obb_corners'
            elif fmt == INPUT_FORMAT_OBB_XYWHR:
                if n != 6:
                    return (False,
                            f"Line {ln}: expected obb_xywhr (6 values), got {n}",
                            color_type, bit_type)
                kind = 'obb_xywhr'
            elif fmt == INPUT_FORMAT_SEGMENT:
                if not (n >= 7 and (n - 1) % 2 == 0):
                    return (False,
                            f"Line {ln}: expected segment (even >= 6 coords), got {n}",
                            color_type, bit_type)
                kind = 'segment'
            else:
                return (False, f"Line {ln}: unknown input_format {fmt}",
                        color_type, bit_type)

            if kind == 'detect':
                types_in_file.add('detect')
                cx, cy, bw, bh = coords
                if not in_range(cx) or not in_range(cy):
                    return (False, f"Line {ln}: center out of [0,1]",
                            color_type, bit_type)
                if bw <= 0 or bh <= 0:
                    return (False, f"Line {ln}: non-positive bbox size",
                            color_type, bit_type)
                if bw > 1 or bh > 1:
                    return (False, f"Line {ln}: bbox size > 1",
                            color_type, bit_type)
            elif kind == 'obb_corners':
                types_in_file.add('obb')
                for i in range(0, 8, 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return (False,
                                f"Line {ln}: OBB point {i // 2 + 1} out of [0,1]",
                                color_type, bit_type)
            elif kind == 'obb_xywhr':
                types_in_file.add('obb')
                cx, cy, bw, bh, angle = coords
                if not in_range(cx) or not in_range(cy):
                    return (False, f"Line {ln}: center out of [0,1]",
                            color_type, bit_type)
                if bw <= 0 or bh <= 0:
                    return (False, f"Line {ln}: non-positive OBB size",
                            color_type, bit_type)
            else:  # segment
                types_in_file.add('segment')
                num_pts = (n - 1) // 2
                if num_pts < 3:
                    return (False,
                            f"Line {ln}: polygon needs >=3 points, got {num_pts}",
                            color_type, bit_type)
                for i in range(0, len(coords), 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return (False,
                                f"Line {ln}: polygon point out of [0,1]",
                                color_type, bit_type)

        if len(types_in_file) > 1:
            return (False,
                    f"Mixed types in file: {', '.join(sorted(types_in_file))}",
                    color_type, bit_type)

        return True, "", color_type, bit_type

    @staticmethod
    def detect_annotation_type(label_path, format_hint=INPUT_FORMAT_AUTO):
        types = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    l = len(parts)
                    if format_hint in (INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED):
                        if l == 5:
                            types.add('detect')
                        elif l == 6:
                            types.add('obb')
                        elif l == 9:
                            types.add('obb')
                        elif l >= 7 and (l - 1) % 2 == 0:
                            types.add('segment')
                        else:
                            types.add('unknown')
                    elif format_hint == INPUT_FORMAT_DETECT:
                        types.add('detect' if l == 5 else 'unknown')
                    elif format_hint == INPUT_FORMAT_OBB_CORNERS:
                        types.add('obb' if l == 9 else 'unknown')
                    elif format_hint == INPUT_FORMAT_OBB_XYWHR:
                        types.add('obb' if l == 6 else 'unknown')
                    elif format_hint == INPUT_FORMAT_SEGMENT:
                        types.add('segment'
                                  if (l >= 7 and (l - 1) % 2 == 0)
                                  else 'unknown')
        except Exception:
            return 'error'
        if len(types) == 0:
            return 'empty'
        if len(types) == 1:
            return next(iter(types))
        return 'mixed'

    @staticmethod
    def check_pair_type_consistency():
        return None


# ============================================================
# Сканирование
# ============================================================
def _list_images(folder):
    result = {}
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTENSIONS:
            result[name] = os.path.join(folder, f)
    return result


def scan_detection_pairs(images_folder, labels_folder, log_cb=None,
                          format_hint=INPUT_FORMAT_AUTO):
    types_stats = {}
    pairs = []
    img_files = _list_images(images_folder)
    for name, img_path in img_files.items():
        label_path = os.path.join(labels_folder, name + '.txt')
        if not os.path.exists(label_path):
            continue
        try:
            ann_type = PairValidator.detect_annotation_type(label_path, format_hint)
            types_stats[ann_type] = types_stats.get(ann_type, 0) + 1
            pairs.append((img_path, label_path))
        except Exception as e:
            if log_cb:
                log_cb(f"Ошибка обработки пары {name}: {e}")
    return pairs, types_stats


def scan_segmentation_pairs(images_folder, masks_folder, log_cb=None):
    pairs = []
    img_files = _list_images(images_folder)
    for name, img_path in img_files.items():
        mask_path = None
        for ext in IMG_EXTENSIONS:
            cand = os.path.join(masks_folder, name + ext)
            if os.path.exists(cand):
                if os.path.abspath(cand) == os.path.abspath(img_path):
                    if log_cb:
                        log_cb(f"Предупреждение: маска совпадает с изображением {img_path}")
                    mask_path = None
                    break
                mask_path = cand
                break
        if mask_path is None:
            for f in os.listdir(masks_folder):
                mask_name, mask_ext = os.path.splitext(f)
                if mask_name == name and mask_ext.lower() in IMG_EXTENSIONS:
                    cand = os.path.join(masks_folder, f)
                    if os.path.abspath(cand) != os.path.abspath(img_path):
                        mask_path = cand
                    break
        if mask_path is None:
            continue
        test = read_mask_with_fallback(str(mask_path))
        if test is None:
            if log_cb:
                log_cb(f"Предупреждение: не удалось прочитать маску {mask_path}")
            continue
        pairs.append((img_path, mask_path))
    return pairs


def scan_image_only_pairs(images_folder):
    return [(p, None) for p in _list_images(images_folder).values()]


# ============================================================
# Пакетные конвертации (без изменений в логике)
# ============================================================
def select_pairs_for_conversion(pairs, validator, target):
    to_convert = []
    for pair in pairs:
        img_path = pair[0]
        _, _, ctype, _ = validator.validate_full(pair)
        if target == 'gray':
            if ctype in ('color', 'gray_visual'):
                to_convert.append((pair, img_path, ctype))
        elif target == 'color':
            if ctype == 'gray_real':
                to_convert.append((pair, img_path, ctype))
    return to_convert


def convert_pairs_color(pairs, target, progress_cb=None, log_cb=None):
    converted = 0; errors = 0
    total = len(pairs)
    step = max(1, total // 100) if total else 1
    for idx, item in enumerate(pairs):
        img_path = item[1] if len(item) > 1 else item[0][0]
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors += 1
            else:
                out = ImageStats.to_gray(img) if target == 'gray' \
                    else ImageStats.to_bgr3(img)
                if out is None:
                    errors += 1
                else:
                    cv2.imwrite(str(img_path), out)
                    converted += 1
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"Ошибка конвертации {img_path}: {e}")
        if progress_cb and total and ((idx + 1) % step == 0 or idx == total - 1):
            progress_cb(idx + 1, total)
    return converted, errors


def select_pairs_for_8bit_png(pairs, validator):
    """
    Возвращает (all_pairs, need_convert).
    ВАЖНО: для dataset_type==1 (masks) маски НЕ включаются —
    они должны оставаться в исходном dtype.
    """
    need = []
    for pair in pairs:
        _, _, _, btype = validator.validate_full(pair)
        if btype != 'uint8_png':
            need.append(pair)
    return list(pairs), need


def convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None,
                              skip_masks=False):
    """
    skip_masks: если True — pairs трактуются как pairs из (img, mask),
                но конвертируем только изображения (pair[0]), не маски.
    """
    converted = 0; errors = 0
    path_map = {}
    total = len(pairs)
    step = max(1, total // 100) if total else 1
    for idx, pair in enumerate(pairs):
        img_path = pair[0]
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors += 1
            else:
                img8 = ImageStats.to_uint8(img)
                base, ext = os.path.splitext(str(img_path))
                target_path = str(img_path) if ext.lower() == '.png' else base + '.png'
                if not cv2.imwrite(target_path, img8):
                    errors += 1
                    if log_cb:
                        log_cb(f"Не удалось сохранить {target_path}")
                else:
                    if target_path != str(img_path):
                        try:
                            os.remove(str(img_path))
                        except Exception as e:
                            if log_cb:
                                log_cb(f"Не удалось удалить {img_path}: {e}")
                        path_map[str(img_path)] = target_path
                    converted += 1
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"Ошибка конвертации {img_path}: {e}")
        if progress_cb and total and ((idx + 1) % step == 0 or idx == total - 1):
            progress_cb(idx + 1, total)
    return converted, errors, path_map


# ============================================================
# Отображение
# ============================================================
def build_pair_display(pair, validator, dataset_type, include_empty):
    img_path = pair[0]
    aux_path = pair[1] if len(pair) > 1 else None
    name = os.path.splitext(os.path.basename(img_path))[0]

    is_valid, err, color_type, bit_type = validator.validate_full(pair)

    if dataset_type == 0 and aux_path:
        try:
            ann_type = PairValidator.detect_annotation_type(
                aux_path, validator.input_format
            )
        except Exception:
            ann_type = 'error'
    elif dataset_type == 1:
        ann_type = 'segment'
    else:
        ann_type = None

    color_marker = {
        'gray_real':   " [ч/б 1ch]",
        'gray_visual': " [визуально ч/б]",
        'color':       " [цветное]",
    }.get(color_type, "")

    bit_marker = {
        'uint8_other': " [8b не-PNG]",
        'uint16':      " [16bit]",
        'int16':       " [int16]",
        'float':       " [float]",
        'other':       " [??bit]",
    }.get(bit_type, "")

    if not is_valid:
        tag = 'invalid'
        kind = 'label' if dataset_type == 0 else ('mask' if dataset_type == 1 else '')
        text = f"{name}{color_marker}{bit_marker} (image + {kind}) [INVALID: {err}]"
    elif ann_type == 'empty':
        tag = 'empty'
        marker = "empty → include" if include_empty else "empty → skip"
        text = f"{name}{color_marker}{bit_marker} (image + label) [{marker}]"
    elif ann_type in ('detect', 'obb', 'segment', 'error', 'mixed', 'unknown'):
        tag = ann_type
        text = f"{name}{color_marker}{bit_marker} (image + label) [{ann_type}]"
    else:
        tag = None
        suffix = "image + mask" if dataset_type == 1 else "image only"
        text = f"{name}{color_marker}{bit_marker} ({suffix})"

    return {
        'name': name, 'is_valid': is_valid, 'err': err,
        'color_type': color_type, 'bit_type': bit_type,
        'ann_type': ann_type, 'tag': tag, 'text': text,
    }


def draw_annotations_on_image(img, annotations):
    out = img.copy()
    h, w = out.shape[:2]
    for ann in annotations:
        typ = ann[0]; cls = ann[1]
        color = (0, 255, 0)
        if typ == 'detect':
            _, cls, cx, cy, bw, bh = ann
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"cls={cls}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif typ in ('obb', 'segment'):
            _, cls, points = ann
            pts = []
            for i in range(0, len(points), 2):
                pts.append([int(points[i] * w), int(points[i + 1] * h)])
            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(out, [pts_np], isClosed=True, color=color, thickness=2)
                cv2.putText(out, f"cls={cls}",
                            (pts[0][0], max(15, pts[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def apply_mask_overlay(img, mask):
    """
    Накладывает цветной overlay.
    Маска — одноканальная uint8 или uint16.
    """
    overlay = img.copy()
    if mask is None:
        return img
    if mask.ndim != 2:
        return img
    unique_vals = np.unique(mask)
    for val in unique_vals:
        if val == 0:
            continue
        # Если val > 255 — берём модуль для цвета, но отображение пикселя работает.
        hue = (int(val) * 37) % 180
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                            cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        overlay[mask == val] = color
    return cv2.addWeighted(img, 0.5, overlay, 0.5, 0)


class PairValidationDialog(QDialog):
    def __init__(self, parent, pair, dataset_type, validator):
        super().__init__(parent)
        self.dataset_type = dataset_type
        self.validator = validator

        img_path = pair[0]
        aux_path = pair[1] if len(pair) > 1 else None

        self.setWindowTitle(f"Валидация: {os.path.basename(img_path)}")
        self.resize(1100, 800)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Ошибка",
                                f"Не удалось загрузить изображение:\n{img_path}")
            self.reject()
            return

        disp = ImageStats.to_display_bgr(img)

        annotations = []
        if self.dataset_type == 0 and aux_path:
            annotations = load_annotations(aux_path, disp.shape[1], disp.shape[0],
                                            format_hint=validator.input_format)

        if self.dataset_type == 1 and aux_path:
            mask = read_mask_with_fallback(str(aux_path))
            if mask is not None:
                if mask.shape[:2] != disp.shape[:2]:
                    mask = cv2.resize(mask, (disp.shape[1], disp.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                disp = apply_mask_overlay(disp, mask)
        elif annotations:
            disp = draw_annotations_on_image(disp, annotations)

        layout = QVBoxLayout(self)

        is_valid, err, color_type, bit_type = self.validator.validate_full(pair)
        color_str = {
            'color': 'Реально цветное',
            'gray_real': 'Реально Ч/Б (1 канал)',
            'gray_visual': 'Визуально Ч/Б (3 канала)',
            'unknown': '—',
        }.get(color_type, '—')
        bit_str = {
            'uint8_png': 'uint8 PNG', 'uint8_other': 'uint8 (не PNG)',
            'uint16': 'uint16', 'int16': 'int16', 'float': 'float',
            'other': 'other', 'unknown': '—',
        }.get(bit_type, '—')
        status = (f"{'✅ Пара корректна' if is_valid else f'❌ Ошибка валидации: {err}'}"
                  f"   |   Цвет: {color_str}   |   Битность: {bit_str}")
        status_label = QLabel(status)
        status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_label.setStyleSheet(
            "color: #007700; font-weight: bold;" if is_valid
            else "color: #b00000; font-weight: bold;"
        )
        layout.addWidget(status_label)

        view = SmartGraphicsView()
        view.set_pixmap(numpy_to_qpixmap(disp))
        view.setMinimumHeight(500)
        layout.addWidget(view, 1)

        path_label = QLabel(f"Image: {img_path}\nAux:   {aux_path or '—'}")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(path_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)




# ============================================================
# Общая логика загрузки датасетов для вкладок Анализ / Подготовка
# ============================================================
class DatasetLoaderMixin:
    """
    Общие методы загрузки и валидации для вкладок «Анализ» и «Подготовка».

    Подкласс должен:
      1. В __init__ вызвать setup_*_ui(self) (создаст нужные виджеты).
      2. Инициализировать атрибуты:
         images_folder, labels_folder, masks_folder, pairs, filtered_pairs,
         dataset_type, input_format, mask_pixel_offset_in, validator,
         color_type_stats, bit_type_stats, annotation_types_stats,
         unique_classes, original_ids, mask_class_remap, included_orig_ids,
         class_remap, num_classes,
         _yaml_mode, _yaml_root, _yaml_path, _yaml_names,
         _coco_tmp_labels_dir, _coco_json_counter, _updating_class_table.
      3. Переопределить хуки:
         _after_load_hook()             — после успешной загрузки/сканирования;
         _after_type_change_hook(new)   — после смены dataset_type.
      4. При необходимости переопределить _refresh_mask_pixel_column()
         и _sync_all_states().
    """

    # ---------------- хуки ----------------
    def _after_load_hook(self):
        """Вызывается в конце scan_pairs / _apply_loaded_pairs /
        _load_coco_from_paths / _on_input_format_changed.
        Переопределить в подклассе (Анализ: apply_filters_and_update_metrics;
        Подготовка: update_generate_button_state + update_split_counts)."""
        pass

    def _after_type_change_hook(self, new_type):
        """Вызывается в конце _on_dataset_type_changed. Переопределить."""
        pass

    # ---------------- синхронизация UI ----------------
    def _sync_all_states(self, *_):
        """Дефолт: синхронизирует input_format_combo и кнопку Load Labels.
        Подготовка переопределяет (там ещё output-group и аугментации)."""
        input_type = self.dataset_type_combo.currentData()
        self.input_format_combo.setEnabled(input_type == 0)
        if input_type == 0:
            self.btn_labels.setText("Load Labels")
            self.btn_labels.setEnabled(True)
        elif input_type == 1:
            self.btn_labels.setText("Load Masks")
            self.btn_labels.setEnabled(True)
        else:
            self.btn_labels.setText("No labels (images only)")
            self.btn_labels.setEnabled(False)
        self._refresh_mask_pixel_column()

    def _refresh_mask_pixel_column(self):
        """Дефолт: колонка Mask pixel видна только при input=masks.
        Подготовка переопределяет (там ещё out_masks и заполнение значений)."""
        input_type = getattr(self, 'dataset_type', 0)
        try:
            self.class_table.setColumnHidden(3, input_type != 1)
        except Exception:
            pass

    # ---------------- логирование ----------------
    def log(self, message):
        try:
            self.log_text.append(message)
            sb = self.log_text.verticalScrollBar()
            sb.setValue(sb.maximum())
            QApplication.processEvents()
        except Exception:
            print(f"[log] {message}")

    def _log_debug(self, msg):
        try:
            print(f"[DEBUG] {msg}")
            self.log(f"[DEBUG] {msg}")
        except Exception:
            pass

    def _log_exception(self, context, exc=None):
        tb = traceback.format_exc()
        head = f"❌ {context}: {exc if exc is not None else ''}"
        try:
            print(head)
            print(tb)
            self.log(head)
            self.log(tb)
        except Exception:
            pass

    # ---------------- клипборд / двойной клик ----------------
    def _copy_selected_rows_to_clipboard(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        text = "\n".join(it.text() for it in items)
        QApplication.clipboard().setText(text)
        self.log(f"Скопировано строк: {len(items)}")

    def _on_file_list_double_clicked(self, item):
        try:
            if item is None:
                return
            raw_data = item.data(Qt.UserRole + 1)
            if not raw_data:
                return
            self._open_pair_validation(raw_data)
        except Exception as e:
            self._log_exception("Ошибка в _on_file_list_double_clicked", e)

    # ---------------- выбор папок ----------------
    def select_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self._yaml_mode = False
            self.images_folder = folder
            self.log(f"Images folder: {folder}")
            self.validator.clear_cache()
            self.scan_pairs()

    def select_labels_folder(self):
        if self.dataset_type == 2:
            return
        self._yaml_mode = False
        self.validator.clear_cache()
        if self.dataset_type == 0:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
            if folder:
                self.labels_folder = folder
                self.masks_folder = None
                self.log(f"Labels folder: {folder}")
                self.scan_pairs()
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with mask images")
            if folder:
                self.masks_folder = folder
                self.labels_folder = folder
                self.log(f"Masks folder: {folder}")
                self.scan_pairs()

    # ---------------- YOLO YAML ----------------
    def load_from_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO data.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not file_path:
            return
        if is_unet_yaml(file_path):
            reply = QMessageBox.question(
                self, "Похоже на U-Net YAML",
                f"Файл {os.path.basename(file_path)} похож на U-Net-датасет "
                f"(есть masks-поля). Загрузить его как U-Net?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self._load_unet_from_path(file_path)
                return
        self._load_yolo_yaml_from_path(file_path)

    def _load_yolo_yaml_from_path(self, file_path):
        try:
            pairs, names, root, splits_info = load_yolo_yaml(file_path, log_cb=self.log)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать YAML:\n{e}")
            return

        if not pairs:
            QMessageBox.warning(self, "Ошибка",
                                "В YAML не найдено ни одной существующей папки с изображениями.")
            return

        missing = sum(1 for _, lbl in pairs if lbl is None)
        with_labels = [(img, lbl) for img, lbl in pairs if lbl is not None]
        if not with_labels:
            QMessageBox.warning(
                self, "Ошибка",
                f"Найдено {len(pairs)} изображений, но ни одного .txt.")
            return

        self._yaml_mode = True
        self._yaml_root = root
        self._yaml_path = file_path
        self._yaml_names = names

        self.images_folder = root
        self.labels_folder = root
        self.masks_folder = None

        self._apply_loaded_pairs(
            pairs=with_labels,
            input_type=0,
            input_format='auto',
            class_names_from_yaml=names,
            source_label=f"YAML: {file_path}",
        )

        self.log(f"=== Загрузка из YAML: {file_path} ===")
        self.log(f"Root: {root}")
        for split, n in splits_info.items():
            if n:
                self.log(f"  • {split}: {n} изображений")
        self.log(f"Пар с метками: {len(with_labels)}, без .txt: {missing}")

    # ---------------- COCO ----------------
    def load_from_coco(self):
        self._handle_load_coco(replace=True)

    def add_from_coco(self):
        self._handle_load_coco(replace=False)

    def _handle_load_coco(self, replace):
        choice = self._ask_coco_source()
        if choice is None:
            return
        if choice == 'files':
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select COCO JSON(s)", "", "JSON files (*.json);;All files (*)")
            if not paths:
                return
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with COCO JSONs")
            if not folder:
                return
            paths = self._find_coco_json_in_folder(folder)
            if not paths:
                QMessageBox.warning(self, "COCO", f"В папке не найдено COCO JSON.\n{folder}")
                return
            self.log(f"[COCO] Найдено {len(paths)} JSON в {folder}")
        self._load_coco_from_paths(paths, replace=replace)

    def _ask_coco_source(self):
        box = QMessageBox(self)
        box.setWindowTitle("COCO")
        box.setText("Как выбрать COCO JSON?")
        btn_files = box.addButton("Select files...", QMessageBox.AcceptRole)
        btn_folder = box.addButton("Select folder...", QMessageBox.AcceptRole)
        box.addButton("Отмена", QMessageBox.RejectRole)
        box.setDefaultButton(btn_files)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_files:
            return 'files'
        if clicked is btn_folder:
            return 'folder'
        return None

    def _find_coco_json_in_folder(self, folder):
        candidates = []
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith('.json'):
                    continue
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
                        candidates.append(p)
                except Exception:
                    continue
        return sorted(candidates)

    def _get_coco_tmp_labels_dir(self, first_json_path=None):
        existing = getattr(self, '_coco_tmp_labels_dir', None)
        if existing:
            return existing
        base = os.path.dirname(first_json_path) if first_json_path else os.path.expanduser('~')
        tmp = os.path.join(base, '_coco_labels_tmp')
        os.makedirs(tmp, exist_ok=True)
        self._coco_tmp_labels_dir = tmp
        return tmp

    def _load_coco_from_paths(self, json_paths, replace=True):
        if not json_paths:
            return
        if replace:
            self._coco_tmp_labels_dir = None
            self._coco_json_counter = 0

        per_json_data = []
        for jp in json_paths:
            try:
                items, categories, root = load_coco(jp, log_cb=self.log)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка COCO",
                                     f"Не удалось загрузить:\n{jp}\n\n{e}")
                return
            if not items:
                self.log(f"⚠️ {os.path.basename(jp)}: нет изображений — пропуск.")
                continue
            per_json_data.append((jp, items, categories, root))

        if not per_json_data:
            QMessageBox.warning(self, "COCO", "Не найдено ни одного изображения.")
            return

        if replace:
            name_to_global = {}
            next_gid = 0
            base_pairs = []
        else:
            name_to_global = {}
            max_gid = -1
            for row in range(self.class_table.rowCount()):
                try:
                    gid = int(self.class_table.item(row, 0).text())
                    name_item = self.class_table.item(row, 2)
                    if name_item is None:
                        continue
                    name = name_item.text().strip() or str(gid)
                    name_to_global[name] = gid
                    if gid > max_gid:
                        max_gid = gid
                except (ValueError, TypeError, AttributeError):
                    continue
            next_gid = max_gid + 1
            base_pairs = list(self.pairs)

        tmp_dir = self._get_coco_tmp_labels_dir(per_json_data[0][0])
        new_pairs = []

        for idx, (jp, items, categories, _root) in enumerate(per_json_data):
            local_map = {}
            for coco_cat_id, name in categories.items():
                name_str = str(name) if name is not None else str(coco_cat_id)
                if name_str not in name_to_global:
                    name_to_global[name_str] = next_gid
                    next_gid += 1
                local_map[int(coco_cat_id)] = name_to_global[name_str]

            for it in items:
                new_anns = []
                for a in it['annotations']:
                    coco_cat_id = int(a[1]) + 1
                    gid = local_map.get(coco_cat_id)
                    if gid is None:
                        name_str = str(coco_cat_id)
                        if name_str not in name_to_global:
                            name_to_global[name_str] = next_gid
                            next_gid += 1
                        gid = name_to_global[name_str]
                        local_map[coco_cat_id] = gid
                    new_anns.append((a[0], gid, *a[2:]))
                it['annotations'] = new_anns

            if not hasattr(self, '_coco_json_counter'):
                self._coco_json_counter = 0
            prefix = f"j{self._coco_json_counter:03d}_"
            self._coco_json_counter += 1

            try:
                pairs_i = write_yolo_txt_for_coco_items(
                    items, tmp_dir, class_remap=None, log_cb=self.log, name_prefix=prefix)
                new_pairs.extend(pairs_i)
            except Exception as e:
                self._log_exception(f"Ошибка записи .txt из {jp}", e)
                return

        class_names = {gid: name for name, gid in name_to_global.items()}
        self.pairs = (new_pairs if replace else base_pairs + new_pairs)

        self.images_folder = os.path.dirname(json_paths[0])
        self.labels_folder = tmp_dir
        self.masks_folder = None

        self.dataset_type = 0
        self.input_format = 'auto'
        self.validator.set_dataset_type(0)
        self.validator.set_input_format('auto')

        self.dataset_type_combo.blockSignals(True)
        self.input_format_combo.blockSignals(True)
        try:
            self.dataset_type_combo.setCurrentIndex(0)
            idx = self.input_format_combo.findData('auto')
            if idx >= 0:
                self.input_format_combo.setCurrentIndex(idx)
        finally:
            self.dataset_type_combo.blockSignals(False)
            self.input_format_combo.blockSignals(False)

        self._sync_all_states()
        self.validator.clear_cache()
        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._fill_class_table_from_dict(class_names, preserve_include=True)
        self._rebuild_file_list_display(pairs=self.pairs)

        action = "Загрузка" if replace else "Добавление"
        self.log(f"=== {action} COCO ({len(per_json_data)} файлов) ===")
        for jp, items, _, _ in per_json_data:
            self.log(f"  • {os.path.basename(jp)}: images={len(items)}")
        self.log(f"Итого: пар={len(self.pairs)}, классов={len(class_names)}")

        extra = f"  [COCO: {len(per_json_data)} json, {action.lower()}]"
        if invalid_count > 0:
            extra += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra, error=invalid_count > 0)

        self._after_load_hook()

    # ---------------- U-Net YAML ----------------
    def load_from_unet_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net YAML", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not yaml_path:
            return
        self._load_unet_from_path(yaml_path)

    def _load_unet_from_path(self, yaml_path):
        try:
            data = load_unet_yaml(yaml_path, log_cb=self.log)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка U-Net YAML",
                                 f"Не удалось загрузить:\n{e}")
            return

        pairs = data['pairs']
        yolo_class_names = data['class_names']
        mask_offset = data['mask_class_offset']
        root = data['root']

        if not pairs:
            QMessageBox.warning(self, "U-Net", "Не найдено ни одной пары.")
            return

        pixel_class_names = {int(yc) + int(mask_offset): name
                             for yc, name in yolo_class_names.items()}
        self.images_folder = root
        self.labels_folder = None
        self.masks_folder = root
        self.mask_pixel_offset_in = 0

        ui_pairs = [(img, mask) for (img, mask) in pairs]

        self._apply_loaded_pairs(
            pairs=ui_pairs,
            input_type=1,
            input_format='auto',
            class_names_override=pixel_class_names,
            source_label=f"U-Net YAML: {yaml_path}",
        )
        self.log(f"=== Загрузка из U-Net YAML: {yaml_path} ===")
        self.log(f"Root: {root}")
        self.log(f"Пар: {len(pairs)}, классов: {len(yolo_class_names)}, "
                 f"mask_class_offset: {mask_offset}")

    # ---------------- общая инициализация после загрузки ----------------
    def _apply_loaded_pairs(self, pairs, input_type, input_format,
                            class_names_from_yaml=None,
                            class_names_override=None,
                            source_label=""):
        if input_type != 0 or class_names_override is None:
            self._coco_tmp_labels_dir = None
            self._coco_json_counter = 0
        self.validator.clear_cache()
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
        self.annotation_types_stats = {}

        self.dataset_type = input_type
        self.input_format = input_format
        self.validator.set_dataset_type(input_type)
        self.validator.set_input_format(input_format)

        self.dataset_type_combo.blockSignals(True)
        self.input_format_combo.blockSignals(True)
        try:
            self.dataset_type_combo.setCurrentIndex(input_type)
            idx = self.input_format_combo.findData(input_format)
            if idx >= 0:
                self.input_format_combo.setCurrentIndex(idx)
        finally:
            self.dataset_type_combo.blockSignals(False)
            self.input_format_combo.blockSignals(False)

        self._sync_all_states()
        self.pairs = list(pairs)

        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()

        self._yaml_names = class_names_from_yaml
        if class_names_override:
            self._fill_class_table_from_dict(class_names_override)
        else:
            self.collect_classes()

        self._rebuild_file_list_display(pairs=self.pairs)

        extra = ""
        if source_label:
            extra = f"  [{source_label}]"
        if invalid_count > 0:
            extra += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra, error=invalid_count > 0)
        self.log(f"Пар: {len(self.pairs)} (корректных: {valid_count}, "
                 f"некорректных: {invalid_count})")

        self._after_load_hook()

    # ---------------- сканирование папок ----------------
    def scan_pairs(self):
        try:
            if self._yaml_mode:
                self.validator.clear_cache()
                self._recompute_color_stats()
                self._rebuild_file_list_display(pairs=self.pairs)
                self._after_load_hook()
                return

            self.pairs = []
            self.file_list.clear()
            self.annotation_types_stats = {}
            self.color_type_stats = ImageStats.empty_color_stats()
            self.bit_type_stats = ImageStats.empty_bit_stats()
            self.validator.clear_cache()

            if not self.images_folder:
                self.log("Папка с изображениями не выбрана")
                self._update_color_stats_label()
                return

            dataset_type_idx = self.dataset_type

            if dataset_type_idx == 0:
                if not self.labels_folder:
                    self.log("Папка с метками не выбрана — пары пока не строим.")
                    self._update_pair_count_label("", error=False)
                    return
                self.log("=== Сканирование пар (детекция) ===")
                self.pairs, self.annotation_types_stats = scan_detection_pairs(
                    self.images_folder, self.labels_folder,
                    log_cb=self.log, format_hint=self.input_format)
            elif dataset_type_idx == 1:
                if not self.masks_folder:
                    self.log("Папка с масками не выбрана.")
                    self._update_pair_count_label("", error=False)
                    return
                self.log("=== Сканирование пар (сегментация) ===")
                self.pairs = scan_segmentation_pairs(
                    self.images_folder, self.masks_folder, log_cb=self.log)
            else:
                self.log("=== Сканирование изображений (без меток) ===")
                self.pairs = scan_image_only_pairs(self.images_folder)
                self.annotation_types_stats = {}

            self.log(f"Найдено пар до валидации: {len(self.pairs)}")
            valid_count, invalid_count = self._validate_all_pairs()
            self._recompute_color_stats()

            extra_info = ""
            if self.annotation_types_stats:
                type_str = ", ".join(f"{k}:{v}" for k, v in self.annotation_types_stats.items())
                extra_info = f"  [{type_str}]"
            if invalid_count > 0:
                extra_info += f"  ⚠️ некорректных: {invalid_count}"
            self._update_pair_count_label(extra_info, error=invalid_count > 0)
            self.log(f"Валидация: корректных={valid_count}, некорректных={invalid_count}")

            self.collect_classes()
            self._rebuild_file_list_display(pairs=self.pairs)
            self._after_load_hook()
        except Exception as e:
            self._log_exception("Ошибка в scan_pairs", e)

    # ---------------- валидация ----------------
    def _validate_all_pairs(self):
        if not self.pairs:
            return 0, 0
        total = len(self.pairs)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Проверка пар: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Проверка пар: {current}/{tot} ({percent}%)")
            QApplication.processEvents()

        valid, invalid = self.validator.validate_all(self.pairs, progress_cb=on_progress)

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        QApplication.processEvents()
        return valid, invalid

    def _recompute_color_stats(self):
        try:
            self.color_type_stats, self.bit_type_stats = self.validator.recompute_stats(self.pairs)
        except Exception:
            pass

    def _update_color_stats_label(self):
        if not hasattr(self, 'color_stats_label'):
            return
        try:
            text, _ = ImageStats.format_stats_text(
                self.color_type_stats, self.bit_type_stats, bool(self.pairs))
            self.color_stats_label.setText(text)
        except Exception:
            pass

    def _open_pair_validation(self, pair):
        try:
            img_path = None
            aux_path = None
            if isinstance(pair, str):
                img_path = pair
            elif isinstance(pair, (tuple, list)) and len(pair) >= 1:
                img_path = pair[0]
                if len(pair) >= 2:
                    aux_path = pair[1]

            if not img_path or not os.path.exists(str(img_path)):
                QMessageBox.warning(self, "Файл не найден",
                                    f"Изображение не существует:\n{img_path}")
                return

            dlg = PairValidationDialog(
                self, (str(img_path), str(aux_path) if aux_path else None),
                self.dataset_type, self.validator)
            dlg.exec_()
        except Exception as e:
            self._log_exception("Ошибка валидации пары", e)
            QMessageBox.critical(self, "Ошибка валидации",
                                 f"{type(e).__name__}: {e}\n\nСм. лог.")

    # ---------------- отрисовка списка пар ----------------
    def _make_list_item(self, text, tag=None, error_msg=""):
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()
        if tag == 'invalid':
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Некорректная пара: {error_msg}")
        elif tag == 'empty':
            if include_empty:
                item.setForeground(QColor(0, 140, 0))
                item.setToolTip("Пустая аннотация — будет включена как негатив.")
            else:
                item.setForeground(QColor(200, 120, 0))
                item.setToolTip("Пустая аннотация — будет пропущена.")
        elif tag == 'error':
            item.setForeground(QColor(200, 0, 0))
            item.setToolTip("Ошибка чтения аннотации.")
        elif tag == 'mixed':
            item.setForeground(QColor(160, 0, 200))
            item.setToolTip("Смешанные типы аннотаций.")
        elif tag == 'unknown':
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _rebuild_file_list_display(self, pairs=None):
        if pairs is None:
            pairs = self.pairs

        if not pairs:
            self.file_list.clear()
            self.validation_warning_label.setVisible(False)
            self.validation_progress_label.setVisible(False)
            self._update_color_stats_label()
            return

        self.file_list.blockSignals(True)
        self.file_list.clear()
        valid_count = 0
        invalid_count = 0
        invalid_names = []
        total = len(pairs)
        step = max(1, total // 100)
        include_empty = self._is_include_empty()

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Построение списка: 0/{total}...")
        QApplication.processEvents()

        for idx, pair in enumerate(pairs):
            try:
                info = build_pair_display(pair, self.validator, self.dataset_type, include_empty)
            except Exception as e:
                self._log_exception(f"build_pair_display упал на паре #{idx}", e)
                info = {
                    'text': f"<ошибка: {pair!r}>", 'tag': 'error',
                    'err': f"{type(e).__name__}: {e}",
                    'is_valid': False, 'name': f"pair#{idx}",
                    'color_type': None, 'bit_type': None,
                }

            if info['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_names.append(info['name'])

            item = self._make_list_item(info['text'], tag=info['tag'], error_msg=info['err'])

            img_path = pair[0] if isinstance(pair, (tuple, list)) and pair else None
            aux_path = (pair[1] if isinstance(pair, (tuple, list)) and len(pair) > 1 else None)
            item.setData(Qt.UserRole, img_path)
            item.setData(Qt.UserRole + 1, (img_path, aux_path))
            item.setData(Qt.UserRole + 2, 'valid' if info['is_valid'] else 'invalid')
            item.setData(Qt.UserRole + 3, info['color_type'])
            item.setData(Qt.UserRole + 4, info['bit_type'])

            self.file_list.addItem(item)

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Построение списка: {idx + 1}/{total} ({percent}%)")
                QApplication.processEvents()

        self.file_list.blockSignals(False)
        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")

        if invalid_count > 0:
            preview = ", ".join(invalid_names[:5])
            more = "" if invalid_count <= 5 else f" и ещё {invalid_count - 5}"
            self.validation_warning_label.setText(
                f"⚠️ Обнаружено {invalid_count} некорректных пар: {preview}{more}. "
                f"Они НЕ будут включены в датасет.")
            self.validation_warning_label.setVisible(True)
        else:
            self.validation_warning_label.setVisible(False)

        self._update_color_stats_label()

    def _is_include_empty(self):
        return bool(self.include_empty_annotations.isChecked())

    def _on_include_empty_toggled(self, *_):
        if not self.pairs:
            return
        self._rebuild_file_list_display(pairs=self.pairs)

    # ---------------- смена режима ----------------
    def _on_dataset_type_changed(self, idx):
        new_type = self.dataset_type_combo.currentData()
        if new_type is None:
            new_type = idx
        self.dataset_type = new_type
        self.validator.set_dataset_type(new_type)

        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        if new_type == 2:
            self.pairs = []
            self.file_list.clear()
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.validation_warning_label.setVisible(False)
            self._yaml_mode = False
            self._update_color_stats_label()

        self._after_type_change_hook(new_type)

        if self.images_folder and not self._yaml_mode:
            self.scan_pairs()

    def _on_input_format_changed(self, idx):
        fmt = self.input_format_combo.currentData()
        if fmt is None:
            fmt = 'auto'
        self.input_format = fmt
        self.validator.set_input_format(fmt)
        if self.pairs:
            self._validate_all_pairs()
            self._rebuild_file_list_display(pairs=self.pairs)
            self._after_load_hook()

    # ---------------- работа с класс-таблицей ----------------
    def _fill_class_table_from_dict(self, class_names_dict, preserve_include=True):
        include_by_name = {}
        if preserve_include and self.class_table.rowCount() > 0:
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 4)
                name_item = self.class_table.item(row, 2)
                if cb is None or name_item is None:
                    continue
                include_by_name[name_item.text().strip()] = cb.isChecked()

        self._updating_class_table = True
        self.class_table.blockSignals(True)
        try:
            self.class_table.setRowCount(0)
            if not class_names_dict:
                self.original_ids = []
                return

            sorted_ids = sorted(class_names_dict.keys())
            self.original_ids = list(sorted_ids)
            self.class_table.setRowCount(len(sorted_ids))
            for i, cid in enumerate(sorted_ids):
                new_item = QTableWidgetItem(str(i))
                new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, new_item)

                orig_item = QTableWidgetItem(str(cid))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)

                name_str = str(class_names_dict[cid])
                name_item = QTableWidgetItem(name_str)
                self.class_table.setItem(i, 2, name_item)

                cb = QCheckBox()
                cb.setChecked(include_by_name.get(name_str, True))
                cb.stateChanged.connect(self.update_included_classes)
                self.class_table.setCellWidget(i, 4, cb)
        finally:
            self.class_table.blockSignals(False)
            self._updating_class_table = False

        self.update_included_classes()

    def _get_yaml_name_for_id(self, cid):
        names = self._yaml_names
        if not names:
            return None
        try:
            if isinstance(names, dict):
                if cid in names:
                    return str(names[cid])
                if str(cid) in names:
                    return str(names[str(cid)])
            elif isinstance(names, (list, tuple)):
                if 0 <= cid < len(names):
                    return str(names[cid])
        except Exception:
            pass
        return None

    def collect_classes(self):
        if not self.pairs:
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(0)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            return

        if self.dataset_type == 0:
            unique_ids = set()
            for pair in self.pairs:
                label_path = pair[1] if len(pair) > 1 else None
                if not label_path:
                    continue
                try:
                    anns = load_annotations(label_path, 1, 1, format_hint=self.input_format)
                except Exception:
                    continue
                for ann in anns:
                    try:
                        unique_ids.add(int(ann[1]))
                    except (ValueError, TypeError, IndexError):
                        continue

            self.original_ids = sorted(unique_ids)
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(len(self.original_ids))
                for i, cid in enumerate(self.original_ids):
                    new_item = QTableWidgetItem(str(i))
                    new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 0, new_item)

                    orig_item = QTableWidgetItem(str(cid))
                    orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 1, orig_item)

                    yaml_name = self._get_yaml_name_for_id(cid)
                    name_item = QTableWidgetItem(yaml_name if yaml_name else str(cid))
                    self.class_table.setItem(i, 2, name_item)

                    cb = QCheckBox()
                    cb.setChecked(True)
                    cb.stateChanged.connect(self.update_included_classes)
                    self.class_table.setCellWidget(i, 4, cb)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False

            self.update_included_classes()

        elif self.dataset_type == 1:
            unique_vals = set()
            valid_pairs_for_classes = []
            for pair in self.pairs:
                mask_path = pair[1] if len(pair) > 1 else None
                if not mask_path:
                    continue
                try:
                    mask = read_mask_with_fallback(str(mask_path))
                    if mask is None or mask.size == 0:
                        continue
                    if mask.dtype not in (np.uint8, np.uint16):
                        mask = mask.astype(np.uint8)
                    uniq = np.unique(mask)
                    unique_vals.update(int(v) for v in uniq)
                    valid_pairs_for_classes.append((pair[0], mask_path))
                except Exception as e:
                    self.log(f"Ошибка маски {mask_path}: {e}")
            if not valid_pairs_for_classes:
                self.original_ids = []
                self._updating_class_table = True
                self.class_table.blockSignals(True)
                try:
                    self.class_table.setRowCount(0)
                finally:
                    self.class_table.blockSignals(False)
                    self._updating_class_table = False
                return
            self.pairs = valid_pairs_for_classes
            all_vals = sorted(unique_vals)
            self.original_ids = all_vals
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(len(all_vals))
                for i, old_id in enumerate(all_vals):
                    new_item = QTableWidgetItem(str(i))
                    new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 0, new_item)
                    orig_item = QTableWidgetItem(str(old_id))
                    orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 1, orig_item)
                    name_item = QTableWidgetItem(str(old_id))
                    self.class_table.setItem(i, 2, name_item)
                    cb = QCheckBox()
                    cb.setChecked(True)
                    cb.stateChanged.connect(self.update_included_classes)
                    self.class_table.setCellWidget(i, 4, cb)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            self.update_included_classes()

        else:
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(0)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            self.original_ids = []
            self.update_included_classes()

    def update_included_classes(self):
        if getattr(self, '_updating_class_table', False):
            return

        self._updating_class_table = True
        self.class_table.blockSignals(True)
        try:
            if self.dataset_type == 0:
                self.included_orig_ids = set()
                included_rows = []
                for row in range(self.class_table.rowCount()):
                    cb = self.class_table.cellWidget(row, 4)
                    if cb and cb.isChecked():
                        try:
                            original = int(self.class_table.item(row, 1).text())
                            self.included_orig_ids.add(original)
                            included_rows.append((row, original))
                        except (ValueError, TypeError):
                            continue

                included_rows.sort(key=lambda t: t[1])
                self.class_remap = {}
                included_set = set()
                for new_id, (row, orig) in enumerate(included_rows):
                    self.class_remap[orig] = new_id
                    included_set.add(row)
                    item = self.class_table.item(row, 0)
                    if item is not None and item.text() != str(new_id):
                        item.setText(str(new_id))

                for row in range(self.class_table.rowCount()):
                    if row not in included_set:
                        item = self.class_table.item(row, 0)
                        if item is not None and item.text() != "-":
                            item.setText("-")

                self.unique_classes = list(range(len(self.class_remap)))
                self.num_classes = len(self.class_remap)
                self.mask_class_remap = {}

            elif self.dataset_type == 1:
                if self.class_table.rowCount() == 0:
                    return
                self.included_orig_ids = set()
                for row in range(self.class_table.rowCount()):
                    cb = self.class_table.cellWidget(row, 4)
                    if cb and cb.isChecked():
                        try:
                            orig_id = int(self.class_table.item(row, 1).text())
                            self.included_orig_ids.add(orig_id)
                        except (ValueError, TypeError, AttributeError):
                            continue
                sorted_included = sorted(self.included_orig_ids)
                self.mask_class_remap = {old: new for new, old in enumerate(sorted_included)}
                self.unique_classes = list(range(len(sorted_included)))
                self.num_classes = len(self.unique_classes)
                for row in range(self.class_table.rowCount()):
                    try:
                        orig_id = int(self.class_table.item(row, 1).text())
                        item = self.class_table.item(row, 0)
                        if item is None:
                            continue
                        new_text = str(self.mask_class_remap[orig_id]) \
                            if orig_id in self.mask_class_remap else "-"
                        if item.text() != new_text:
                            item.setText(new_text)
                    except (ValueError, TypeError, AttributeError):
                        continue
            else:
                self.included_orig_ids = set()
                self.class_remap = {}
                self.mask_class_remap = {}
                self.unique_classes = []
                self.num_classes = 0
        finally:
            self.class_table.blockSignals(False)
            self._updating_class_table = False

        self._refresh_mask_pixel_column()

    def on_class_table_item_changed(self, item):
        if getattr(self, '_updating_class_table', False):
            return
        if item.column() == 0:
            self.update_included_classes()

    def _update_pair_count_label(self, extra_text, error=False):
        text = f"Всего пар: {len(self.pairs)}"
        if extra_text:
            text += f"  {extra_text}"
        self.pair_count_label.setText(text)
        if error:
            self.pair_count_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pair_count_label.setStyleSheet("")