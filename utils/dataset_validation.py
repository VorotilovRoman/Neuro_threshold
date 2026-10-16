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