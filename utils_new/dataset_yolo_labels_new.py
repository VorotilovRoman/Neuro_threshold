# -*- coding: utf-8 -*-
"""YOLO TXT decode/encode at the boundary of the pixel-space annotation core."""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
from utils_new.annotations import Annotation, ImageSize, from_legacy, to_legacy
from utils_new.annotations.geometry import is_rectangle
LABEL_HINTS = {'auto','mixed','detect','obb_corners','obb_xywhr','obb_xywhr_legacy','segment'}

def _xywhr_points(vals, size=None):
    """Decode normalized ``cx,cy,w,h,angle`` into rectangle corners.

    V19.4 contract (``size`` supplied): cx and width are normalized by image
    width, cy and height by image height; the angle is measured in *physical
    pixel coordinates*. This round-trips actual OBB on non-square images.

    ``size=None`` is retained only for explicitly requested legacy decoding,
    where all five geometric values lived in normalized (distorted) space.
    """
    cx, cy, w, h, radians = map(float, vals)
    if size is not None:
        cx *= size.width
        cy *= size.height
        w *= size.width
        h *= size.height
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    v1 = np.array((w*cos_a/2, w*sin_a/2), dtype=np.float64)
    v2 = np.array((-h*sin_a/2, h*cos_a/2), dtype=np.float64)
    c = np.array((cx, cy), dtype=np.float64)
    corners = np.array([c+v1+v2, c+v1-v2, c-v1-v2, c-v1+v2])
    if size is not None:
        corners[:, 0] /= size.width
        corners[:, 1] /= size.height
    return corners.flatten().tolist()


def read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None):
    """Read YOLO txt variants into canonical pixel-space annotations.

    9 tokens are OBB in auto mode. Explicit 'segment' disambiguates a 4-point
    polygon. Invalid rows are logged and skipped as in legacy image_io.
    """
    if format_hint not in LABEL_HINTS:
        raise ValueError(f'Unsupported YOLO input format {format_hint!r}')
    if path is None or not Path(path).exists():
        return []
    result = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            parts = line.split()
            cls = int(parts[0])
            numbers = list(map(float, parts[1:]))
            count = len(parts)
            if format_hint == 'detect':
                kind = 'detect' if count == 5 else None
            elif format_hint == 'obb_corners':
                kind = 'obb' if count == 9 else None
            elif format_hint in ('obb_xywhr', 'obb_xywhr_legacy'):
                kind = format_hint if count == 6 else None
            elif format_hint == 'segment':
                kind = 'segment' if count >= 7 and (count-1)%2 == 0 else None
            else:
                kind = ('detect' if count == 5 else 'obb_xywhr' if count == 6
                        else 'obb' if count == 9 else 'segment' if count >= 7 and (count-1)%2 == 0
                        else None)
            if kind is None:
                raise ValueError(f'{count} tokens incompatible with format_hint={format_hint}')
            if kind in ('obb_xywhr', 'obb_xywhr_legacy'):
                legacy_xywhr = kind.endswith('_legacy')
                kind, numbers = 'obb', _xywhr_points(numbers, None if legacy_xywhr else size)
                if legacy_xywhr and log:
                    log(f'[labels] {path}:{line_no}: explicit legacy normalized-space xywhr decoding')
            elif kind == 'obb' and count == 9 and format_hint in ('auto', 'mixed'):
                # 9 tokens can describe OBB OR a 4-point Segment. The file
                # format cannot disambiguate them, but a nonrectangular vertex
                # set cannot be an OBB. Avoid misclassifying that as a rotated
                # rectangle and later rejecting the entire image on augment.
                p = np.asarray(numbers, dtype=np.float64).reshape(4, 2)
                p_px = p * np.array([size.width, size.height], dtype=float)
                hull = cv2.convexHull(p_px.astype(np.float32)).reshape(-1, 2)
                if len(hull) == 4 and is_rectangle(hull, tol=2e-3):
                    # Repair only vertex ORDER, not polygon outline: a set
                    # of four rectangle corners is unambiguous geometry.
                    if not is_rectangle(p_px, tol=2e-3):
                        p = hull.astype(np.float64)
                        p[:, 0] /= size.width
                        p[:, 1] /= size.height
                        numbers = p.flatten().tolist()
                        if log:
                            log(f'[labels] {path}:{line_no}: OBB corner order repaired')
                else:
                    kind = 'segment'
                    if log:
                        log(f'[labels] {path}:{line_no}: 9-token polygon is not '
                            'rectangular; interpreted as Segment (choose '
                            'obb_corners/segment explicitly if needed)')
            value = ((kind, cls, *numbers) if kind == 'detect' else (kind, cls, numbers))
            result.append(from_legacy(value, size))
        except (TypeError, ValueError, IndexError) as exc:
            if log:
                log(f'[labels] {path}:{line_no}: {exc}; запись пропущена')
    return result



def _normalized_xywhr(ann, size):
    """OBB as ``cx/W, cy/H, width_px/W, height_px/H, radians_px``.

    The final angle is in *pixel-space*; it is not an angle in normalized
    coordinates. A proper rectangle remains a rectangle after decoding even
    when width and height differ substantially.
    """
    if ann.kind != 'obb':
        raise TypeError('xywhr conversion requires OBB annotation')
    points = np.asarray(ann.points_px, dtype=np.float64).reshape(4, 2)
    if not is_rectangle(points, tol=2e-3):
        raise ValueError('xywhr requires true rectangular OBB in pixel coordinates')
    center = points.mean(axis=0)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    idx = int(np.argmax(lengths))
    width_px = float((lengths[idx] + lengths[(idx+2) % 4]) / 2.)
    height_px = float((lengths[(idx+1) % 4] + lengths[(idx+3) % 4]) / 2.)
    angle = math.atan2(float(edges[idx, 1]), float(edges[idx, 0])) % math.pi
    return [float(center[0]/size.width), float(center[1]/size.height),
            width_px/size.width, height_px/size.height, angle]


def write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners'):
    if obb_format not in ('corners', 'xywhr'):
        raise ValueError('obb_format must be corners/xywhr')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ann in annotations:
        if not isinstance(ann, Annotation):
            raise TypeError('Expected canonical Annotation objects')
        legacy = to_legacy(ann, size)
        kind = legacy[0]
        if kind == 'detect':
            coords = legacy[2:]
        elif kind == 'obb' and obb_format == 'xywhr':
            coords = _normalized_xywhr(ann, size)
        else:
            coords = legacy[2]
        # Four-point Segment has 9 tokens (class+8), ambiguous with OBB.
        # Insert a collinear midpoint to preserve the exact polygon shape and
        # make AUTO / mixed readers identify Segment unambiguously.
        if kind == 'segment' and len(coords) == 8:
            coords = list(coords)
            midpoint = [(float(coords[0])+float(coords[2]))/2,
                        (float(coords[1])+float(coords[3]))/2]
            coords = coords[:2] + midpoint + coords[2:]
        rows.append(f'{ann.class_id} ' + ' '.join(f'{float(v):.6f}' for v in coords))
    path.write_text(('\n'.join(rows) + '\n') if rows else '', encoding='utf-8')



# The editor/viewer use normalized legacy tuples at their Qt boundary.
# These are thin adaptors around the same canonical parser/writer as generation.
FORMAT_HINT_AUTO = 'auto'
FORMAT_HINT_MIXED = 'mixed'
FORMAT_HINT_DETECT = 'detect'
FORMAT_HINT_OBB_CORNERS = 'obb_corners'
FORMAT_HINT_OBB_XYWHR = 'obb_xywhr'
FORMAT_HINT_SEGMENT = 'segment'


def canonicalize_obb_corners(norm_points, width=1, height=1):
    """Order the original four vertices, without altering their coordinates.

    Full-size width/height are optional so hand-edited normalized coordinates
    can be ordered too; unlike minAreaRect this does not refit the polygon.
    """
    vals = np.asarray(norm_points, dtype=np.float64).reshape(-1)
    if len(vals) != 8 or not np.isfinite(vals).all():
        return list(norm_points)
    points = vals.reshape(4, 2).copy()
    pixels = points * np.array([width, height], dtype=np.float64)
    hull_idx = cv2.convexHull(pixels.astype(np.float32), returnPoints=False)
    if hull_idx is None or len(hull_idx) != 4:
        return list(norm_points)
    points = points[hull_idx.reshape(-1)]
    pixel_ordered = pixels[hull_idx.reshape(-1)]
    first = min(range(4), key=lambda i: (pixel_ordered[i, 1], pixel_ordered[i, 0]))
    return np.roll(points, -first, axis=0).reshape(-1).tolist()





def load_annotations_for_image(label_path, image_path, format_hint='auto', log=None):
    """Read normalized tuples using the associated image's *real* dimensions."""
    from utils_new.dataset_files_new import image_size_from_file
    width, height = image_size_from_file(image_path)
    return load_annotations(label_path, width, height,
                            format_hint=format_hint, log=log)


def save_annotations_for_image(annotations, label_path, image_path,
                               obb_as_xywhr=False):
    """Prevent Save All from using the current (different) image dimensions."""
    from utils_new.dataset_files_new import image_size_from_file
    width, height = image_size_from_file(image_path)
    return save_annotations(annotations, label_path, width, height,
                            obb_as_xywhr=obb_as_xywhr)


def _tolerant_editor_rows(path, size, format_hint='auto'):
    """Recover syntactically valid, geometrically invalid rows for manual repair.

    The strict generator continues using read_yolo_labels() and must not consume
    these tuples. This fallback intentionally preserves vertices and ordering.
    """
    result = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            parts = line.split()
            cls = int(parts[0])
            nums = [float(v) for v in parts[1:]]
            if not np.isfinite(nums).all():
                continue
            count = len(nums)
            if count == 4 and format_hint in ('auto','mixed','detect'):
                result.append(('detect',cls,*nums))
            elif count == 5 and format_hint in ('auto','mixed','obb_xywhr','obb_xywhr_legacy'):
                vals = _xywhr_points(nums, None if format_hint=='obb_xywhr_legacy' else size)
                result.append(('obb',cls,vals))
            elif count >= 6 and count % 2 == 0:
                if format_hint in ('detect','obb_xywhr','obb_xywhr_legacy'):
                    continue
                if format_hint=='obb_corners' and count != 8:
                    continue
                if count == 8 and format_hint in ('auto','mixed','obb_corners'):
                    pixel = np.asarray(nums).reshape(4,2) * [size.width,size.height]
                    hull = cv2.convexHull(pixel.astype(np.float32)).reshape(-1,2)
                    rect = (len(hull)==4 and is_rectangle(hull, tol=2e-3))
                    kind = 'obb' if (rect or format_hint=='obb_corners') else 'segment'
                else:
                    kind = 'segment'
                result.append((kind,cls,nums))
        except (ValueError, TypeError, IndexError, cv2.error):
            continue
    return result


# Edit-mode needs a permissive tuple boundary. Strict core remains unchanged.
def load_annotations(txt_path, img_w, img_h, format_hint='auto', log=None):
    size = ImageSize(int(img_w), int(img_h))
    output = [to_legacy(a, size) for a in
              read_yolo_labels(txt_path, size, format_hint=format_hint, log=log)]
    if txt_path is None or not Path(txt_path).is_file():
        return output
    source_lines = sum(bool(line.strip()) and not line.lstrip().startswith('#')
                       for line in Path(txt_path).read_text(encoding='utf-8-sig').splitlines())
    if len(output) == source_lines:
        return output
    recovered = _tolerant_editor_rows(txt_path, ImageSize(int(img_w),int(img_h)),
                                      format_hint=format_hint)
    if len(recovered) > len(output):
        if log:
            log(f'[editor] {txt_path}: invalid geometry retained for manual editing '
                f'({len(recovered)-len(output)} additional annotations)')
        return recovered
    return output


def save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False):
    try:
        size = ImageSize(int(img_w), int(img_h))
        canonical = [from_legacy(a, size) for a in annotations]
        write_yolo_labels(txt_path, canonical, size,
                          obb_format='xywhr' if obb_as_xywhr else 'corners')
        return True
    except (OSError, TypeError, ValueError, cv2.error):
        pass
    # Invalid annotations can be edited/repaired and must not silently vanish.
    # Preserve normalized values when saving as corners; do not fabricate an
    # xywhr rectangle from invalid geometry.
    if obb_as_xywhr:
        return False
    try:
        rows = []
        for ann in annotations:
            kind, cls = ann[0], int(ann[1])
            if kind == 'detect' and len(ann) == 6:
                coords = ann[2:]
            elif kind in ('obb','segment') and len(ann) == 3:
                coords = list(ann[2])
                if kind == 'obb' and len(coords) != 8:
                    raise ValueError('Invalid OBB coordinate count')
                if kind == 'segment' and (len(coords) < 6 or len(coords) % 2):
                    raise ValueError('Invalid segment coordinate count')
                if kind == 'segment' and len(coords) == 8:
                    coords = coords[:2] + [(coords[0]+coords[2])/2,
                                            (coords[1]+coords[3])/2] + coords[2:]
            else:
                raise ValueError('Unsupported editor annotation')
            if not np.isfinite(coords).all():
                raise ValueError('Nonfinite editor annotation')
            rows.append(f'{cls} ' + ' '.join(f'{float(x):.6f}' for x in coords))
        path = Path(txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(('\n'.join(rows)+'\n') if rows else '', encoding='utf-8')
        return True
    except (OSError, TypeError, ValueError, IndexError):
        return False


# --- Shared save confirmation plan for editor and viewer, without Qt imports. ---
from dataclasses import dataclass

@dataclass(frozen=True)
class LabelSavePlan:
    """Exact output paths and overwrite counts, computed before confirmation."""
    entries: tuple[tuple[str, str], ...]  # (image_path, label_path)
    overwrite_count: int
    create_count: int
    directories: tuple[str, ...]


def build_label_save_plan(entries):
    """Refuse collisions; count only existing *destination* .txt files."""
    seen = set()
    normalized = []
    for image, destination in entries:
        image = str(image)
        destination = str(destination)
        if not destination.lower().endswith('.txt'):
            raise ValueError(f'Expected YOLO .txt destination: {destination}')
        key = str(Path(destination).resolve()).casefold()
        if key in seen:
            raise ValueError(f'Duplicate output label path: {destination}')
        seen.add(key)
        normalized.append((image, destination))
    if not normalized:
        raise ValueError('No loaded image annotations to save')
    old = sum(Path(dest).is_file() for _, dest in normalized)
    directories = tuple(sorted({str(Path(dest).parent) for _, dest in normalized}))
    return LabelSavePlan(tuple(normalized), old, len(normalized) - old, directories)


def format_label_save_confirmation(plan, *, operation='Save', obb_format='corners',
                                   source_kind='images', unloaded_count=0):
    """Human-readable summary for the shared Qt Yes/No dialogs."""
    if not plan.entries:
        raise ValueError('Empty save plan')
    count = len(plan.entries)
    parts = [
        f'{operation}: подтвердить сохранение аннотаций?',
        f'Файлов .txt: {count}',
        f'Перезапись существующих .txt: {plan.overwrite_count}',
        f'Новых .txt: {plan.create_count}',
        f'Формат OBB: {obb_format}',
    ]
    if operation == 'Save All' and unloaded_count:
        parts.append(f'Не загружены/не изменены в памяти — не сохраняются: {unloaded_count}')
    if count == 1:
        parts.extend(('', 'Путь назначения:', plan.entries[0][1]))
    else:
        parts.extend(('', 'Папки назначения:'))
        parts.extend(plan.directories[:8])
        if len(plan.directories) > 8:
            parts.append(f'... и ещё {len(plan.directories) - 8} папок')
        parts.extend(('', 'Примеры файлов назначения:'))
        parts.extend(destination for _, destination in plan.entries[:5])
        if count > 5:
            parts.append(f'... и ещё {count - 5} файлов')
    if source_kind == 'unet':
        parts.extend(('', 'ВНИМАНИЕ: источник — U-Net YAML.',
                      'Сохраняются ТОЛЬКО преобразованные YOLO-аннотации (.txt).',
                      'Исходные PNG-маски НЕ изменяются; обратной записи в маски нет.'))
    if plan.overwrite_count:
        parts.extend(('', 'Существующие файлы .txt будут заменены после подтверждения.'))
    return '\n'.join(parts)
