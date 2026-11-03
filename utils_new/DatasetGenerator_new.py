# -*- coding: utf-8 -*-
"""DatasetGenerator_new: V14 canonical annotations -> YOLO/U-Net/COCO.

This module does not import the old augmentation/converter/image_io/exporter
algorithms or import_libs_internal. The engine runs without Qt for integration
and regression testing. A QThread wrapper retains the old constructor and
signal contract for optional future tab migration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Callable

import cv2
import numpy as np

from utils_new.annotations import (
    Annotation, AnnotationSample, ImageSize, MaskEncoding,
    Resize, Flip, TransformPlan, AugmentationPlan, transform_sample, build_augmentation_plan, apply_augmentation_plan,
    convert_annotation, polygon_area, remap_mask_indices, mask_to_annotations,
)
from utils_new.dataset_mask_diagnostics_new import mask_statistics
from utils_new.dataset_io_new import (
    read_image, read_index_mask, read_yolo_labels, write_image, write_index_mask,
    write_yolo_labels, make_index_mask, write_classes_and_yaml, write_coco_split,
)

INPUT_LABELS = 0
INPUT_MASKS = 1
INPUT_IMAGES = 2
FORMAT_HINT_AUTO = 'auto'
SUBTYPE_KEEP = 'keep'


@dataclass
class _Generated:
    stem: str
    extension: str
    source_key: str
    image: Path
    label: Path | None
    mask: Path | None
    size: ImageSize
    annotations: list[Annotation] | None


class DatasetGeneratorCoreNew:
    """Qt-independent generator; constructor parameters match original thread.

    Added options:
      random_seed: deterministic augmentation and splitting, None=unseeded.
      split_by_source: keep augmented variants of one source in same split;
                       False by default for compatibility with existing UI.
      obb_policy: 'promote_to_segment' (exact geometry) or 'refit' (lossy).
      hole_policy: 'error' or 'drop' for cropped polygons.
    """

    def __init__(self,
                 images_folder, labels_folder, output_folder, pairs,
                 train_pct, val_pct, test_pct, target_size,
                 augmentations, has_augmentations, aug_multiplier,
                 class_names, class_mapping, bg_color,
                 input_type, masks_folder, num_classes, mask_class_remap,
                 included_orig_ids=None, included_class_ids=None,
                 total_orig_classes=1, resize_mode='fixed',
                 include_empty_annotations=False,
                 delete_source_after_success=False,
                 input_format=FORMAT_HINT_AUTO,
                 out_yolo_txt=True, out_yolo_subtype=SUBTYPE_KEEP,
                 out_yolo_obb_format='corners', out_masks=False,
                 out_masks_offset=1, out_unet_yaml=False,
                 out_coco=False, mask_pixel_offset_in=1, min_ann_area_px=0,
                 *, random_seed=None, split_by_source=False,
                 obb_policy='promote_to_segment', hole_policy='error',
                 log_cb: Callable[[str], None] | None = None,
                 progress_cb: Callable[[int, int], None] | None = None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.output_folder = str(output_folder)
        self.pairs = list(pairs)
        self.train_pct = float(train_pct)
        self.val_pct = float(val_pct)
        self.test_pct = float(test_pct)
        self.target_size = int(target_size) if target_size is not None else None
        self.augmentations = dict(augmentations or {})
        self.has_augmentations = bool(has_augmentations)
        self.aug_multiplier = max(1, int(aug_multiplier))
        self.class_names = {int(k): str(v) for k, v in (class_names or {}).items()}
        self.class_mapping = dict(class_mapping or {})
        self.bg_color = tuple(bg_color) if isinstance(bg_color, (list, tuple)) else int(bg_color)
        self.input_type = int(input_type)
        self.masks_folder = masks_folder
        self.num_classes = int(num_classes)
        self.mask_class_remap = dict(mask_class_remap or {})
        self.included_orig_ids = included_orig_ids
        self.included_class_ids = included_class_ids
        self.total_orig_classes = total_orig_classes
        self.resize_mode = resize_mode
        self.include_empty_annotations = bool(include_empty_annotations)
        self.delete_source_after_success = bool(delete_source_after_success)
        self.input_format = input_format or FORMAT_HINT_AUTO
        self.out_yolo_txt = bool(out_yolo_txt)
        self.out_yolo_subtype = out_yolo_subtype or SUBTYPE_KEEP
        self.out_yolo_obb_format = out_yolo_obb_format or 'corners'
        self.out_masks = bool(out_masks)
        self.out_masks_offset = max(1, int(out_masks_offset))
        self.out_unet_yaml = bool(out_unet_yaml)
        self.out_coco = bool(out_coco)
        self.mask_pixel_offset_in = int(mask_pixel_offset_in)
        self.min_ann_area_px = max(0, int(min_ann_area_px or 0))
        self.random_seed = random_seed
        self.split_by_source = bool(split_by_source)
        self.obb_policy = obb_policy
        self.hole_policy = hole_policy
        self._log_cb = log_cb or (lambda _message: None)
        self._progress_cb = progress_cb or (lambda _cur, _total: None)
        self._is_cancelled = False
        self._rng = random.Random(random_seed)
        self._processed_source_files: list[tuple[str, str | None, str | None]] = []
        self._dropped_ann_count = 0
        self._used_stems: set[str] = set()
        self._written = 0
        self._skipped = 0
        self._warnings: list[str] = []
        self._mask_value_totals: dict[int, int] = {}
        self._mask_files_written = 0
        self._mask_negative_files = 0

    def log(self, message):
        self._log_cb(str(message))

    def cancel(self):
        self._is_cancelled = True

    def _check_cancelled(self):
        if self._is_cancelled:
            raise _Cancelled('Генерация отменена пользователем')

    def _validate_config(self):
        if self.input_type not in (INPUT_LABELS, INPUT_MASKS, INPUT_IMAGES):
            raise ValueError(f'Unsupported input_type: {self.input_type}')
        if self.out_yolo_subtype not in ('keep', 'detect', 'obb', 'segment'):
            raise ValueError('Unsupported YOLO output subtype')
        if self.out_yolo_obb_format not in ('corners', 'xywhr'):
            raise ValueError('Unsupported OBB output format')
        if self.resize_mode not in ('fixed', 'stretch', 'keep'):
            raise ValueError('resize_mode must be fixed/stretch/keep')
        if self.target_size is not None and self.target_size < 1:
            raise ValueError('target_size must be positive')
        if self.obb_policy not in ('promote_to_segment', 'refit', 'error'):
            raise ValueError('Invalid OBB geometry policy')
        if self.hole_policy not in ('error', 'drop'):
            raise ValueError('Invalid hole policy')
        if any(x < 0 or x > 1 for x in (self.train_pct, self.val_pct, self.test_pct)):
            raise ValueError('Split fractions must be in [0,1]')
        if abs(self.train_pct+self.val_pct+self.test_pct-1) > 0.02:
            raise ValueError('Train/val/test split must total 100%')
        if self.input_type != INPUT_IMAGES and not (self.out_yolo_txt or self.out_masks or self.out_coco):
            raise ValueError('Select at least one output format')
        if self.input_type == INPUT_IMAGES and (self.out_masks or self.out_coco):
            self.log('[INFO] Images-only input: mask/COCO outputs ignored')
        if self.out_unet_yaml and not self.out_masks and self.input_type != INPUT_IMAGES:
            raise ValueError('U-Net YAML requires masks output')
        if self.out_masks_offset > 65535:
            raise ValueError('PNG cannot encode mask class indices > 65535')
        if self.input_type == INPUT_MASKS and not self.mask_class_remap \
                and not self.include_empty_annotations:
            raise ValueError('Mask source requires orig_pixel -> class mapping, '
                             'unless only empty masks are requested')
        if not self.pairs:
            raise ValueError('No input images/pairs')

    @staticmethod
    def _dense_map(mapping):
        if not mapping:
            return {}
        vals = sorted(set(int(v) for v in mapping.values()))
        renum = {v: i for i, v in enumerate(vals)}
        return {int(k): renum[int(v)] for k, v in mapping.items()}

    def _map_annotations(self, annotations):
        mapping = self._dense_map(self.class_mapping)
        if not mapping:
            # No selected classes must not accidentally retain original labels.
            # An empty map with explicitly supplied class_names means the
            # canonical input already uses those IDs (headless API case).
            return [a for a in annotations if a.class_id in self.class_names]
        return [Annotation(a.kind, mapping[a.class_id], a.points_px)
                for a in annotations if a.class_id in mapping]

    def _map_mask(self, mask):
        # Source mapping is physical pixel value -> dense class id.
        # For mask source 0 is always the background, even if a map contains 0.
        mapping = {old: new for old, new in self._dense_map(self.mask_class_remap).items()
                   if old != 0}
        source_encoding = MaskEncoding(offset=0, background_value=0, allow_ambiguous_zero=True)
        output_encoding = MaskEncoding(offset=self.out_masks_offset, background_value=0)
        return remap_mask_indices(mask, mapping, source_encoding=source_encoding,
                                  target_encoding=output_encoding)

    def _verify_mask_export(self, mask, mask_path, *, has_objects: bool):
        """Fail loudly if a nonempty annotation list yields an all-black DATA mask.

        Small indexed values (1,2,3) are intentionally nearly black in a
        standard viewer. Report the exact pixel histogram so users can tell
        valid class-index masks apart from empty masks.
        """
        from utils_new.dataset_files_new import read_index_mask
        info = mask_statistics(mask)
        if has_objects and info.is_empty:
            raise ValueError(
                f'Generated mask has ONLY background pixels although annotations '
                f'are present: {Path(mask_path).name}; check YOLO coordinates, '
                'annotation format and object sizes')
        # Check the actual disk representation on the first three masks.
        # Decode every mask here would be slow on large/high-resolution datasets.
        if self._mask_files_written < 3:
            saved = read_index_mask(mask_path)
            if not np.array_equal(saved, mask):
                raise OSError(f'Saved PNG mask does not match rasterized pixels: {mask_path}')
            self.log(f'[mask check] {Path(mask_path).name}: {info.brief()}')
            if info.foreground_pixels and max(info.counts) <= 10:
                self.log('[mask info] Индексные значения 1–10 выглядят '
                         'почти чёрными в обычном просмотрщике PNG; '
                         'для просмотра используйте цветной предпросмотр')
        if info.is_empty:
            self._mask_negative_files += 1
            self.log(f'[mask warning] {Path(mask_path).name}: only background (negative sample)')
        self._mask_files_written += 1
        for value, count in info.counts.items():
            self._mask_value_totals[value] = self._mask_value_totals.get(value, 0) + count

    def _filter_annotations(self, anns):
        if not self.min_ann_area_px:
            return list(anns)
        out = []
        for ann in anns:
            if polygon_area(ann.points_px) >= self.min_ann_area_px:
                out.append(ann)
            else:
                self._dropped_ann_count += 1
        return out

    def _filter_mask(self, mask):
        if not self.min_ann_area_px:
            return mask
        result = mask.copy()
        for value in np.unique(mask):
            if int(value) == 0:
                continue
            n, comps, stats, _ = cv2.connectedComponentsWithStats(
                (mask == value).astype(np.uint8), connectivity=8)
            for idx in range(1, n):
                if int(stats[idx, cv2.CC_STAT_AREA]) < self.min_ann_area_px:
                    result[comps == idx] = 0
                    self._dropped_ann_count += 1
        return result

    def _resize(self, sample):
        if self.target_size is None or self.resize_mode == 'keep':
            return sample.detached()
        mode = 'letterbox' if self.resize_mode == 'fixed' else 'stretch'
        return transform_sample(sample, [Resize((self.target_size, self.target_size), mode=mode,
                                                fill_color=self.bg_color)],
                                obb_policy=self.obb_policy, hole_policy=self.hole_policy,
                                multi_policy='all', fill_color=self.bg_color)

    def _make_variant(self, sample, *, augmented=False, require_operation=False):
        if not augmented:
            return self._resize(sample)
        config = dict(self.augmentations)
        if self.input_type == INPUT_MASKS:
            config = {k: config.get(k, False) for k in
                      ('horizontal_flip', 'vertical_flip', 'rotate_90')}
        plan = build_augmentation_plan(sample.size, config, rng=self._rng)
        # In 1x mode the original is REPLACED by an augmented variant.
        # A sole flip checkbox normally has p=.5 and can produce an empty plan.
        # Avoid that no-op by selecting an enabled flip deterministically if
        # the randomized plan contains no operations.
        if require_operation and not plan.transform.steps:
            if config.get('horizontal_flip', False):
                fallback = Flip(horizontal=True)
            elif config.get('vertical_flip', False):
                fallback = Flip(vertical=True)
            else:
                raise ValueError('1x augmentation: no enabled transformation produced an operation')
            plan = AugmentationPlan(TransformPlan((fallback,)))
        transformed, report = apply_augmentation_plan(
            sample, plan, obb_policy=self.obb_policy,
            hole_policy=self.hole_policy, multi_policy='all',
            fill_color=self.bg_color, return_report=True)
        for note in report.notes:
            self.log('[geometry] ' + note)
        return self._resize(transformed)

    def _get_name(self, img_path):
        stem = Path(img_path).stem
        candidate = stem
        n = 2
        while candidate in self._used_stems:
            candidate = f'{stem}__{n}'
            n += 1
        self._used_stems.add(candidate)
        return candidate

    def _reserve_stem(self, proposed):
        candidate = proposed
        n = 2
        while candidate in self._used_stems:
            candidate = f'{proposed}__{n}'
            n += 1
        self._used_stems.add(candidate)
        return candidate

    def _sanitize_input_annotations(self, annotations, image_path):
        """Repair documented topology defects and isolate unrecoverable labels."""
        from utils_new.annotations.geometry import (GeometryError,
                                                    sanitize_annotation_topology)
        result = []
        for index, ann in enumerate(annotations, 1):
            try:
                fixed, diagnostics = sanitize_annotation_topology(
                    ann, hole_policy=self.hole_policy)
                for detail in diagnostics:
                    self.log(f'[geometry repair] {Path(image_path).name}: '
                             f'object {index}, class={ann.class_id}, '
                             f'{ann.kind}: {detail}')
                result.extend(fixed)
            except GeometryError as exc:
                self.log(f'[geometry warning] {Path(image_path).name}: '
                         f'object {index}, class={ann.class_id}, '
                         f'{ann.kind} excluded: {exc}')
                self._dropped_ann_count += 1
        return result

    def _prepare_sample(self, pair):
        image_path = str(pair[0])
        image = read_image(image_path)
        size = ImageSize(image.shape[1], image.shape[0])
        if self.input_type == INPUT_IMAGES:
            return AnnotationSample(image=image), (image_path, None, None), False
        if self.input_type == INPUT_LABELS:
            label_path = str(pair[1]) if len(pair) > 1 and pair[1] else None
            annotations = read_yolo_labels(label_path, size,
                                           format_hint=self.input_format, log=self.log)
            originally_empty = not annotations
            mapped = self._map_annotations(annotations)
            # A malformed polygon must not invalidate the entire image.  Validate
            # each object before transforms; log repairs/individual exclusions.
            mapped = self._sanitize_input_annotations(mapped, image_path)
            if not mapped and not (originally_empty and self.include_empty_annotations):
                self.log(f'[skip] no annotations after mapping: {image_path}')
                return None, None, originally_empty
            return AnnotationSample(image=image, annotations=mapped), (image_path, label_path, None), originally_empty
        mask_path = (str(pair[2]) if len(pair)>2 and pair[2] else
                     str(pair[1]) if len(pair)>1 and pair[1] else None)
        if mask_path is None:
            raise ValueError(f'Mask path missing for {image_path}')
        mask = read_index_mask(mask_path)
        if mask.shape != image.shape[:2]:
            raise ValueError(f'Image/mask size mismatch: {image_path}, {mask_path}')
        mapped = self._map_mask(mask)
        if not np.any(mapped) and not self.include_empty_annotations:
            self.log(f'[skip] empty mask after class remap: {mask_path}')
            return None, None, True
        return AnnotationSample(image=image, mask=mapped,
                                mask_encoding=MaskEncoding(offset=self.out_masks_offset)),\
               (image_path, None, mask_path), not np.any(mapped)

    def _generate_pair(self, pair, stage, generated, target_count):
        self._check_cancelled()
        sample, source, originally_empty = self._prepare_sample(pair)
        if sample is None:
            self._skipped += 1
            return
        image_path = source[0]
        stem = self._get_name(image_path)
        extension = Path(image_path).suffix.lower()
        if extension not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            extension = '.png'
        source_key = str(Path(image_path).resolve())
        prior_hashes = set()
        attempts = 1 if not self.has_augmentations else self.aug_multiplier
        replace_original = bool(self.has_augmentations and self.aug_multiplier == 1)
        for copy_idx in range(attempts):
            self._check_cancelled()
            max_attempts = 12 if replace_original else (1 if copy_idx == 0 else 10)
            for attempt in range(max_attempts):
                if self._is_cancelled:
                    self._check_cancelled()
                try:
                    variant = self._make_variant(
                        sample, augmented=(copy_idx > 0 or replace_original),
                        require_operation=replace_original)
                except Exception as exc:
                    # Do not discard the ORIGINAL source because one randomized
                    # transform produced invalid geometry. Retry 1x, then fall
                    # back to one unaugmented sample with an explicit warning.
                    from utils_new.annotations.geometry import GeometryError
                    if not isinstance(exc, GeometryError):
                        raise
                    if replace_original and attempt + 1 < max_attempts:
                        if attempt == 0:
                            self.log(f'[augmentation retry] {Path(image_path).name}: '
                                     f'{exc}; retrying up to {max_attempts} times')
                        continue
                    if replace_original:
                        self.log(f'[1x fallback] {Path(image_path).name}: '
                                 f'all augmented variants failed: {exc}; '
                                 'writing one unchanged source')
                        variant = self._resize(sample)
                    elif copy_idx > 0:
                        self.log(f'[augmentation warning] {Path(image_path).name}: '
                                 f'augmented copy {copy_idx} excluded: {exc}')
                        break
                    else:
                        raise
                if variant.annotations:
                    filtered = self._filter_annotations(variant.annotations)
                    variant = AnnotationSample(variant.image, filtered, variant.mask,
                                               variant.mask_encoding)
                elif variant.mask is not None:
                    variant = AnnotationSample(variant.image, variant.annotations,
                                               self._filter_mask(variant.mask), variant.mask_encoding)
                if self.input_type == INPUT_LABELS and not variant.annotations and not originally_empty:
                    if copy_idx == 0:
                        if self.include_empty_annotations:
                            self.log(f'[negative] all objects filtered: {image_path}')
                        elif replace_original and attempt + 1 < max_attempts:
                            # Retry a new random transform: 1x must not lose a
                            # valid source because a crop happened to remove it.
                            continue
                        elif replace_original:
                            # Conservative fallback: retain one valid source
                            # instead of dropping it; log that no safe random
                            # variant was found.
                            variant = self._resize(sample)
                            kept = self._filter_annotations(variant.annotations)
                            if not kept:
                                self.log(f'[skip] original annotations fail area filter: {image_path}')
                                return
                            variant = AnnotationSample(variant.image, kept, variant.mask,
                                                       variant.mask_encoding)
                            self.log(f'[1x fallback] could not preserve annotations after augmentation: {image_path}')
                        else:
                            self.log(f'[skip] all objects filtered: {image_path}')
                            return
                    else:
                        continue
                # Mask input may become empty after filtering.
                if self.input_type == INPUT_MASKS and variant.mask is not None and not np.any(variant.mask):
                    if not self.include_empty_annotations:
                        if copy_idx == 0 and replace_original and attempt + 1 < max_attempts:
                            continue
                        if copy_idx == 0 and replace_original:
                            variant = self._resize(sample)
                            fallback_mask = self._filter_mask(variant.mask)
                            if not np.any(fallback_mask):
                                self.log(f'[skip] original mask fails area filter: {image_path}')
                                return
                            variant = AnnotationSample(variant.image, variant.annotations,
                                                       fallback_mask, variant.mask_encoding)
                            self.log(f'[1x fallback] could not preserve mask after augmentation: {image_path}')
                        elif copy_idx == 0:
                            self.log(f'[skip] empty mask after area filter: {image_path}')
                            return
                        else:
                            continue
                digest = hashlib.md5(variant.image.tobytes()).hexdigest()
                if copy_idx and digest in prior_hashes:
                    continue
                prior_hashes.add(digest)
                proposed = stem if not copy_idx else f'{stem}_aug{copy_idx}'
                output_stem = proposed if copy_idx == 0 else self._reserve_stem(proposed)
                img_file = stage/'all'/'images'/f'{output_stem}{extension}'
                write_image(img_file, variant.image)
                annotations = None
                label_file = None
                mask_file = None
                if self.input_type == INPUT_LABELS:
                    annotations = [convert_annotation(ann, self.out_yolo_subtype, allow_lossy=True)
                                   for ann in variant.annotations]
                    if self.out_yolo_txt:
                        label_file = stage/'all'/'labels'/f'{output_stem}.txt'
                        write_yolo_labels(label_file, annotations, variant.size,
                                          obb_format=self.out_yolo_obb_format)
                    if self.out_masks:
                        mask_file = stage/'all'/'masks'/f'{output_stem}.png'
                        mask_array = make_index_mask(annotations, variant.size,
                                                     offset=self.out_masks_offset)
                        write_index_mask(mask_file, mask_array)
                        self._verify_mask_export(mask_array, mask_file,
                                                 has_objects=bool(annotations))
                elif self.input_type == INPUT_MASKS:
                    mask_file = stage/'all'/'masks'/f'{output_stem}.png'
                    write_index_mask(mask_file, variant.mask)
                    if self.out_masks:
                        self._verify_mask_export(variant.mask, mask_file,
                                                 has_objects=bool(np.any(variant.mask)))
                    if self.out_yolo_txt:
                        raw = mask_to_annotations(variant.mask, encoding=variant.mask_encoding,
                                                  kind='segment', holes='drop')
                        annotations = [convert_annotation(ann, self.out_yolo_subtype, allow_lossy=True)
                                       for ann in raw]
                        label_file = stage/'all'/'labels'/f'{output_stem}.txt'
                        write_yolo_labels(label_file, annotations, variant.size,
                                          obb_format=self.out_yolo_obb_format)
                        self.log(f'[lossy] mask→YOLO: {output_stem}')
                generated.append(_Generated(output_stem, extension, source_key,
                                            img_file, label_file, mask_file,
                                            variant.size, annotations))
                self._written += 1
                if copy_idx == 0:
                    self._processed_source_files.append(source)
                self._progress_cb(len(generated), target_count)
                break
            else:
                self.log(f'[skip] cannot generate unique augmentation: {stem}_aug{copy_idx} ({max_attempts} attempts)')

    def _assign_splits(self, records):
        self._rng.shuffle(records)
        if not self.split_by_source:
            n_train = int(len(records)*self.train_pct)
            n_val = int(len(records)*self.val_pct)
            return {'train': records[:n_train], 'val': records[n_train:n_train+n_val],
                    'test': records[n_train+n_val:]}
        groups = {}
        for record in records:
            groups.setdefault(record.source_key, []).append(record)
        keys = list(groups)
        self._rng.shuffle(keys)
        # Group-aware greedy allocation; protects against source leakage, even
        # when exact percentage is impossible due to large augmentation groups.
        targets = {'train': len(records)*self.train_pct,
                   'val': len(records)*self.val_pct,
                   'test': len(records)*self.test_pct}
        out = {'train': [], 'val': [], 'test': []}
        for key in keys:
            possible = [s for s in ('train','val','test') if targets[s] > 0]
            split = max(possible or ('train','val','test'),
                        key=lambda s: targets[s]-len(out[s]))
            out[split].extend(groups[key])
        return out

    def _distribute_and_export(self, stage, records):
        splits = self._assign_splits(records)
        for name, collection in splits.items():
            self.log(f'[split] {name}: {len(collection)} images')
            coco_rows = []
            for rec in collection:
                self._check_cancelled()
                image_target = stage/'images'/name/rec.image.name
                image_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rec.image), str(image_target))
                if rec.label is not None:
                    label_target = stage/'labels'/name/rec.label.name
                    label_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(rec.label), str(label_target))
                if rec.mask is not None:
                    mask_target = stage/'masks'/name/rec.mask.name
                    mask_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(rec.mask), str(mask_target))
                else:
                    mask_target = None
                if self.out_coco and self.input_type != INPUT_IMAGES:
                    mask = (read_index_mask(mask_target) if self.input_type == INPUT_MASKS else None)
                    coco_rows.append((rec.image.name, rec.size,
                                      None if self.input_type == INPUT_MASKS else rec.annotations, mask))
            if self.out_coco and self.input_type != INPUT_IMAGES and self.class_names:
                count = write_coco_split(stage, name, coco_rows, self.class_names,
                                         masks_offset=self.out_masks_offset, log=self.log)
                self.log(f'[COCO] {name}: {count} annotations')
        # metadata YAML refers to the FINAL folder, not the temporary stage.
        write_classes_and_yaml(stage, self.class_names,
                               out_yolo_txt=self.out_yolo_txt and self.input_type != INPUT_IMAGES,
                               out_unet_yaml=self.out_unet_yaml and self.input_type != INPUT_IMAGES,
                               out_masks_offset=self.out_masks_offset,
                               input_type=self.input_type,
                               dataset_path=self.output_folder)

    def _publish(self, stage, destination):
        # Publishing is postponed until ALL computations, masks and JSON/YAML
        # have completed. A failed/cancelled build never deletes the source.
        for item in sorted(stage.rglob('*')):
            if not item.is_file() or 'all' in item.relative_to(stage).parts:
                continue
            self._check_cancelled()
            relative = item.relative_to(stage)
            if relative.parts[0] == 'masks' and not self.out_masks:
                continue
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(item, target)

    def _delete_processed_sources(self):
        destination = Path(self.output_folder).resolve()
        seen = set()
        deleted = 0
        for source_set in self._processed_source_files:
            for path in source_set:
                if not path:
                    continue
                source = Path(path).resolve()
                if source in seen or source == destination or destination in source.parents:
                    continue
                seen.add(source)
                if not source.is_file():
                    continue
                try:
                    source.unlink()
                    deleted += 1
                except OSError as exc:
                    self.log(f'[delete] {source}: {exc}')
        self.log(f'[delete] {deleted} source files removed after successful generation')

    def execute(self):
        """Run synchronously; return (success, message) for Qt/headless clients."""
        self._written = self._skipped = self._dropped_ann_count = 0
        self._processed_source_files.clear()
        self._used_stems.clear()
        self._mask_value_totals.clear()
        self._mask_files_written = 0
        self._mask_negative_files = 0
        self._rng = random.Random(self.random_seed)
        self._is_cancelled = bool(self._is_cancelled)
        try:
            self._validate_config()
            self._check_cancelled()
            destination = Path(self.output_folder)
            destination.mkdir(parents=True, exist_ok=True)
            total = len(self.pairs)*(self.aug_multiplier if self.has_augmentations else 1)
            self.log('[DatasetGenerator_new] start, canonical pixel geometry V14')
            if self.has_augmentations and self.aug_multiplier == 1:
                self.log('[augmentation] 1x: each source replaced by one augmented output, no extra copies')
            if self.input_type == INPUT_MASKS and any(self.augmentations.get(k, False)
                            for k in ('random_rotate','shear','random_crop')):
                self.log('[INFO] Mask input: arbitrary rotate/shear/crop disabled (lossless geometry policy)')
            if self.out_masks_offset != 1:
                self.log(f'[masks] output offset = {self.out_masks_offset}')
            with tempfile.TemporaryDirectory(prefix='.dataset_gen_new_', dir=destination) as tmp:
                stage = Path(tmp)
                records = []
                for pair in self.pairs:
                    self._check_cancelled()
                    try:
                        self._generate_pair(pair, stage, records, total)
                    except _Cancelled:
                        raise
                    except Exception as exc:
                        self._skipped += 1
                        self.log(f'[skip] {pair[0]}: {type(exc).__name__}: {exc}')
                if not records:
                    raise ValueError('No valid images generated')
                self._check_cancelled()
                self._distribute_and_export(stage, records)
                self._check_cancelled()
                self._publish(stage, destination)
            if self._mask_files_written:
                hist = ', '.join(f'{v}:{n}' for v, n in sorted(self._mask_value_totals.items())[:24])
                if len(self._mask_value_totals) > 24:
                    hist += f', ... ({len(self._mask_value_totals)} values total)'
                self.log(f'[mask summary] files={self._mask_files_written}, '
                         f'background_only={self._mask_negative_files}, '
                         f'pixel_counts={{ {hist} }}')
            if self.delete_source_after_success:
                self._delete_processed_sources()
            self.log(f'[DatasetGenerator_new] done: images={self._written}, skipped={self._skipped}, '
                     f'filtered_annotations={self._dropped_ann_count}')
            return True, f'Датасет создан: {self._written} изображений'
        except _Cancelled as exc:
            self.log(str(exc))
            return False, str(exc)
        except Exception as exc:
            import traceback
            self.log(f'[DatasetGenerator_new] ERROR: {exc}\n{traceback.format_exc()}')
            return False, str(exc)


class _Cancelled(Exception):
    pass


try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = None


if QThread is not None:
    class DatasetGeneratorThread(QThread):
        """Drop-in Qt adapter for preparing_dataset_yaml.py (not switched by default)."""
        log_signal = pyqtSignal(str)
        progress_signal = pyqtSignal(int, int)
        finished_signal = pyqtSignal(bool, str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            kwargs['log_cb'] = self.log_signal.emit
            kwargs['progress_cb'] = self.progress_signal.emit
            self.engine = DatasetGeneratorCoreNew(*args, **kwargs)

        def cancel(self):
            self.engine.cancel()

        def run(self):
            success, message = self.engine.execute()
            self.finished_signal.emit(success, message)
else:
    class DatasetGeneratorThread:
        def __init__(self, *args, **kwargs):
            raise ImportError('DatasetGeneratorThread requires PyQt5; use DatasetGeneratorCoreNew for headless use')


DatasetGeneratorThreadNew = DatasetGeneratorThread
