"""Seeded augmentation plans: random choices materialized before any mutation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .model import AnnotationSample, ImageSize
from .transforms import (
    Crop,
    Flip,
    Resize,
    Rotate,
    Rotate90,
    Shear,
    TransformPlan,
    transform_sample,
)


@dataclass(frozen=True)
class AugmentationPlan:
    transform: TransformPlan
    seed: int | None = None


def _next_size(size: ImageSize, op):
    w, h = size.width, size.height
    if isinstance(op, Rotate90):
        return ImageSize(h, w) if op.k % 2 else size
    if isinstance(op, Crop):
        return ImageSize(op.width, op.height)
    if isinstance(op, Resize):
        return ImageSize(*op.target)
    if isinstance(op, (Rotate, Shear)) and op.expand_canvas:
        if isinstance(op, Rotate):
            r = math.radians(op.angle_deg)
            M = np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])
        else:
            M = np.array([[1, math.tan(math.radians(op.x_deg))],
                          [math.tan(math.radians(op.y_deg)), 1]])
        q = np.array([[0, 0], [w, 0], [w, h], [0, h]]) @ M.T
        span = q.max(axis=0) - q.min(axis=0)
        return ImageSize(max(1, int(np.ceil(span[0]-1e-9))),
                         max(1, int(np.ceil(span[1]-1e-9))))
    return size


def build_augmentation_plan(size: ImageSize, config: dict, *, seed=None, rng=None):
    """Old-style config accepted; repeatable when seed is fixed.

    Enabled flips occur with p=0.5, rotate_90 chooses 1/2/3 clockwise turns,
    random_rotate/shear sample from [-limit,+limit] degrees, then random_crop.
    Crop happens even for image-only samples (unlike legacy implementation).
    """
    if seed is not None and rng is not None:
        raise ValueError('Specify seed OR rng, not both')
    r = random.Random(seed) if rng is None else rng
    if not isinstance(size, ImageSize):
        raise TypeError('Expected ImageSize')
    steps = []
    now = size

    def add(step):
        nonlocal now
        steps.append(step)
        now = _next_size(now, step)

    if config.get('horizontal_flip', False) and r.random() < 0.5:
        add(Flip(horizontal=True))
    if config.get('vertical_flip', False) and r.random() < 0.5:
        add(Flip(vertical=True))
    if config.get('rotate_90', False):
        add(Rotate90(r.choice([1, 2, 3])))
    if config.get('random_rotate', False):
        limit = abs(float(config.get('random_rotate_angle', 10)))
        add(Rotate(r.uniform(-limit, limit), expand_canvas=True))
    if config.get('shear', False):
        limit = abs(float(config.get('shear_angle', 10)))
        add(Shear(x_deg=r.uniform(-limit, limit), expand_canvas=True))
    if config.get('random_crop', False):
        low = float(config.get('random_crop_scale', 0.7))
        low = max(0.1, min(1., low))
        fraction = r.uniform(low, 1.) if low < 1 else 1.
        cw = min(now.width, max(1, 16, int(round(now.width*fraction))))
        ch = min(now.height, max(1, 16, int(round(now.height*fraction))))
        x = r.randint(0, now.width-cw)
        y = r.randint(0, now.height-ch)
        add(Crop(x, y, cw, ch))
    return AugmentationPlan(TransformPlan(steps), seed)


def apply_augmentation_plan(sample: AnnotationSample, plan: AugmentationPlan, **kwargs):
    if not isinstance(plan, AugmentationPlan):
        raise TypeError('Expected AugmentationPlan')
    return transform_sample(sample, plan.transform, **kwargs)
