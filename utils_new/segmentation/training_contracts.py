"""Mask and training-parameter contracts shared by segmentation workers."""

import numpy as np
from PIL import Image


def validate_mask_files(paths, num_classes, ignore_index=None):
    """Read EVERY source mask before training; never silently relabel bad IDs."""
    for path in paths:
        with Image.open(path) as source:
            mask = np.array(source)
        if mask.ndim != 2 or mask.dtype.kind not in "bui":
            raise ValueError(f"{path}: нужна одноканальная индексная маска, не RGB.")
        if num_classes > 1:
            valid = (mask >= 0) & (mask < num_classes)
            if ignore_index is not None:
                valid |= mask == ignore_index
            if not valid.all():
                raise ValueError(
                    f"{path}: индексы маски {np.unique(mask[~valid])[:10].tolist()} вне диапазона 0..{num_classes - 1}."
                )


def apply_decoder_dropout(model, probability):
    """SMP U-Net has no decoder_dropout argument. Apply it to decoder features.

    A forward hook adds no state_dict keys and is disabled by model.eval().
    Inference/export therefore use the same original network parameter names.
    """
    if not np.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError("Dropout должен быть в диапазоне [0, 1).")
    if probability:
        from torch.nn import functional

        model.decoder.register_forward_hook(
            lambda module, inputs, output: functional.dropout2d(
                output, p=probability, training=module.training
            )
        )
