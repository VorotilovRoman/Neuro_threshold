"""Shared binary/multiclass segmentation losses, class weights and void handling."""

from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F


def class_weight_tensor(weights, num_classes, device="cpu"):
    if weights is None:
        return None
    if isinstance(weights, str):
        weights = json.loads(Path(weights).read_text(encoding="utf-8-sig"))
    result = torch.as_tensor(weights, dtype=torch.float32, device=device)
    allowed = (1, 2) if num_classes == 1 else (num_classes,)
    if (
        result.ndim != 1
        or len(result) not in allowed
        or not torch.isfinite(result).all()
        or (result < 0).any()
        or not result.any()
    ):
        raise ValueError("Некорректные веса классов.")
    if num_classes == 1 and len(result) == 2:
        if result[0] <= 0:
            raise ValueError("Вес фона должен быть положительным.")
        result = (result[1] / result[0]).reshape(1)
    return result


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        name,
        num_classes,
        ignore_index=None,
        class_weights=None,
        loss_weights=None,
        device="cpu",
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        normalized = (
            name.lower()
            .replace(" ", "")
            .replace("bce", "ce")
            .replace("crossentropy", "ce")
        )
        if normalized not in (
            "ce",
            "dice",
            "focal",
            "tversky",
            "ce+dice",
            "ce+focal",
            "dice+focal",
        ):
            raise ValueError(f"Неизвестный loss: {name}")
        self.names = normalized.split("+")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weights = (
            {self.names[0]: 1.0}
            if len(self.names) == 1
            else loss_weights or {"ce": 1.0, "dice": 1.0, "focal": 1.0}
        )
        if not any(self.weights.get(key, 1.0) > 0 for key in self.names):
            raise ValueError("Все выбранные веса loss равны нулю.")
        self.register_buffer(
            "class_weights", class_weight_tensor(class_weights, num_classes, device)
        )
        mode = "binary" if num_classes == 1 else "multiclass"
        self.overlap = nn.ModuleDict()
        for key in self.names:
            if key == "dice":
                self.overlap[key] = smp.losses.DiceLoss(
                    mode=mode, from_logits=True, ignore_index=ignore_index
                )
            elif key == "focal":
                self.overlap[key] = smp.losses.FocalLoss(
                    mode=mode, alpha=0.25, gamma=2, ignore_index=ignore_index
                )
            elif key == "tversky":
                self.overlap[key] = smp.losses.TverskyLoss(
                    mode=mode,
                    alpha=0.3,
                    beta=0.7,
                    from_logits=True,
                    ignore_index=ignore_index,
                )

    def forward(self, prediction, target):
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if self.ignore_index is None
            else target != self.ignore_index
        )
        if not valid.any():
            return prediction.sum() * 0
        if self.num_classes == 1:
            target = target[:, None].float()
            valid = valid[:, None]
        else:
            target = target.long()
        total = prediction.new_zeros(())
        for key in self.names:
            if self.weights.get(key, 1.0) == 0:
                continue
            if key == "ce":
                if self.num_classes == 1:
                    safe = torch.where(valid, target, 0.0)
                    value = F.binary_cross_entropy_with_logits(
                        prediction,
                        safe,
                        pos_weight=self.class_weights,
                        reduction="none",
                    )[valid].mean()
                else:
                    values = F.cross_entropy(
                        prediction,
                        target,
                        weight=self.class_weights,
                        ignore_index=self.ignore_index
                        if self.ignore_index is not None
                        else -100,
                        reduction="none",
                    )
                    mass = (valid.sum() if self.class_weights is None
                            else self.class_weights[target[valid]].sum())
                    # A batch containing only zero-weight classes is valid. Native
                    # weighted CE mean divides by zero and poisons mixed losses.
                    value = values[valid].sum() / torch.where(mass > 0, mass, 1)
            else:
                value = self.overlap[key](prediction, target)
            total = total + self.weights.get(key, 1.0) * value
        return total


def get_loss_function(
    loss_name,
    num_classes=1,
    ignore_index=None,
    class_weights=None,
    loss_weights=None,
    device="cpu",
):
    return SegmentationLoss(
        loss_name, num_classes, ignore_index, class_weights, loss_weights, device
    ).to(device)
