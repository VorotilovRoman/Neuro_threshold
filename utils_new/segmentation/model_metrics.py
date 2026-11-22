"""Semantic segmentation metrics from class confusion, not detection AP."""


def prediction_labels(logits, num_classes, threshold=0.5):
    import torch

    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("Число классов должно быть положительным целым.")
    if not 0 <= threshold <= 1:
        raise ValueError("Порог вероятности должен быть 0..1.")
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError("Ожидается NCHW с числом каналов модели.")
    if not torch.isfinite(logits).all():
        raise ValueError("Предсказание содержит NaN или бесконечность.")
    if num_classes == 1:
        return (torch.sigmoid(logits[:, 0]) > threshold).long()
    elif num_classes == 2:
        return (torch.softmax(logits, dim=1)[:, 1] > threshold).long()
    else:
        return logits.argmax(dim=1)


def segmentation_metrics(
    logits, target, num_classes, *, threshold=0.5, ignore_index=None
):
    matrix = confusion_matrix(
        logits, target, num_classes, threshold=threshold, ignore_index=ignore_index
    ).double()
    return metrics_from_confusion(matrix, num_classes)


def metrics_from_confusion(matrix, num_classes):
    import torch

    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("Число классов должно быть положительным целым.")
    count = max(2, num_classes)
    if (matrix.shape != (count, count) or not torch.isfinite(matrix).all()
            or (matrix < 0).any()):
        raise ValueError("Ожидается конечная неотрицательная квадратная матрица ошибок по классам.")
    keys = ("IoU", "Dice", "Precision", "Recall", "F1", "Accuracy")
    if not matrix.sum():
        return dict.fromkeys(keys, float("nan"))
    tp = matrix.diag()
    actual, detected = matrix.sum(dim=1), matrix.sum(dim=0)
    union = actual + detected - tp
    selected = (
        torch.arange(count, device=matrix.device) == 1
        if num_classes <= 2
        else union > 0
    )
    if num_classes <= 2 and union[1] == 0:
        # Empty foreground correctly predicted: perfect overlap; no object FP/FN.
        return dict.fromkeys(keys, 1.0)
    tp, actual, detected, union = (
        tp[selected],
        actual[selected],
        detected[selected],
        union[selected],
    )
    iou = tp / union.clamp_min(1)
    dice = 2 * tp / (actual + detected).clamp_min(1)
    precision = tp / detected.clamp_min(1)
    recall = tp / actual.clamp_min(1)
    return dict(
        zip(
            keys,
            (
                iou.mean().item(),
                dice.mean().item(),
                precision.mean().item(),
                recall.mean().item(),
                dice.mean().item(),
                matrix.diag().sum().item() / matrix.sum().item(),
            ),
        )
    )


def confusion_matrix(logits, target, num_classes, *, threshold=0.5, ignore_index=None):
    import torch

    predicted = prediction_labels(logits, num_classes, threshold)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != predicted.shape:
        raise ValueError("Размеры предсказания и эталонной маски не совпадают.")
    valid = (
        torch.ones_like(target, dtype=torch.bool)
        if ignore_index is None
        else target != ignore_index
    )
    count = max(2, num_classes)
    selected = target[valid]
    if (
        not torch.isfinite(selected).all()
        or (selected != selected.long()).any()
        or (selected < 0).any()
        or (selected >= count).any()
    ):
        raise ValueError("Индекс класса эталона вне диапазона модели или нецелый.")
    return torch.bincount(
        selected.long() * count + predicted[valid], minlength=count * count
    ).reshape(count, count)
