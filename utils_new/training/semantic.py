"""Shared semantic-segmentation training for U-Net, DeepLab and SegFormer."""

from concurrent.futures import CancelledError
from datetime import datetime
from pathlib import Path
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils_new.configuration.training import semantic_config
from utils_new.dataset.segmentation import ImageMaskDataset
from utils_new.import_export.mask_yaml import load_training_sources
from utils_new.import_export.training import (
    read_model_meta,
    read_semantic_checkpoint,
    read_weights,
    save_weights,
    save_model_meta,
    save_training_history,
    save_training_reports,
)
from utils_new.import_export.json_objects import write_json_atomic
from utils_new.segmentation.model_metrics import (
    confusion_matrix,
    metrics_from_confusion,
    prediction_labels,
)
from utils_new.training.losses import get_loss_function
from utils_new.training.optimization import (
    ModelEMA,
    PolyLRSchedulerIter,
    _forward_safe,
    predict_batch_multiscale,
    get_optimizer,
    get_scheduler,
    layerwise_optimizer,
    resolve_device,
    set_encoder_phase,
    set_bn_eval,
    probe_batch_size,
    singleton_batchnorm,
)
from utils_new.training.protocol import StopRequest, CANCELLED


def stop_check(stop):
    if stop.is_set():
        raise CancelledError("Обучение остановлено.")


def build_model(cfg):
    from utils_new.segmentation.training_contracts import build_semantic_model

    model = build_semantic_model(cfg)
    if cfg["model_path"]:
        model.load_state_dict(read_weights(cfg["model_path"]), strict=True)
    return model


def automatic_class_weights(dataset, classes, ignore_index):
    from utils_new.import_export.images import read_index_mask

    count = np.zeros(max(2, classes), dtype=np.int64)
    for path in dataset.mask_paths:
        values = read_index_mask(path, ignore_index=ignore_index)
        if ignore_index is not None:
            values = values[values != ignore_index]
        if classes == 1:
            values = (values > 0).astype(np.int64)
        count += np.bincount(values.ravel().astype(np.int64), minlength=len(count))
    if classes == 1:
        return [float(count[0] / count[1]) if count[0] and count[1] else 1.0]
    return (count.sum() / (classes * np.maximum(count, 1))).tolist()


def optimize_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    cfg,
    stop,
    *,
    ema=None,
    poly=None,
    scaler=None,
):
    model.train()
    if cfg["two_phase_training"] and (cfg["_frozen"] or cfg["freeze_bn_second_phase"]):
        set_bn_eval(model.encoder)
    use_fp16 = (
        cfg["use_amp"] and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    total, samples, pending, steps = 0.0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    def step():
        nonlocal pending, steps
        if not pending:
            return
        if use_fp16:
            scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(norm) and not use_fp16:
            raise ValueError("Градиенты содержат NaN/Inf.")
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if torch.isfinite(norm):
            steps += 1
            if ema is not None:
                ema.update(model)
            if poly is not None:
                poly.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0

    for index, (images, masks) in enumerate(loader):
        stop_check(stop)
        images, masks = images.to(device), masks.to(device)
        valid = (
            masks != cfg["ignore_index"]
            if cfg["ignore_index"] is not None
            else torch.ones_like(masks, dtype=torch.bool)
        )
        if valid.any():
            with singleton_batchnorm(model, len(images)):
                outputs = _forward_safe(model, images, cfg["use_amp"], device)
                loss = criterion(outputs, masks)
            if not torch.isfinite(loss):
                raise ValueError("Потери обучения содержат NaN/Inf.")
            weighted = loss * len(images)
            if use_fp16:
                scaler.scale(weighted).backward()
            else:
                weighted.backward()
            pending += len(images)
            samples += len(images)
            total += float(loss.detach()) * len(images)
        if (index + 1) % cfg["accum_steps"] == 0:
            step()
    step()
    if not samples or not steps:
        raise ValueError("Нет корректного шага оптимизации: проверьте маски/AMP.")
    return total / samples


class EvaluationSample:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        self.scores = None
        self.labels = np.empty(0, np.int64)
        self.priorities = np.empty(0)
        self.examples = []

    def add(self, outputs, masks, images):
        channels = outputs.shape[1]
        scores = (
            (outputs.sigmoid() if channels == 1 else outputs.softmax(1))
            .permute(0, 2, 3, 1)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, channels)
        )
        labels = masks.detach().cpu().numpy().reshape(-1)
        valid = (
            labels != self.cfg["ignore_index"]
            if self.cfg["ignore_index"] is not None
            else np.ones(labels.shape, bool)
        )
        scores, labels = scores[valid], labels[valid]
        # Bound the total probability array even with hundreds of classes.
        limit = max(1, min(self.cfg["pr_pixels"], 2000000 // channels))
        keys = self.rng.random(len(labels))
        if len(labels) > limit:
            chosen = np.argpartition(keys, limit - 1)[:limit]
            scores, labels, keys = scores[chosen], labels[chosen], keys[chosen]
        self.scores = (
            scores if self.scores is None else np.concatenate((self.scores, scores))
        )
        self.labels = np.concatenate((self.labels, labels))
        self.priorities = np.concatenate((self.priorities, keys))
        if len(self.labels) > limit:
            chosen = np.argpartition(self.priorities, limit - 1)[:limit]
            self.scores, self.labels, self.priorities = (
                self.scores[chosen],
                self.labels[chosen],
                self.priorities[chosen],
            )
        predicted = (
            prediction_labels(outputs, self.cfg["num_classes"], self.cfg["threshold"])
            .cpu()
            .numpy()
        )
        for image, target, prediction in zip(
            images.cpu(), masks.cpu().numpy(), predicted
        ):
            if len(self.examples) >= 8:
                break
            if self.cfg["normalize"]:
                image = (
                    image * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                    + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                )
            self.examples.append(
                (image.permute(1, 2, 0).numpy().clip(0, 1), target, prediction)
            )


@torch.inference_mode()
def evaluate(model, loader, criterion, device, cfg, stop, *, retain_samples=False):
    model.eval()
    total, samples = 0.0, 0
    matrix = torch.zeros(
        max(2, cfg["num_classes"]), max(2, cfg["num_classes"]), dtype=torch.long
    )
    retained = EvaluationSample(cfg) if retain_samples else None
    for images, masks in loader:
        stop_check(stop)
        images, masks = images.to(device), masks.to(device)
        if cfg["ignore_index"] is not None and not (masks != cfg["ignore_index"]).any():
            continue
        outputs = (
            predict_batch_multiscale(
                model, images, device, cfg["infer_scales"], cfg["use_amp"]
            )
            if cfg["multi_scale_infer"]
            else _forward_safe(model, images, cfg["use_amp"], device)
        )
        loss = criterion(outputs, masks)
        if not torch.isfinite(loss):
            raise ValueError("Валидация содержит NaN/Inf.")
        total += float(loss) * len(images)
        samples += len(images)
        matrix += confusion_matrix(
            outputs,
            masks,
            cfg["num_classes"],
            threshold=cfg["threshold"],
            ignore_index=cfg["ignore_index"],
        ).cpu()
        if retained is not None:
            retained.add(outputs, masks, images)
    if not samples or not matrix.sum():
        raise ValueError("Нет размеченных пикселей для оценки.")
    return (
        total / samples,
        metrics_from_confusion(matrix, cfg["num_classes"]),
        matrix,
        retained,
    )


def run_training(params, kind, *, stop=None, log=print):
    cfg = semantic_config(params, kind)
    stop = stop or StopRequest()
    stop_check(stop)
    model_meta = read_semantic_checkpoint(cfg['model_path'], require_metadata=False)[1] if cfg['model_path'] else {}
    for key in (
        "architecture",
        "encoder",
        "backbone",
        "decoder_channels",
        "decoder_segmentation_channels",
        "dropout",
        "input_size",
        "num_classes",
        "output_stride",
        "atrous_rates",
        "normalize",
        "resize_mode",
    ):
        if key in model_meta:
            cfg[key] = model_meta[key]
    cfg = semantic_config(cfg, kind)
    source = cfg["unet_yaml"] or cfg["deeplab_yaml"] or cfg["dataset_path"]
    if not source:
        raise ValueError("Не указан источник датасета.")
    data = load_training_sources(source, cfg["num_classes"])
    if (
        cfg["model_path"]
        and model_meta.get("num_classes", data["num_classes"]) != data["num_classes"]
    ):
        raise ValueError("Число классов модели не соответствует датасету.")
    cfg["num_classes"] = data["num_classes"]
    device = resolve_device(cfg["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train = ImageMaskDataset(
        data["splits"]["train"],
        cfg["input_size"],
        cfg["num_classes"],
        cfg["ignore_index"],
        cfg["augmentations"],
        normalize=cfg["normalize"],
        resize_mode=cfg["resize_mode"],
    )
    datasets = {
        key: ImageMaskDataset(
            pairs,
            cfg["input_size"],
            cfg["num_classes"],
            cfg["ignore_index"],
            normalize=cfg["normalize"],
            resize_mode=cfg["resize_mode"],
            is_train=False,
        )
        if pairs
        else None
        for key, pairs in data["splits"].items()
        if key != "train"
    }
    if isinstance(cfg["class_weights"], str) and cfg["class_weights"] == "auto":
        cfg["class_weights"] = automatic_class_weights(
            train, cfg["num_classes"], cfg["ignore_index"]
        )
    criterion = get_loss_function(
        cfg["loss"],
        cfg["num_classes"],
        cfg["ignore_index"],
        cfg["class_weights"],
        cfg["loss_weights"],
        device,
    )
    model = build_model(cfg).to(device)
    batch = (
        probe_batch_size(model, criterion, train, device, log=log)
        if cfg["auto_batch"] or cfg["batch_size"] == -1
        else cfg["batch_size"]
    )
    cfg["batch_size"] = batch
    from utils_new.training.optimization import seed_worker

    generator = torch.Generator().manual_seed(cfg["seed"])
    loader_args = dict(
        batch_size=batch,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    loader = DataLoader(train, shuffle=True, generator=generator, **loader_args)
    eval_loaders = {
        key: DataLoader(ds, shuffle=False, **loader_args) if ds else None
        for key, ds in datasets.items()
    }
    root = Path(cfg["project_path"] or ("runs/" + kind))
    if cfg["name"]:
        root = root / cfg["name"]
    base = root
    serial = 2
    while root.exists() and any(root.iterdir()):
        root = base.with_name(base.name + f"_{serial}")
        serial += 1
    root = root.resolve()
    weights = root / "weights"
    weights.mkdir(parents=True)
    from utils_new.import_export.dataset_sources import persist_source
    persistent_source = persist_source(source, root / 'dataset_source')
    for key in ('dataset_path', 'unet_yaml', 'deeplab_yaml'):
        if cfg.get(key) == str(source):
            cfg[key] = persistent_source
    write_json_atomic(root / "args.json", cfg)
    write_json_atomic(root / "split.json", data)
    log(f"RESULT_DIR:{root}")
    log(
        f"[DATA] train={len(train)}, val={len(datasets['val']) if datasets['val'] else 0}, test={len(datasets['test']) if datasets['test'] else 0}; batch={batch}, device={device}"
    )
    if batch == 1 or len(train) % batch == 1:
        log(
            "[BN] Для одиночного снимка используются сохранённые running-статистики BatchNorm."
        )
    frozen = cfg["two_phase_training"] and cfg["freeze_epochs"] > 0
    set_encoder_phase(
        model,
        frozen=frozen,
        last_n=cfg["unfreeze_last_n_stages"]
        if cfg["two_phase_training"] and not frozen
        else None,
    )
    cfg["_frozen"] = frozen

    def optimization(phase_epochs):
        lr = cfg["lr"] * (0.1 if cfg["_frozen"] else 1.0)
        opt = (
            layerwise_optimizer(
                model,
                cfg["optimizer"],
                lr,
                cfg["momentum"],
                cfg["weight_decay"],
                cfg["layerwise_decay"],
            )
            if cfg["use_layerwise_lr"]
            else get_optimizer(
                model, cfg["optimizer"], lr, cfg["momentum"], cfg["weight_decay"]
            )
        )
        scheduler = (
            PolyLRSchedulerIter(
                opt,
                math.ceil(len(loader) / cfg["accum_steps"]) * phase_epochs,
                cfg["poly_power"],
            )
            if cfg["scheduler"].lower() == "poly"
            else get_scheduler(opt, cfg["scheduler"], phase_epochs)
        )
        return opt, scheduler

    optimizer, scheduler = optimization(
        cfg["freeze_epochs"] if frozen else cfg["epochs"]
    )
    ema = ModelEMA(model, cfg["ema_decay"], device) if cfg["use_ema"] else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg["use_amp"]
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported(),
    )
    history = []
    best_loss = float("inf")
    best_iou = -float("inf")
    stale = 0
    epoch = 0
    metadata = dict(
        cfg,
        dataset_path=data["root"],
        device_used=str(device),
        output_activation="sigmoid" if cfg["num_classes"] == 1 else "softmax",
        preprocessing_version="semantic-opencv-v1",
        class_names=data.get('class_names', []),
        annotation_class_offset=data.get('mask_class_offset', 1),
        image_resize_interpolation="linear",
        output_kind="logits",
    )
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            stop_check(stop)
            if frozen and epoch == cfg["freeze_epochs"] + 1:
                frozen = False
                cfg["_frozen"] = False
                set_encoder_phase(
                    model, frozen=False, last_n=cfg["unfreeze_last_n_stages"]
                )
                optimizer, scheduler = optimization(
                    cfg["epochs"] - cfg["freeze_epochs"]
                )
                log(
                    "[PHASE] Backbone разморожен; оптимизатор и scheduler второй фазы созданы."
                )
            train_loss = optimize_epoch(
                model,
                loader,
                criterion,
                optimizer,
                device,
                cfg,
                stop,
                ema=ema,
                poly=scheduler if isinstance(scheduler, PolyLRSchedulerIter) else None,
                scaler=scaler,
            )
            eval_model = ema.ema if ema else model
            val_loss, metrics, _, _ = (
                evaluate(eval_model, eval_loaders["val"], criterion, device, cfg, stop)
                if eval_loaders["val"]
                else (None, None, None, None)
            )
            selection_loss = val_loss if val_loss is not None else train_loss
            row = dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                iou=metrics["IoU"] if metrics else None,
                dice=metrics["Dice"] if metrics else None,
                f1=metrics["F1"] if metrics else None,
                precision=metrics["Precision"] if metrics else None,
                recall=metrics["Recall"] if metrics else None,
                accuracy=metrics["Accuracy"] if metrics else None,
            )
            history.append(row)
            checkpoint_meta = dict(
                metadata,
                epoch=epoch,
                epochs_completed=epoch,
                status="completed_epoch",
                epoch_complete=True,
                validation_metrics=metrics,
                best_val_loss=val_loss,
                best_iou=metrics["IoU"] if metrics else None,
                weights_source="ema" if ema else "model",
                date=datetime.now().isoformat(),
            )
            if selection_loss < best_loss:
                best_loss = selection_loss
                stale = 0
                save_weights(
                    weights / "best_loss.pth",
                    eval_model,
                    dict(
                        checkpoint_meta,
                        selection_metric="val_loss"
                        if val_loss is not None
                        else "train_loss",
                    ),
                )
            else:
                stale += 1
            if metrics and metrics["IoU"] > best_iou:
                best_iou = metrics["IoU"]
                save_weights(
                    weights / "best_iou.pth",
                    eval_model,
                    dict(checkpoint_meta, selection_metric="IoU"),
                )
            save_weights(weights / "last.pth", eval_model, checkpoint_meta)
            save_training_history(root, history)
            if scheduler is not None and not isinstance(scheduler, PolyLRSchedulerIter):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(selection_loss)
                else:
                    scheduler.step()
            log(
                f"PROGRESS:{min(100, int(epoch / cfg['epochs'] * 100))}:{epoch}:{cfg['epochs']}"
            )
            log(f"[EPOCH] {row}")
            if stale >= cfg["patience"]:
                break
        stop_check(stop)
        selected = weights / (
            "best_iou.pth" if (weights / "best_iou.pth").exists() else "best_loss.pth"
        )
        model.load_state_dict(read_weights(selected), strict=True)
        model.eval()
        selected_meta = read_model_meta(selected)
        retained = None
        matrix = None
        final_metrics = None
        if eval_loaders["val"]:
            _, final_metrics, matrix, retained = evaluate(
                model,
                eval_loaders["val"],
                criterion,
                device,
                cfg,
                stop,
                retain_samples=cfg["plots"],
            )
        log("METRICS: " + __import__("json").dumps(final_metrics))
        if eval_loaders["test"]:
            _, test_metrics, _, _ = evaluate(
                model, eval_loaders["test"], criterion, device, cfg, stop
            )
            write_json_atomic(
                root / "test_metrics.json", dict(test_metrics, checkpoint=str(selected))
            )
            selected_meta["test_metrics"] = test_metrics
            save_model_meta(selected_meta, selected)
            log("TEST_METRICS: " + __import__("json").dumps(test_metrics))
        if cfg["plots"]:
            try:
                sample = (
                    (retained.scores, retained.labels) if retained is not None else None
                )
                train_samples = []
                # Use a deterministic dataset view; rendering must not consume training augmentations.
                visual = ImageMaskDataset(
                    train.pairs[:4],
                    cfg["input_size"],
                    cfg["num_classes"],
                    cfg["ignore_index"],
                    normalize=cfg["normalize"],
                    resize_mode=cfg["resize_mode"],
                    is_train=False,
                )
                display = EvaluationSample(cfg)
                with torch.inference_mode():
                    for image, mask in visual:
                        output = _forward_safe(
                            model, image[None].to(device), False, device
                        )
                        display.add(
                            output, mask[None].to(device), image[None].to(device)
                        )
                train_samples = display.examples
                save_training_reports(
                    root,
                    history,
                    matrix.numpy() if matrix is not None else None,
                    sample,
                    retained.examples if retained else (),
                    train_samples,
                )
            except Exception as error:
                log(f"[REPORT][WARNING] Веса сохранены; отчёт не построен: {error}")
        log("[MODEL] Для экспорта весов используйте вкладку «Экспорт модели».")
        log("SUCCESS")
        return 0
    except CancelledError:
        interrupted = dict(
            metadata,
            epoch=epoch,
            epochs_completed=len(history),
            status="stopped",
            epoch_complete=False,
            validation_metrics=None,
            best_val_loss=None,
            best_iou=None,
        )
        save_weights(
            weights / f"interrupted_epoch_{epoch}.pth",
            ema.ema if ema else model,
            interrupted,
        )
        save_training_history(root, history)
        log("CANCELLED: лучшие веса и последние полные эпохи сохранены.")
        return CANCELLED
