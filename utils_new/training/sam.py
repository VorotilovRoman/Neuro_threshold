"""Local SAM/SAM2 decoder fine-tuning in a separate, cancellable process."""

import argparse
from concurrent.futures import CancelledError
from contextlib import nullcontext
import json
from pathlib import Path
import traceback

from runtime_torch import require_torch
from utils_new.configuration.sam_training import sam_training_config
from utils_new.training.protocol import CANCELLED, StopRequest, configure_worker_streams


class PromptTrainer:
    """Encode once/image without gradients, decode with autograd; never call predict()."""

    def __init__(self, model, model_key, cfg):
        import importlib

        self.model, self.cfg = model, cfg
        self.sam2 = not model_key.startswith("sam-vit-")
        model.requires_grad_(False)
        self.decoder = model.sam_mask_decoder if self.sam2 else model.mask_decoder
        self.encoder = model.sam_prompt_encoder if self.sam2 else model.prompt_encoder
        self.decoder.requires_grad_(True)
        # SAM2 forward_image preprojects these under no_grad: keep them explicitly frozen.
        for name in ("conv_s0", "conv_s1"):
            if hasattr(self.decoder, name):
                getattr(self.decoder, name).requires_grad_(False)
        self.encoder.requires_grad_(cfg["train_prompt_encoder"])
        if self.sam2:
            predictor = importlib.import_module("sam2.sam2_image_predictor").SAM2ImagePredictor
        else:
            predictor = importlib.import_module("segment_anything").SamPredictor
        self.predictor = predictor(model)

    def mode(self, training):
        self.model.eval()  # Frozen backbone and memory stay deterministic.
        self.decoder.train(training)
        self.encoder.train(training and self.cfg["train_prompt_encoder"])

    def encode(self, image):
        torch = require_torch()
        with torch.no_grad():
            self.predictor.set_image(image)
        self.shape = image.shape[:2]

    def logits(self, prompt):
        torch = require_torch()
        box, points, labels = prompt
        device = next(self.model.parameters()).device
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
        labels = torch.as_tensor(labels, device=device, dtype=torch.int64)[None]
        box = torch.as_tensor(box, device=device, dtype=torch.float32)[None] if box is not None else None
        if self.sam2:
            transform = self.predictor._transforms
            coords = transform.transform_coords(points, normalize=True, orig_hw=self.shape)[None]
            coords = coords if points.numel() else None
            concat = (coords, labels) if coords is not None else None
            if box is not None:
                corners = transform.transform_boxes(box, normalize=True, orig_hw=self.shape)
                tags = torch.tensor([[2, 3]], dtype=torch.int64, device=device)
                concat = (torch.cat([corners, coords], 1), torch.cat([tags, labels], 1)) if concat else (corners, tags)
            sparse, dense = self.encoder(points=concat, boxes=None, masks=None)
            features = self.predictor._features
            low, scores, _, _ = self.decoder(
                image_embeddings=features["image_embed"], image_pe=self.encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True, repeat_image=False, high_res_features=features["high_res_feats"])
            logits = torch.nn.functional.interpolate(low.float(), self.shape, mode="bilinear", align_corners=False)
        else:
            transform = self.predictor.transform
            coords = transform.apply_coords_torch(points, self.shape)[None] if points.numel() else None
            boxes = transform.apply_boxes_torch(box, self.shape) if box is not None else None
            sparse, dense = self.encoder(points=(coords, labels) if coords is not None else None,
                                         boxes=boxes, masks=None)
            low, scores = self.decoder(image_embeddings=self.predictor.features,
                image_pe=self.encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense, multimask_output=True)
            logits = self.model.postprocess_masks(low.float(), self.predictor.input_size, self.shape)
        return logits[0].float(), scores[0].float()


def mask_loss(logits, scores, target, valid):
    """Best-of-three mask supervision plus calibration for all three IoU scores."""
    torch = require_torch()
    functional = torch.nn.functional
    truth = target[None].expand_as(logits)
    weights = valid[None].expand_as(logits)
    denominator = weights.sum((1, 2)).clamp_min(1)
    ce = (functional.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weights).sum((1, 2)) / denominator
    probability = logits.sigmoid() * weights
    truth = truth * weights
    dice = 1 - (2 * (probability * truth).sum((1, 2)) + 1) / (probability.sum((1, 2)) + truth.sum((1, 2)) + 1)
    predicted = (logits.detach() > 0) & weights.bool()
    actual = truth.bool()
    intersection = (predicted & actual).sum((1, 2)).float()
    union = (predicted | actual).sum((1, 2)).float()
    iou = intersection / union.clamp_min(1)
    loss = (ce + dice).min() + 0.1 * functional.mse_loss(scores, iou)
    chosen = scores.detach().argmax()
    selected_dice = (2 * intersection[chosen] + 1e-6) / (predicted[chosen].sum() + actual[chosen].sum() + 1e-6)
    return loss, float(iou[chosen]), float(selected_dice)


def prepared_sample(sample, cfg):
    import cv2
    import numpy as np

    image = sample.rgb()
    masks, valid = sample.masks(cfg)
    h, w = image.shape[:2]
    scale = min(1, cfg["max_side"] / max(h, w))
    if scale < 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        masks = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
        valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    if any(not m.any() for m in masks):
        raise ValueError(f"Объект исчез после resize; увеличьте max_side или используйте кропы: {sample.image}")
    return image, masks, valid


def run_training(params, *, stop=None, log=print):
    import numpy as np
    torch = require_torch()
    from utils_new.dataset.sam import check_stop, dataset_signature, sample_prompt, training_samples
    from utils_new.import_export.dataset_sources import log_manifest, metadata, persist_source
    from utils_new.import_export.atomic import atomic_path
    from utils_new.import_export.json_objects import write_json_atomic
    from utils_new.import_export.training import save_model_meta, save_training_history
    from utils_new.segmentation.backends.sam import load_prompt_model
    from utils_new.training.optimization import resolve_device

    cfg = sam_training_config(params)
    stop = stop or StopRequest()
    check_stop(stop)
    device = resolve_device(cfg["device"])
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    for key in ("source", "val_source"):
        if cfg[key]:
            log(f"[SAM] {key}: {cfg[key]}")
            log_manifest(cfg[key], log)
    train, val = training_samples(cfg, stop, log)
    signature = dataset_signature({"train": train, "val": val}, stop)
    check_stop(stop)
    resume = torch.load(cfg["resume"], map_location="cpu", weights_only=True) if cfg["resume"] else None
    if resume:
        if resume.get("format") != "neuro-sam-resume-v1" or not resume.get("complete_epoch"):
            raise ValueError("Для продолжения выберите last.resume.pt с последней полной эпохой.")
        previous = resume["config"]
        for key in ("model_key", "train_prompt_encoder", "prompt_type", "accum_steps", "lr",
                    "weight_decay", "positive_points", "negative_points", "max_objects", "seed",
                    "box_jitter", "max_side", "background_id", "ignore_index", "validation_fraction"):
            if cfg[key] != previous[key]:
                raise ValueError(f"Resume: изменён параметр {key}; для нового режима загрузите last.pt как начальные веса.")
        if resume["images"] != {"train": [s.image for s in train], "val": [s.image for s in val]}:
            raise ValueError("Resume: изменилось разделение train/val.")
        if resume["dataset_signature"] != signature:
            raise ValueError("Resume: изменились снимки или разметка. Начните новый запуск из last.pt.")
    model = load_prompt_model(cfg["model_key"], cfg["resume"] or cfg["model_path"], device, training=True)
    trainer = PromptTrainer(model, cfg["model_key"], cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = cfg["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, start, best, best_trainable = [], 0, -1.0, {}
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        history, start, best = resume["history"], resume["epoch"], resume["best_iou"]
        best_trainable = resume["best_trainable"]
        rng.bit_generator.state = resume["rng"]
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda" and resume.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    if start >= cfg["epochs"]:
        raise ValueError("epochs — общее число эпох; оно должно превышать число уже выполненных.")
    root = Path(cfg["project_path"]) / (cfg["name"] or "sam")
    base, index = root, 1
    while root.exists():
        root = base.with_name(f"{base.name}_{index}")
        index += 1
    root.mkdir(parents=True)
    log(f"RESULT_DIR:{root}")
    for key in ("source", "val_source"):
        if cfg[key] and metadata(cfg[key]):
            if Path(cfg[key]).suffix.lower() == ".json":
                from shutil import copyfile
                target = root / f"{key}.json"
                copyfile(cfg[key], target)
                cfg[key] = str(target)
            else:
                cfg[key] = persist_source(cfg[key], root / key)
    write_json_atomic(root / "params.json", cfg)
    write_json_atomic(root / "dataset.json", {name: [s.image for s in group] for name, group in (("train", train), ("val", val))})
    log(f"[SAM] {cfg['model_key']} / {device}; batch=1, накопление={cfg['accum_steps']}; "
        f"обучаемых параметров: {sum(p.numel() for p in parameters):,}. Кодировщик снимков заморожен.")
    log("[SAM] Начинаем пробный шаг; OOM не уменьшает batch ниже 1 — выберите tiny, меньший max_side или CPU.")
    if not amp and cfg["use_amp"]:
        log("[SAM] AMP отключён на CPU.")

    def save(name, epoch, *, complete, full=False, best_weights=False):
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if best_weights:
            state.update(best_trainable)
        metrics = (max(history, key=lambda row: row["val_iou"]) if best_weights
                   else history[-1] if history else {})
        meta = dict(kind="sam", model_key=cfg["model_key"], epoch=epoch,
                    complete_epoch=complete, max_side=cfg["max_side"],
                    prompt_type=cfg["prompt_type"], metrics=metrics,
                    dataset_signature=signature, config=cfg)
        payload = {"model": state, "model_key": cfg["model_key"]} if not cfg["model_key"].startswith("sam-vit-") else state
        if full:
            payload = dict(format="neuro-sam-resume-v1", model=state, model_key=cfg["model_key"],
                config=cfg, epoch=epoch, complete_epoch=complete, history=history,
                best_iou=best, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                best_trainable=best_trainable,
                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                dataset_signature=signature,
                images={"train": [s.image for s in train], "val": [s.image for s in val]})
        with atomic_path(root / name) as temporary:
            torch.save(payload, temporary)
        save_model_meta(meta, root / name)

    if resume:
        best_epoch = max(history, key=lambda row: row["val_iou"])["epoch"]
        save("best.pt", best_epoch, complete=True, best_weights=True)

    def phase(samples, training, epoch):
        trainer.mode(training)
        sums, count = np.zeros(3, np.float64), 0
        order = rng.permutation(len(samples)) if training else range(len(samples))
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        val_rng = np.random.default_rng(cfg["seed"])
        for position, index in enumerate(order):
            check_stop(stop)
            image, masks, valid = prepared_sample(samples[index], cfg)
            selection = (rng.choice(len(masks), min(cfg["max_objects"], len(masks)), replace=False)
                         if training else range(len(masks)))
            autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.float16) if amp else nullcontext()
            with autocast():
                trainer.encode(image)
            for obj in selection:
                check_stop(stop)
                target = torch.as_tensor(masks[obj].astype(np.float32), device=device)
                valid_tensor = torch.as_tensor(valid.astype(np.float32), device=device)
                prompt = sample_prompt(masks[obj], valid, cfg, rng if training else val_rng, training=training)
                with torch.set_grad_enabled(training), autocast():
                    logits, scores = trainer.logits(prompt)
                    loss, iou, dice = mask_loss(logits, scores, target, valid_tensor)
                if not torch.isfinite(loss):
                    raise ValueError(f"Нечисловой loss: {samples[index].image}")
                if training:
                    # Average objects within each image; accumulation counts images, not objects.
                    scaler.scale(loss / len(selection)).backward()
                sums += (float(loss.detach()), iou, dice)
                count += 1
            if training:
                pending += 1
                if pending == cfg["accum_steps"] or position + 1 == len(samples):
                    scaler.unscale_(optimizer)
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(pending)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=not amp)
                    if amp and not torch.isfinite(norm):
                        log("[SAM] Переполнение AMP: шаг пропущен, масштаб градиентов будет уменьшен.")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
            log(f"[SAM] Эпоха {epoch} {'train' if training else 'val'}: {position + 1}/{len(samples)}; loss={sums[0]/count:.4f}")
            fraction = (position + 1) / len(samples) / 2 + (0 if training else 0.5)
            percentage = int((epoch - 1 + fraction) * 100 / cfg["epochs"])
            log(f"PROGRESS:{percentage}:{epoch}:{cfg['epochs']}")
        return (sums / count).tolist()

    epoch = start
    try:
        for epoch in range(start + 1, cfg["epochs"] + 1):
            check_stop(stop)
            train_loss, train_iou, train_dice = phase(train, True, epoch)
            val_loss, val_iou, val_dice = phase(val, False, epoch)
            row = dict(epoch=epoch, train_loss=train_loss, train_iou=train_iou,
                       train_dice=train_dice, val_loss=val_loss, val_iou=val_iou, val_dice=val_dice)
            history.append(row)
            if val_iou > best:
                best = val_iou
                best_trainable = {key: parameter.detach().cpu().clone()
                                  for key, parameter in model.named_parameters() if parameter.requires_grad}
                save("best.pt", epoch, complete=True, best_weights=True)
            save("last.pt", epoch, complete=True)
            save("last.resume.pt", epoch, complete=True, full=True)
            save_training_history(root, history)
            log(f"[SAM] {row}")
            log(f"PROGRESS:{int(epoch * 100 / cfg['epochs'])}:{epoch}:{cfg['epochs']}")
    except CancelledError:
        save("interrupted.pt", epoch, complete=False)
        save_training_history(root, history)
        log("[SAM] Остановлено. interrupted.pt — частичные веса; last.resume.pt — последняя полная эпоха.")
        return CANCELLED
    except torch.OutOfMemoryError as error:
        raise RuntimeError("Недостаточно памяти для SAM. Batch уже равен 1: выберите SAM2.1 tiny, "
                           "отключите обучение prompt encoder/AMP при необходимости или используйте CPU. "
                           "Уменьшение max_side снижает память масок, но не внутренний размер SAM.") from error
    return 0


def main(argv=None):
    configure_worker_streams()
    parser = argparse.ArgumentParser(description="Локальное дообучение SAM/SAM2 по маскам объектов.")
    parser.add_argument("--params_file", required=True)
    parser.add_argument("--stop_file")
    args = parser.parse_args(argv)
    try:
        cfg = sam_training_config(json.loads(Path(args.params_file).read_text(encoding="utf-8-sig")))
        stop = StopRequest(args.stop_file)
        if stop.is_set():
            return CANCELLED
        with stop.signals():
            return run_training(cfg, stop=stop)
    except CancelledError:
        return CANCELLED
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
