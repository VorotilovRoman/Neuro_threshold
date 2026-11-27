"""Validated prompt-model fine-tuning settings; no Torch or Qt imports."""

from pathlib import Path

from utils_new.configuration.training import _float, _int
from utils_new.import_export.sam_weights import MODELS


def sam_training_config(params):
    if not isinstance(params, dict):
        raise ValueError("Нужны параметры обучения SAM.")
    cfg = dict(model_key="sam2.1-tiny", model_path="", resume="", source="",
               val_source="", project_path="runs/sam", name="", epochs=20,
               batch_size=1, accum_steps=4, lr=0.0001, weight_decay=0.0001,
               device="auto", train_prompt_encoder=False, use_amp=True,
               prompt_type="mixed", box_jitter=0.05, positive_points=1,
               negative_points=1, max_objects=8, seed=42, grad_clip=1.0,
               background_id=0, ignore_index=255, validation_fraction=0.2,
               max_side=1024)
    unknown = set(params) - set(cfg)
    if unknown:
        raise ValueError(f"Неизвестные параметры SAM: {sorted(unknown)}")
    cfg.update(params)
    if cfg["model_key"] not in MODELS or str(cfg["model_key"]).startswith("sam3"):
        raise ValueError("Обучение поддерживает SAM и SAM2/2.1; обучение SAM3/3.1 не реализовано.")
    for key in ("epochs", "accum_steps", "positive_points", "max_objects", "max_side"):
        _int(cfg[key], key)
    if cfg["max_side"] < 32:
        raise ValueError("max_side должен быть ≥ 32.")
    for key in ("seed", "negative_points", "background_id"):
        _int(cfg[key], key, 0)
    if cfg["ignore_index"] is not None:
        _int(cfg["ignore_index"], "ignore_index", 0)
        if cfg["ignore_index"] == cfg["background_id"]:
            raise ValueError("Фон и ignore_index должны различаться.")
    if type(cfg["batch_size"]) is not int or cfg["batch_size"] not in (1, -1):
        raise ValueError("SAM обрабатывает один снимок за шаг. Используйте accum_steps вместо batch > 1.")
    cfg["batch_size"] = 1
    for key in ("lr", "grad_clip"):
        if _float(cfg[key], key) == 0:
            raise ValueError(f"{key} должен быть > 0.")
    _float(cfg["weight_decay"], "weight_decay")
    _float(cfg["box_jitter"], "box_jitter", maximum=0.25)
    if not 0 < _float(cfg["validation_fraction"], "validation_fraction", maximum=0.5):
        raise ValueError("validation_fraction должна быть > 0 и ≤ 0.5.")
    for key in ("train_prompt_encoder", "use_amp"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key}: требуется bool.")
    if cfg["prompt_type"] not in ("box", "points", "mixed"):
        raise ValueError("Подсказки: box, points или mixed.")
    if cfg["device"] not in ("auto", "cpu", "0", "cuda", "cuda:0"):
        raise ValueError("Устройство: auto, cpu или cuda:0.")
    for key in ("source", "val_source", "model_path", "resume"):
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
            if key == "model_path" and cfg["resume"]:
                continue  # A full resume contains its own model; initial weights may have moved.
            if not Path(cfg[key]).exists():
                raise ValueError(f"{key}: путь не найден: {cfg[key]}")
    if not cfg["source"] or not (cfg["model_path"] or cfg["resume"]):
        raise ValueError("Выберите датасет и локальные начальные веса либо checkpoint продолжения.")
    if cfg["name"] and (Path(cfg["name"]).name != cfg["name"] or cfg["name"] in (".", "..")):
        raise ValueError("Имя эксперимента должно быть именем подпапки.")
    cfg["project_path"] = str(Path(cfg["project_path"] or "runs/sam").expanduser().resolve())
    return cfg
