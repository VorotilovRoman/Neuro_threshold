#!/usr/bin/env python
"""YOLO worker: keep Ultralytics AutoBatch intact and report each pipeline stage."""

if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import shutil
import signal
import traceback
from datetime import datetime
from pathlib import Path

from utils_new.configuration.training import build_train_args
from utils_new.import_export.model_export import export_onnx
from utils_new.import_export.training import save_model_meta

stop_requested = False
trainer_ref = None
CANCELLED = 130


def log(message):
    print(message, flush=True)


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    log(f"Получен сигнал {signum}. Запрошена остановка обучения.")
    if trainer_ref is not None:
        # Ultralytics 8.4.x checks `stop`, not `stop_training`.
        trainer_ref.stop = True


def resolved_settings(trainer, args):
    effective = getattr(trainer, "args", None)
    batch = getattr(trainer, "batch_size", None)
    if type(batch) is not int or batch < 1:
        batch = getattr(effective, "batch", args["batch"])
    if type(batch) is not int or batch < 1:
        # Only a post-training validation fallback; training AutoBatch remains untouched.
        batch = 1
    return {
        "batch": batch,
        "imgsz": getattr(effective, "imgsz", args["imgsz"]),
        "amp": bool(getattr(trainer, "amp", args.get("amp", False))),
        "device": str(getattr(trainer, "device", args.get("device", "auto"))),
    }


def main(argv=None):
    global stop_requested, trainer_ref
    stop_requested, trainer_ref = False, None
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--params", type=str)
    source.add_argument("--params_file", type=str)
    parser.add_argument("--stop_file")
    args = parser.parse_args(argv)
    if args.stop_file and Path(args.stop_file).exists():
        return CANCELLED
    old_signals = {}
    phase = "проверка параметров"
    train_args = {}
    started = datetime.now().isoformat()
    try:
        log("WORKER STARTED")
        params = (
            json.loads(Path(args.params_file).read_text(encoding="utf-8-sig"))
            if args.params_file
            else json.loads(args.params)
        )
        train_args = build_train_args(params)
        if train_args["batch"] == -1 or 0 < train_args["batch"] < 1:
            log(
                "[BATCH] AutoBatch передан Ultralytics без изменения; ошибка отдельной пробы не обязательно означает сбой обучения."
            )
        phase = "импорт Ultralytics"
        from runtime_policy import configure_ultralytics

        configure_ultralytics()
        import ultralytics
        from ultralytics import YOLO

        log(f"Ultralytics YOLO version: {ultralytics.__version__}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, signal_handler)
        phase = "загрузка модели"
        model = YOLO(params["model"])
        task = getattr(model, "task", "unknown")
        log(f"Модель загружена: {params['model']}; task={task}")

        def capture_trainer(trainer):
            global trainer_ref, stop_requested
            if args.stop_file and Path(args.stop_file).exists():
                stop_requested = True
            trainer_ref = trainer
            if stop_requested:
                trainer.stop = True

        def on_train_start(trainer):
            nonlocal phase
            capture_trainer(trainer)
            phase = "обучение"
            log(
                "[TRAIN] Фактические параметры: "
                + json.dumps(resolved_settings(trainer, train_args), ensure_ascii=False)
            )

        def on_epoch_end(trainer):
            capture_trainer(trainer)
            epoch = trainer.epoch + 1
            total = trainer.epochs
            log(f"PROGRESS:{min(100, int(epoch / total * 100))}:{epoch}:{total}")

        model.add_callback("on_pretrain_routine_start", capture_trainer)
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_train_batch_end", capture_trainer)
        phase = "подготовка обучения / AutoBatch"
        log("=== Начало обучения ===")
        log(json.dumps(train_args, indent=2, ensure_ascii=False))
        results = model.train(**train_args)
        trainer = getattr(model, "trainer", None)
        effective = resolved_settings(trainer, train_args)
        run_dir = getattr(trainer, "save_dir", None) or getattr(
            results, "save_dir", None
        )
        metadata = {
            "model_type": {
                "detect": "YOLO-detect",
                "segment": "YOLO-seg",
                "obb": "YOLO-obb",
                "pose": "YOLO-pose",
                "classify": "YOLO-cls",
            }.get(task, "YOLO"),
            "task": task,
            "model_source": params["model"],
            "data_yaml": train_args["data"],
            "epochs": train_args["epochs"],
            "epochs_completed": getattr(trainer, "epoch", -1) + 1,
            "batch_requested": train_args["batch"],
            "batch": effective["batch"],
            "imgsz_requested": train_args["imgsz"],
            "imgsz": effective["imgsz"],
            "device": effective["device"],
            "amp": effective["amp"],
            "requested_train_args": train_args,
            "disable_aug": params.get("disable_aug", False),
            "date_start": started,
            "date_end": datetime.now().isoformat(),
            "status": "stopped" if stop_requested else "completed",
        }
        # Keep requested optimizer/augmentation fields available in the sidecar;
        # native optimizer='auto' may resolve these differently during training.
        metadata.update(
            {
                key: train_args.get(key)
                for key in (
                    "optimizer",
                    "lr0",
                    "lrf",
                    "close_mosaic",
                    "mosaic",
                    "project",
                    "name",
                )
            }
        )
        checkpoints = []
        if run_dir:
            run_dir = Path(run_dir).resolve()
            log(f"RESULT_DIR:{run_dir}")
            checkpoints = [
                path
                for path in (
                    run_dir / "weights" / "best.pt",
                    run_dir / "weights" / "last.pt",
                )
                if path.is_file()
            ]
            if stop_requested and checkpoints and checkpoints[0].name == "best.pt":
                backup = checkpoints[0].with_name("best_interrupted.pt")
                shutil.copy2(checkpoints[0], backup)
                log(f"Лучшая модель скопирована в {backup}")
            for path in checkpoints:
                save_model_meta(dict(metadata, checkpoint=str(path)), path)
        if stop_requested:
            log(
                "CANCELLED: обучение остановлено; дополнительная валидация и экспорт пропущены."
            )
            return CANCELLED
        phase = "валидация"
        validation_args = {
            "data": train_args["data"],
            "imgsz": effective["imgsz"],
            "batch": effective["batch"],
            "device": effective["device"],
            "half": effective["amp"],
            "plots": train_args.get("plots", True),
        }
        if run_dir:
            validation_args.update(
                project=str(run_dir), name="validation", exist_ok=True
            )
        log("[VAL] " + json.dumps(validation_args, ensure_ascii=False))
        val_results = model.val(**validation_args)
        from utils_new.segmentation.yolo_metrics import validation_metrics

        metrics = validation_metrics(val_results)
        log(f"METRICS: {json.dumps(metrics)}")
        # Ultralytics reloads best.pt after train, or last.pt if no best exists.
        # Never assign the best checkpoint's metrics to last.pt as its own score.
        validated = checkpoints[0] if checkpoints else None
        if validated:
            metadata.update(
                validation_weights=str(validated),
                validation_metrics=metrics,
                date_end=datetime.now().isoformat(),
            )
            metadata.update(
                {
                    key: metrics.get(key)
                    for key in ("mAP50", "mAP50-95", "precision", "recall")
                }
            )
            save_model_meta(dict(metadata, checkpoint=str(validated)), validated)
        if stop_requested:
            log("CANCELLED: экспорт пропущен.")
            return CANCELLED
        if params.get("export_onnx", True):
            phase = "экспорт ONNX"
            export_onnx(model, metadata, effective["imgsz"], run_dir, log=log)
        log("SUCCESS")
        return 0
    except KeyboardInterrupt:
        log("CANCELLED: обучение прервано пользователем.")
        return CANCELLED
    except Exception as error:  # noqa: BLE001 - process boundary preserves original failures
        log(f"[ERROR] Этап: {phase}. {type(error).__name__}: {error}")
        if "out of memory" in str(error).lower() and (
            "cuda" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"
        ):
            log(
                "[CUDA_OOM] Не хватило памяти GPU. AutoBatch мог завершиться ошибкой во время профилирования/очистки памяти; это не ошибка импорта. Параметры не менялись и автоматического повторного запуска нет. Для нового запуска уменьшите imgsz/модель или укажите меньший batch; batch=1 также может не поместиться."
            )
        # Do not call CUDA cleanup after a CUDA failure: cleanup itself can raise.
        log(traceback.format_exc())
        return 1
    finally:
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)
        trainer_ref = None


if __name__ == "__main__":
    from utils_new.training.protocol import configure_worker_streams

    configure_worker_streams()
    raise SystemExit(main())
