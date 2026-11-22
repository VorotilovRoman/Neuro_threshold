#!/usr/bin/env python
"""One command-line process for all AE/VAE architectures and training profiles."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import signal
from threading import Event
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Обучение AE/VAE и калибровка порогов снимка и пикселей."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON конфигурация обучения")
    source.add_argument(
        "--profile",
        choices=["conv_bn_baseline", "conv_bn", "conv_gn", "unet", "unet_v2_edge_gan"],
        help="Профиль из settings/autoencoders",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Папка нормальных снимков без дефектов"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Новая папка результатов"
    )
    parser.add_argument(
        "--resume", type=Path, help="last.pt с полной завершённой эпохой"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu или cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--edge-cache", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    from concurrent.futures import CancelledError
    from utils_new.configuration.autoencoder import TrainConfig
    from utils_new.import_export.json_objects import read_json_object

    stop = Event()

    def request_stop(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        from runtime_torch import prepare_windows_torch

        prepare_windows_torch()
        if args.resume and not args.config and not args.profile:
            from utils_new.import_export.autoencoder import read_checkpoint

            raw = read_checkpoint(args.resume).get("training_state", {}).get("config")
            if raw is None:
                raise ValueError("В --resume нет конфигурации обучения.")
        else:
            path = args.config or Path(__file__).resolve().parents[
                1
            ] / "settings/autoencoders" / (
                (args.profile or "unet_v2_edge_gan") + ".json"
            )
            raw = read_json_object(path)
        if args.epochs is not None:
            raw["epochs"] = args.epochs
        if args.batch_size is not None:
            raw["batch_size"] = args.batch_size
        config = TrainConfig.from_dict(raw)
        import torch
        from utils_new.autoencoders.training import run_training

        device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else args.device
        )
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Запрошена CUDA, но она недоступна.")
        run_training(
            config,
            args.data,
            args.output,
            device=device,
            resume=args.resume,
            edge_cache=args.edge_cache,
            cancel=stop,
            progress=lambda event: print(
                json.dumps(event, ensure_ascii=False), flush=True
            ),
            plots=not args.no_plots,
        )
        return 0
    except CancelledError:
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
