#!/usr/bin/env python
"""One command-line process for all AE/VAE architectures and training profiles."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from utils_new.training.protocol import StopRequest
import traceback


def main(argv=None):
    from utils_new.configuration.autoencoder import AE_PROFILES
    parser = argparse.ArgumentParser(
        description="Обучение AE/VAE и калибровка порогов снимка и пикселей."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--params_file', type=Path, help='Параметры запуска из вкладки AE/VAE')
    source.add_argument("--config", type=Path, help="JSON конфигурация обучения")
    source.add_argument(
        "--profile",
        choices=AE_PROFILES,
        help="Профиль из settings/autoencoders",
    )
    parser.add_argument(
        "--data", type=Path, help="Папка нормальных снимков или YOLO/MASKS YAML (метки не нужны)"
    )
    parser.add_argument(
        "--output", type=Path, help="Новая папка результатов"
    )
    parser.add_argument(
        "--resume", type=Path, help="last.pt с полной завершённой эпохой"
    )
    parser.add_argument('--initial-weights', type=Path, help='Начальные best.pt/last.pt для нового обучения')
    parser.add_argument("--device", default="auto", help="auto, cpu или cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--edge-cache", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--stop_file")
    args = parser.parse_args(argv)
    from concurrent.futures import CancelledError
    from utils_new.configuration.autoencoder import TrainConfig
    from utils_new.import_export.json_objects import read_json_object

    stop = StopRequest(args.stop_file)
    if stop.is_set():
        return 130
    try:
        from runtime_torch import prepare_windows_torch

        prepare_windows_torch()
        initial_weights, resume = args.initial_weights, args.resume
        data, output, requested_device, plots = args.data, args.output, args.device, not args.no_plots
        if args.params_file:
            from utils_new.configuration.autoencoder import training_request

            request = training_request(read_json_object(args.params_file))
            raw = request['config']
            data = Path(request['source'])
            output = Path(request['project_path']) / request['name']
            requested_device, plots = request['device'], request['plots']
            resume, initial_weights = request['resume'] or None, request['initial_weights'] or None
            if not resume:
                base, number = output, 2
                while output.exists():
                    output = base.with_name(f'{base.name}_{number}')
                    number += 1
        elif args.resume and not args.config and not args.profile:
            from utils_new.import_export.autoencoder import read_checkpoint

            raw = read_checkpoint(args.resume).get("training_state", {}).get("config")
            if raw is None:
                raise ValueError("В --resume нет конфигурации обучения.")
        else:
            path = args.config or Path(__file__).resolve().parents[
                2
            ] / "settings/autoencoders" / (
                (args.profile or "residual_ae") + ".json"
            )
            raw = read_json_object(path)
        if data is None or output is None:
            raise ValueError('Укажите --data и --output либо --params_file.')
        if args.epochs is not None:
            raw["epochs"] = args.epochs
        if args.batch_size is not None:
            raw["batch_size"] = args.batch_size
        config = TrainConfig.from_dict(raw)
        import torch
        from utils_new.autoencoders.training import run_training

        device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if requested_device == "auto"
            else requested_device
        )
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Запрошена CUDA, но она недоступна.")
        with stop.signals():
            def report(event):
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if event['event'] == 'run_directory':
                    print('RESULT_DIR:' + event['output'], flush=True)
                elif event['event'] == 'stage':
                    print('STAGE:' + event['stage'], flush=True)
                elif event['event'] == 'epoch':
                    epoch = event['epoch']
                    print(f'PROGRESS:{int(epoch*100/config.epochs)}:{epoch}:{config.epochs}', flush=True)

            result = run_training(
                config,
                data,
                output,
                device=device,
                resume=resume,
                initial_weights=initial_weights,
                edge_cache=args.edge_cache,
                cancel=stop,
                progress=report,
                plots=plots,
            )
        print('METRICS: ' + json.dumps(result['calibration']), flush=True)
        print('SUCCESS', flush=True)
        return 0
    except CancelledError:
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    from utils_new.training.protocol import configure_worker_streams

    configure_worker_streams()
    raise SystemExit(main())
