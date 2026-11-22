"""Lightweight CLI entry boundary; --help never imports a model backend."""

import argparse
from concurrent.futures import CancelledError
import json
from pathlib import Path
import traceback

from utils_new.configuration.training import semantic_config
from utils_new.training.protocol import StopRequest, CANCELLED


def semantic_main(kind, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Обучение {kind}: общий датасет, метрики и безопасные checkpoints."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--params")
    source.add_argument("--params_file")
    parser.add_argument("--stop_file")
    args = parser.parse_args(argv)
    try:
        values = (
            json.loads(Path(args.params_file).read_text(encoding="utf-8-sig"))
            if args.params_file
            else json.loads(args.params)
        )
        params = semantic_config(values, kind)
        stop = StopRequest(args.stop_file)
        if stop.is_set():
            return CANCELLED
        from runtime_torch import require_torch

        require_torch()
        from utils_new.training.semantic import run_training

        with stop.signals():
            print("WORKER STARTED", flush=True)
            return run_training(
                params, kind, stop=stop, log=lambda message: print(message, flush=True)
            )
    except CancelledError:
        return CANCELLED
    except Exception:
        traceback.print_exc()
        return 1
