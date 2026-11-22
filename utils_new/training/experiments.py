"""Stream parameter grids through isolated workers; keep metrics and test scores distinct."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from itertools import product, islice
import json
import os
from pathlib import Path
import subprocess
import sys

from utils_new.import_export.json_objects import read_json_object, write_json_atomic
from utils_new.training.protocol import worker_arguments


def generate_combinations(parameters):
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Нужна непустая сетка параметров.")
    keys = list(parameters)
    values = []
    for key in keys:
        value = (
            parameters[key] if isinstance(parameters[key], list) else [parameters[key]]
        )
        if not value:
            raise ValueError(f"Пустая ось: {key}")
        values.append(value)
    for row in product(*values):
        yield dict(zip(keys, row))


def run_experiments(kind, grid, output, *, limit=None, log=print, dry_run=False):
    if kind not in ("yolo", "unet", "deeplab"):
        raise ValueError(
            "Сетка поддерживает YOLO, U-Net и DeepLab; AE имеет собственный конфиг."
        )
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit должен быть положительным.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Папка экспериментов должна быть пустой.")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, params in enumerate(islice(generate_combinations(grid), limit), 1):
        values = dict(params)
        values["name"] = str(values.get("name") or kind) + f"_{index}"
        values.setdefault(
            "project" if kind == "yolo" else "project_path", str(output / "models")
        )
        config = output / f"params_{index:04d}.json"
        stop = output / f"stop_{index:04d}"
        write_json_atomic(config, values)
        command = [
            sys.executable,
            *worker_arguments(kind, params_file=config, stop_file=stop),
        ]
        record = dict(
            index=index,
            parameters=values,
            metrics=None,
            test_metrics=None,
            run_dir=None,
            status="planned",
        )
        if not dry_run:
            environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            with (output / f"run_{index:04d}.log").open(
                "w", encoding="utf-8"
            ) as transcript:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                )
                try:
                    for line in process.stdout:
                        transcript.write(line)
                        transcript.flush()
                        log(line.rstrip())
                        prefix, separator, value = line.strip().partition(":")
                        if separator and prefix in ("METRICS", "TEST_METRICS"):
                            try:
                                record[
                                    "metrics" if prefix == "METRICS" else "test_metrics"
                                ] = json.loads(value)
                            except ValueError:
                                log(f"[PROTOCOL] Некорректные метрики: {line.rstrip()}")
                        elif prefix == "RESULT_DIR":
                            record["run_dir"] = value
                    process.wait()
                    record["status"] = (
                        "completed"
                        if process.returncode == 0
                        else "cancelled"
                        if process.returncode == 130
                        else "failed"
                    )
                except KeyboardInterrupt:
                    stop.touch()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    record["status"] = "cancelled"
                finally:
                    process.stdout.close()
                    record["returncode"] = process.returncode
                    stop.unlink(missing_ok=True)
        rows.append(record)
        write_json_atomic(output / "results.json", dict(runs=rows))
        if record["status"] == "cancelled":
            break
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Сетка экспериментов без локальных путей в коде."
    )
    parser.add_argument("--kind", choices=["yolo", "unet", "deeplab"], required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows = run_experiments(
        args.kind,
        read_json_object(args.grid),
        args.output,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    return (
        130
        if any(r["status"] == "cancelled" for r in rows)
        else 1
        if any(r["status"] == "failed" for r in rows)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
