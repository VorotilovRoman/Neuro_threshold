#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import json
import os
import signal
import argparse
import traceback
import shutil
from datetime import datetime

def log(msg):
    """Вывод сообщения с немедленным сбросом буфера."""
    print(msg, flush=True)

def is_standard_yolo_model(model_path):
    """Определяет, является ли имя модели стандартным (без пути)."""
    if os.path.sep not in model_path and model_path.endswith('.pt'):
        return any(model_path.startswith(prefix) for prefix in
                   ['yolov8', 'yolo11', 'yolo26'])
    return False

# Глобальный флаг для остановки обучения
stop_requested = False
trainer_ref = None

def signal_handler(signum, frame):
    global stop_requested, trainer_ref
    log(f"Получен сигнал {signum}. Запрос остановки обучения...")
    stop_requested = True
    if trainer_ref is not None:
        trainer_ref.stop_training = True

def save_model_meta(meta_dict, model_path):
    """Сохраняет JSON-файл с метаданными рядом с .pt файлом."""
    meta_path = os.path.splitext(model_path)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    log(f"Метаданные модели сохранены в {meta_path}")

def main():
    global stop_requested, trainer_ref
    try:
        log("WORKER STARTED")

        parser = argparse.ArgumentParser()
        parser.add_argument('--params', type=str)
        parser.add_argument('--params_file', type=str)
        args = parser.parse_args()

        # Загрузка параметров
        if args.params_file:
            if not os.path.exists(args.params_file):
                log(f"Ошибка: файл {args.params_file} не найден")
                sys.exit(1)
            with open(args.params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
        elif args.params:
            params = json.loads(args.params)
        else:
            log("Ошибка: укажите --params или --params_file")
            sys.exit(1)

        log(f"Параметры получены: {list(params.keys())}")

        # ========== 1. Коррекция и проверка существования data.yaml ==========
        data_path = params.get('data')
        if not data_path:
            log("Ошибка: не указан параметр 'data'")
            sys.exit(1)

        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, 'data.yaml')
            params['data'] = data_path
            log(f"Автоматически скорректирован путь к data.yaml: {data_path}")

        if not os.path.isfile(data_path):
            log(f"Ошибка: файл датасета не найден: {data_path}")
            sys.exit(1)

        # ========== 2. Проверка существования файла модели ==========
        model_path = params.get('model')
        if not model_path:
            log("Ошибка: не указан параметр 'model'")
            sys.exit(1)

        if not os.path.exists(model_path):
            if is_standard_yolo_model(model_path):
                log(f"Предупреждение: модель '{model_path}' не найдена локально, будет загружена из интернета.")
            else:
                log(f"Ошибка: файл модели не найден: {model_path}")
                sys.exit(1)

        # Импорт YOLO
        try:
            from ultralytics import YOLO
            import ultralytics
            log(f"Ultralytics YOLO version: {ultralytics.__version__}")
        except ImportError as e:
            log(f"Ошибка импорта ultralytics: {e}")
            log("Установите: pip install ultralytics")
            sys.exit(1)

        # Устанавливаем обработчики сигналов (без sys.exit внутри)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        log("=== Загрузка модели ===")
        try:
            model = YOLO(model_path)
            log(f"Модель загружена: {model_path}")
        except Exception as e:
            log(f"Ошибка загрузки модели: {e}")
            sys.exit(1)

        # Формирование аргументов обучения
        train_args = {
            'data': data_path,
            'epochs': params.get('epochs', 100),
            'imgsz': params.get('imgsz', 640),
            'batch': params.get('batch', 16),
            'optimizer': params.get('optimizer', 'auto'),
            'lr0': params.get('lr0', 0.01),
            'lrf': params.get('lrf', 0.01),
            'verbose': True,
        }
        if params.get('close_mosaic') is not None:
            train_args['close_mosaic'] = params['close_mosaic']
        if params.get('mosaic') is not None:
            train_args['mosaic'] = params['mosaic']
        if params.get('project'):
            train_args['project'] = params['project']
        if params.get('name'):
            train_args['name'] = params['name']
        if params.get('device') and params['device'] != 'auto':
            train_args['device'] = params['device']
        if params.get('disable_aug'):
            train_args.update({
                'fliplr': 0.0, 'flipud': 0.0, 'scale': 0.0, 'translate': 0.0,
                'shear': 0.0, 'perspective': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0,
                'hsv_v': 0.0, 'copy_paste': 0.0, 'mixup': 0.0, 'erasing': 0.0,
                'auto_augment': 'none',
            })

        log("=== Начало обучения ===")
        log(json.dumps(train_args, indent=2, ensure_ascii=False))

        # Callback для проверки флага остановки
        def on_epoch_end(trainer_obj):
            global trainer_ref, stop_requested
            trainer_ref = trainer_obj
            epoch = trainer_obj.epoch + 1
            total = train_args['epochs']
            percent = int(epoch / total * 100)
            log(f"PROGRESS:{percent}:{epoch}:{total}")
            if stop_requested:
                log("Остановка обучения по запросу пользователя...")
                trainer_obj.stop_training = True

        def on_batch_end(trainer_obj):
            if stop_requested:
                trainer_obj.stop_training = True

        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_train_batch_end", on_batch_end)

        # Запуск обучения (блокирующий)
        results = model.train(**train_args)

        # После завершения обучения (нормально или по stop_training)
        save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else None
        if save_dir:
            weights_dir = os.path.join(save_dir, 'weights')
            if os.path.exists(weights_dir):
                best_pt = os.path.join(weights_dir, 'best.pt')
                last_pt = os.path.join(weights_dir, 'last.pt')

                # Если прерывание и файл best.pt существует – делаем резервную копию
                if stop_requested and os.path.exists(best_pt):
                    backup_best = best_pt.replace('.pt', '_interrupted.pt')
                    shutil.copy2(best_pt, backup_best)
                    log(f"Лучшая модель скопирована в {backup_best}")

                # Собираем метаданные
                meta = {
                    'model_type': 'YOLO-seg',
                    'model_source': model_path,
                    'data_yaml': data_path,
                    'epochs': train_args['epochs'],
                    'imgsz': train_args['imgsz'],
                    'batch': train_args['batch'],
                    'optimizer': train_args['optimizer'],
                    'lr0': train_args['lr0'],
                    'lrf': train_args['lrf'],
                    'close_mosaic': train_args.get('close_mosaic'),
                    'mosaic': train_args.get('mosaic'),
                    'disable_aug': params.get('disable_aug', False),
                    'project': train_args.get('project'),
                    'name': train_args.get('name'),
                    'device': train_args.get('device'),
                    'date_start': datetime.now().isoformat()
                }

                # Добавляем метрики, если они известны (после валидации)
                # Валидацию запустим позже, а пока метрики будут добавлены после
                # Сохраним метафайлы (пока без метрик)
                if os.path.exists(best_pt):
                    save_model_meta(meta, best_pt)
                if os.path.exists(last_pt):
                    save_model_meta(meta, last_pt)
                log("Начальные метаданные сохранены (без метрик).")

        # ========== Валидация и сохранение метрик ==========
        log("=== Запуск валидации для сбора метрик ===")
        val_results = model.val()
        metrics = {}

        if hasattr(val_results, 'results_dict') and val_results.results_dict:
            metrics = val_results.results_dict
        elif hasattr(val_results, 'box'):
            box = val_results.box
            if hasattr(box, 'ap50'):
                metrics['mAP50'] = box.ap50.item() if hasattr(box.ap50, 'item') else box.ap50
                metrics['mAP50-95'] = box.ap.item() if hasattr(box.ap, 'item') else box.ap
            elif hasattr(box, 'mp50'):
                metrics['mAP50'] = box.mp50
                metrics['mAP50-95'] = box.mp
            if hasattr(box, 'p'):
                metrics['precision'] = box.p
                metrics['recall'] = box.r

        metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        log(f"METRICS: {json.dumps(metrics)}")

        # Обновляем метафайлы с добавленными метриками и финальной датой
        if save_dir and os.path.exists(save_dir):
            weights_dir = os.path.join(save_dir, 'weights')
            if os.path.exists(weights_dir):
                best_pt = os.path.join(weights_dir, 'best.pt')
                last_pt = os.path.join(weights_dir, 'last.pt')
                meta['mAP50'] = metrics.get('mAP50')
                meta['mAP50-95'] = metrics.get('mAP50-95')
                meta['precision'] = metrics.get('precision')
                meta['recall'] = metrics.get('recall')
                meta['date_end'] = datetime.now().isoformat()
                if os.path.exists(best_pt):
                    save_model_meta(meta, best_pt)
                if os.path.exists(last_pt):
                    save_model_meta(meta, last_pt)
                log("Метаданные модели обновлены (с метриками).")

        if save_dir:
            log(f"Результаты сохранены в: {save_dir}")

        log("SUCCESS")

    except KeyboardInterrupt:
        log("Обучение прервано пользователем (внешнее прерывание)")
        sys.exit(0)
    except Exception as e:
        log(f"НЕПЕРЕХВАЧЕННАЯ ОШИБКА: {e}")
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()