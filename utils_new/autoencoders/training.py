"""Shared AE/VAE/GAN trainer, reproducible epoch resume and anomaly calibration."""

from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils_new.autoencoders.dataset import (
    AutoencoderDataset,
    training_sources,
)
from utils_new.training.optimization import seed_worker
from utils_new.autoencoders.losses import (
    ReconstructionLoss,
    beta_at,
    discriminator_loss,
    training_objective,
)
from utils_new.autoencoders.models import PatchDiscriminator, build_ae_model_from_cfg
from utils_new.import_export.autoencoder import (
    inference_contract,
    read_checkpoint,
    save_checkpoint,
    save_history,
    save_reconstruction_examples,
)
from utils_new.import_export.json_objects import write_json_atomic


def checkpoint(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Обучение AE отменено.")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def capture_rng(train_generator, val_generator, *, device="cpu"):
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all()
        if torch.device(device).type == "cuda"
        else [],
        train_generator=train_generator.get_state(),
        val_generator=val_generator.get_state(),
    )


def restore_rng(state, train_generator, val_generator):
    random.setstate(state["python"])
    kind, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state(
        (kind, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached)
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Возобновление требует тот же набор CUDA-устройств.")
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])
    val_generator.set_state(state["val_generator"])


def _batch(item, device):
    return tuple(
        item[k].to(device, non_blocking=True) for k in ("input", "target", "valid")
    )


def _check_gradient(model, max_norm):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if not torch.isfinite(norm):
        raise ValueError("Градиенты AE/GAN содержат NaN/Inf; эпоха не сохранена.")


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    beta,
    *,
    discriminator=None,
    opt_d=None,
    use_gan=False,
    cancel=None,
):
    model.train()
    if discriminator is not None:
        discriminator.train()
    totals = dict(loss=0.0, recon=0.0, kl=0.0, adv=0.0, discriminator=0.0)
    count = 0
    cfg = criterion.config
    for item in loader:
        checkpoint(cancel)
        inputs, target, valid = _batch(item, device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, extra = model(inputs)
        loss, recon, kl = training_objective(
            reconstruction, target, valid, extra, criterion, beta
        )
        adv = loss.new_zeros(())
        if use_gan:
            discriminator.requires_grad_(False)
            discriminator.eval()
            try:
                adv = -discriminator(reconstruction).mean()
                loss = loss + cfg.gan_weight * adv
            finally:
                discriminator.requires_grad_(True)
                discriminator.train()
        if not torch.isfinite(loss):
            raise ValueError("Потери AE содержат NaN/Inf; обучение остановлено.")
        loss.backward()
        _check_gradient(model, cfg.grad_clip)
        optimizer.step()
        d_value = loss.new_zeros(())
        if use_gan:
            opt_d.zero_grad(set_to_none=True)
            d_value = discriminator_loss(
                discriminator(target.detach()), discriminator(reconstruction.detach())
            )
            if not torch.isfinite(d_value):
                raise ValueError("Потери дискриминатора содержат NaN/Inf.")
            d_value.backward()
            _check_gradient(discriminator, cfg.grad_clip)
            opt_d.step()
        size = inputs.shape[0]
        count += size
        for name, value in zip(totals, (loss, recon, kl, adv, d_value)):
            totals[name] += float(value.detach()) * size
    if not count:
        raise ValueError("Обучающая выборка пуста.")
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, *, cancel=None):
    model.eval()
    model.deterministic_eval = criterion.config.model["vae_use_mean"]
    total, count = 0.0, 0
    for item in loader:
        checkpoint(cancel)
        inputs, target, valid = _batch(item, device)
        reconstruction, extra = model(inputs)
        # A fixed validation objective makes epochs with different KL warmup comparable.
        loss, _, _ = training_objective(
            reconstruction,
            target,
            valid,
            extra,
            criterion,
            criterion.config.beta_vae_target,
        )
        if not torch.isfinite(loss):
            raise ValueError("Валидация содержит NaN/Inf.")
        size = inputs.shape[0]
        count += size
        total += float(loss) * size
    if not count:
        raise ValueError("Валидационная выборка пуста.")
    return total / count


@torch.inference_mode()
def calibrate(model, loader, config, device, *, cancel=None):
    """Separate image-mean MAE and pixel-MAE thresholds; never score padding.

    A bounded uniform priority sample estimates the pixel percentile when the
    valid-pixel population exceeds calibration_pixels; sample size is recorded.
    """
    model.eval()
    model.deterministic_eval = config.model["vae_use_mean"]
    generator = np.random.default_rng(config.seed)
    scores = []
    pixels = np.empty(0, dtype=np.float32)
    priorities = np.empty(0)
    total_pixels = 0
    for item in loader:
        checkpoint(cancel)
        inputs, target, valid = _batch(item, device)
        reconstruction = model(inputs)[0]
        if (
            reconstruction.shape != target.shape
            or not torch.isfinite(reconstruction).all()
        ):
            raise ValueError("Некорректный выход AE при калибровке.")
        errors = (reconstruction - target).abs().mean(dim=1)
        for error, region in zip(errors, valid[:, 0] > 0):
            values = error[region].cpu().numpy()
            if not len(values):
                raise ValueError("Нет полезных пикселей для калибровки.")
            scores.append(float(values.mean()))
            total_pixels += len(values)
            pixels = np.concatenate((pixels, values))
            priorities = np.concatenate((priorities, generator.random(len(values))))
            if len(pixels) > config.calibration_pixels:
                selected = np.argpartition(priorities, config.calibration_pixels - 1)[
                    : config.calibration_pixels
                ]
                pixels, priorities = pixels[selected], priorities[selected]
    if not scores:
        raise ValueError("Нет снимков для калибровки.")
    return dict(
        version=1,
        metric="mae",
        contract=inference_contract(config.model),
        image_aggregation="mean_valid_pixels",
        pixel_aggregation="valid_pixels",
        image_threshold=float(np.percentile(scores, config.image_percentile)),
        pixel_threshold=float(np.percentile(pixels, config.pixel_percentile)),
        image_percentile=config.image_percentile,
        pixel_percentile=config.pixel_percentile,
        images=len(scores),
        valid_pixels=total_pixels,
        sampled_pixels=len(pixels),
        pixel_sampling="uniform_priority" if total_pixels > len(pixels) else "all",
        seed=config.seed,
    )


@torch.inference_mode()
def anomaly_metrics(model, loader, calibration, device, *, cancel=None):
    """Holdout diagnostics under the declared normal-only data assumption."""
    model.eval()
    scores, rejected, pixels, pixel_rejected = [], 0, 0, 0
    for item in loader:
        checkpoint(cancel)
        inputs, target, valid = _batch(item, device)
        reconstruction = model(inputs)[0]
        if reconstruction.shape != target.shape or not torch.isfinite(reconstruction).all():
            raise ValueError('Некорректная реконструкция AE на test.')
        errors = (reconstruction-target).abs().mean(1)
        for error, region in zip(errors, valid[:, 0] > 0):
            values = error[region]
            score = float(values.mean())
            scores.append(score)
            rejected += score > calibration['image_threshold']
            pixels += values.numel()
            pixel_rejected += int((values > calibration['pixel_threshold']).sum())
    return dict(metric='mae', images=len(scores), mean_image_mae=float(np.mean(scores)),
                fraction_images_above_threshold=rejected/len(scores),
                fraction_pixels_above_threshold=pixel_rejected/pixels,
                assumption='all test images are normal', used_for_selection=False, used_for_calibration=False)


def run_training(
    config,
    data_dir,
    output_dir,
    *,
    device="cpu",
    resume=None,
    edge_cache=None,
    cancel=None,
    progress=lambda value: None,
    plots=True,
    initial_weights=None,
):
    """Each last.pt is a complete epoch. Interrupted partial epochs never replace it."""
    output = Path(output_dir).resolve()
    if resume and initial_weights:
        raise ValueError('Нельзя одновременно выбрать resume и начальные веса.')
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(
            "Папка результатов не пуста. Выберите новую папку или --resume."
        )
    if resume and output.exists() and any(output.iterdir()):
        resume_root = Path(resume).resolve().parent
        if resume_root.name == 'weights':
            resume_root = resume_root.parent
        if resume_root != output:
            raise ValueError('Для продолжения в другой папке выберите пустую папку результатов.')
    progress(dict(event='stage', stage='Проверка источников и содержимого снимков'))
    data = training_sources(data_dir, config.val_split, config.seed,
                            exclude=[output] + ([edge_cache] if edge_cache else []), cancel=cancel)
    splits = data['splits']
    train_paths, val_paths = splits['train'], splits['val']
    calibration_paths, test_paths = splits['calibration'], splits['test']
    manifest = data['manifest']['train'] + data['manifest']['val']
    restored = read_checkpoint(resume) if resume else None
    if restored:
        state = restored.get("training_state")
        if (
            restored["legacy"]
            or not isinstance(state, dict)
            or state.get("epoch_complete") is not True
        ):
            raise ValueError(
                "Этот checkpoint не содержит состояния для возобновления обучения."
            )
        if state["config"] != config.to_dict():
            # JSON normalizes tuples to lists; use the canonical dataclass validation.
            from utils_new.configuration.autoencoder import TrainConfig

            if TrainConfig.from_dict(state["config"]).to_dict() != config.to_dict():
                raise ValueError("Параметры обучения отличаются от --resume.")
        if state.get("device") != str(torch.device(device)) or state.get(
            "torch_version"
        ) != str(torch.__version__):
            raise ValueError("Resume требует то же устройство и версию PyTorch.")
        if (
            train_paths != state['train_paths'] or val_paths != state['val_paths']
            or manifest != state['manifest']
            or calibration_paths != state.get('calibration_paths', [])
            or test_paths != state.get('test_paths', [])
            or ('split_manifest' in state and data['manifest'] != state['split_manifest'])
        ):
            raise ValueError("Датасет изменился после checkpoint; resume остановлен.")
    output.mkdir(parents=True, exist_ok=True)
    progress(dict(event='run_directory', output=str(output)))
    weights = output / 'weights'
    weights.mkdir(exist_ok=True)
    data_info = dict(source=data['source'], kind=data['kind'], auto_split=data['auto_split'],
                     counts={name: len(paths) for name, paths in splits.items()},
                     calibration_split='calibration' if calibration_paths else 'val',
                     annotation_policy='ignored; all selected images assumed normal')
    write_json_atomic(
        output / "args.json",
        dict(
            config=config.to_dict(),
            data=str(Path(data_dir).resolve()),
            device=str(device),
            initial_weights=str(initial_weights) if initial_weights else None,
        ),
    )
    write_json_atomic(
        output / "split.json", dict(**splits, manifest=data['manifest'], source=data['source'])
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train_generator = torch.Generator().manual_seed(config.seed)
    val_generator = torch.Generator().manual_seed(config.seed + 1)
    cache = edge_cache or output / ".edge_cache"
    train_dataset = AutoencoderDataset(
        train_paths, config, training=True, edge_cache=cache
    )
    val_dataset = AutoencoderDataset(val_paths, config, edge_cache=cache)
    arguments = dict(
        batch_size=min(config.batch_size, len(train_paths)),
        num_workers=config.num_workers,
        pin_memory=torch_device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, **arguments
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, generator=val_generator, **arguments
    )
    calibration_loader = (DataLoader(AutoencoderDataset(calibration_paths, config, edge_cache=cache),
                                     shuffle=False, generator=val_generator, **arguments)
                          if calibration_paths else val_loader)
    test_loader = (DataLoader(AutoencoderDataset(test_paths, config, edge_cache=cache),
                             shuffle=False, generator=val_generator, **arguments) if test_paths else None)
    model = build_ae_model_from_cfg(config.model).to(torch_device)
    if initial_weights:
        initial = read_checkpoint(initial_weights)
        for key in ('arch', 'model_type', 'in_channels', 'model_in_channels', 'out_channels'):
            if initial['config'][key] != config.model[key]:
                raise ValueError(f'Начальные веса не соответствуют выбранной модели: {key}.')
        try:
            model.load_state_dict(initial['model'], strict=True)
        except RuntimeError as error:
            raise ValueError(f'Размеры/архитектура начальных весов несовместимы: {error}') from error
    criterion = ReconstructionLoss(config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )
    discriminator = (
        PatchDiscriminator(
            config.model["out_channels"],
            config.discriminator_base,
            config.model["norm_type"],
        ).to(torch_device)
        if config.use_gan
        else None
    )
    opt_d = (
        torch.optim.AdamW(
            discriminator.parameters(), lr=config.gan_lr, betas=(0.5, 0.999)
        )
        if discriminator is not None
        else None
    )
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    start = 1
    if restored:
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if discriminator is not None:
            discriminator.load_state_dict(state["discriminator"], strict=True)
            opt_d.load_state_dict(state["opt_d"])
        history = deepcopy(state["history"])
        best_loss = state["best_loss"]
        best_state = state["best_state"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        start = restored["epoch"] + 1
        restore_rng(state["rng"], train_generator, val_generator)
    progress(
        dict(
            event="dataset",
            train=len(train_paths),
            val=len(val_paths),
            device=str(device),
            start_epoch=start,
            calibration=len(calibration_paths), test=len(test_paths), output=str(output),
        )
    )
    try:
        for epoch in range(start, config.epochs + 1):
            checkpoint(cancel)
            if stale >= config.patience:
                break
            beta = beta_at(epoch, config.beta_vae_warmup_epochs, config.beta_vae_target)
            stats = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                torch_device,
                beta,
                discriminator=discriminator,
                opt_d=opt_d,
                use_gan=config.use_gan and epoch >= config.gan_start_epoch,
                cancel=cancel,
            )
            val_loss = evaluate(
                model, val_loader, criterion, torch_device, cancel=cancel
            )
            scheduler.step()
            history.append(
                dict(
                    epoch=epoch,
                    train_loss=stats["loss"],
                    val_loss=val_loss,
                    reconstruction=stats["recon"],
                    kl=stats["kl"],
                    beta=beta,
                    adversarial=stats["adv"],
                    discriminator=stats["discriminator"],
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = cpu_state(model)
                best_epoch = epoch
                stale = 0
                save_checkpoint(
                    weights / "best.pt",
                    model,
                    config.model,
                    epoch=epoch,
                    val_loss=best_loss,
                    training_config=config.to_dict(), data_info=data_info, status='best_validation',
                )
            else:
                stale += 1
            state = dict(
                epoch_complete=True,
                device=str(torch_device),
                torch_version=str(torch.__version__),
                config=config.to_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                discriminator=discriminator.state_dict()
                if discriminator is not None
                else None,
                opt_d=opt_d.state_dict() if opt_d is not None else None,
                history=history,
                best_loss=best_loss,
                best_state=best_state,
                best_epoch=best_epoch,
                stale=stale,
                train_paths=train_paths,
                val_paths=val_paths,
                manifest=manifest,
                calibration_paths=calibration_paths, test_paths=test_paths, split_manifest=data['manifest'],
                rng=capture_rng(train_generator, val_generator, device=torch_device),
            )
            save_checkpoint(
                weights / "last.pt",
                model,
                config.model,
                epoch=epoch,
                val_loss=val_loss,
                training_state=state,
                training_config=config.to_dict(), data_info=data_info, status='complete_epoch',
            )
            save_history(output, history, plot=False)
            progress(dict(event="epoch", **history[-1]))
        checkpoint(cancel)
        if best_state is None:
            raise ValueError("Не получено ни одной корректной эпохи.")
        model.load_state_dict(best_state, strict=True)
        progress(dict(event='stage', stage=f'Калибровка лучших весов: {data_info["calibration_split"]}'))
        calibration = calibrate(model, calibration_loader, config, torch_device, cancel=cancel)
        calibration['source_split'] = data_info['calibration_split']
        calibration['independent_of_selection'] = bool(calibration_paths)
        save_checkpoint(
            weights / "best.pt",
            model,
            config.model,
            epoch=best_epoch,
            val_loss=best_loss,
            calibration=calibration,
            training_config=config.to_dict(), data_info=data_info, status='completed',
        )
        write_json_atomic(output / "calibration.json", calibration)
        write_json_atomic(output / 'metrics.json', dict(best_epoch=best_epoch, best_val_loss=best_loss,
                          epochs_completed=history[-1]['epoch'], calibration=calibration, data=data_info))
        if test_loader is not None:
            progress(dict(event='stage', stage='Оценка отложенного test (не участвует в выборе весов и порогов)'))
            write_json_atomic(output / 'test_metrics.json', anomaly_metrics(model, test_loader, calibration,
                                                                          torch_device, cancel=cancel))
        save_history(output, history, plot=plots)
        if plots:
            save_reconstruction_examples(
                output / "recon_examples.png", model, val_dataset, torch_device
            )
        progress(
            dict(
                event="completed",
                checkpoint=str(weights / "best.pt"),
                calibration=calibration,
            )
        )
        return dict(
            checkpoint=str(weights / "best.pt"), history=history, calibration=calibration
        )
    except CancelledError:
        save_checkpoint(
            weights / "interrupted.pt",
            model,
            config.model,
            epoch=history[-1]["epoch"] if history else 0,
            training_config=config.to_dict(), data_info=data_info, status='interrupted_partial_epoch',
        )
        progress(
            dict(
                event="cancelled",
                resume=str(weights / "last.pt")
                if (weights / "last.pt").exists()
                else None,
            )
        )
        raise
