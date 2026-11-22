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
    collect_training_images,
    dataset_manifest,
    seed_worker,
    split_images,
)
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
):
    """Each last.pt is a complete epoch. Interrupted partial epochs never replace it."""
    output = Path(output_dir).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(
            "Папка результатов не пуста. Выберите новую папку или --resume."
        )
    paths = collect_training_images(
        data_dir, exclude=[output] + ([edge_cache] if edge_cache else [])
    )
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
        train_paths, val_paths = state["train_paths"], state["val_paths"]
        if (
            set(paths) != set(train_paths + val_paths)
            or dataset_manifest(train_paths + val_paths) != state["manifest"]
        ):
            raise ValueError("Датасет изменился после checkpoint; resume остановлен.")
    else:
        train_paths, val_paths = split_images(paths, config.val_split, config.seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dataset_manifest(train_paths + val_paths)
    write_json_atomic(
        output / "args.json",
        dict(
            config=config.to_dict(),
            data=str(Path(data_dir).resolve()),
            device=str(device),
        ),
    )
    write_json_atomic(
        output / "split.json", dict(train=train_paths, val=val_paths, manifest=manifest)
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
    model = build_ae_model_from_cfg(config.model).to(torch_device)
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
                    output / "best.pt",
                    model,
                    config.model,
                    epoch=epoch,
                    val_loss=best_loss,
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
                rng=capture_rng(train_generator, val_generator, device=torch_device),
            )
            save_checkpoint(
                output / "last.pt",
                model,
                config.model,
                epoch=epoch,
                val_loss=val_loss,
                training_state=state,
            )
            save_history(output, history, plot=False)
            progress(dict(event="epoch", **history[-1]))
        checkpoint(cancel)
        if best_state is None:
            raise ValueError("Не получено ни одной корректной эпохи.")
        model.load_state_dict(best_state, strict=True)
        calibration = calibrate(model, val_loader, config, torch_device, cancel=cancel)
        save_checkpoint(
            output / "best.pt",
            model,
            config.model,
            epoch=best_epoch,
            val_loss=best_loss,
            calibration=calibration,
        )
        write_json_atomic(output / "calibration.json", calibration)
        save_history(output, history, plot=plots)
        if plots:
            save_reconstruction_examples(
                output / "recon_examples.png", model, val_dataset, torch_device
            )
        progress(
            dict(
                event="completed",
                checkpoint=str(output / "best.pt"),
                calibration=calibration,
            )
        )
        return dict(
            checkpoint=str(output / "best.pt"), history=history, calibration=calibration
        )
    except CancelledError:
        save_checkpoint(
            output / "interrupted.pt",
            model,
            config.model,
            epoch=history[-1]["epoch"] if history else 0,
        )
        progress(
            dict(
                event="cancelled",
                resume=str(output / "last.pt")
                if (output / "last.pt").exists()
                else None,
            )
        )
        raise
