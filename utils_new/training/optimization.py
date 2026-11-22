"""Optimizers, EMA, encoder stages, padding and multiscale forward; no Qt."""

import copy
import random
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn, optim


def get_optimizer(model, opt_name, lr, momentum=0.9, weight_decay=0.0001):
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Неизвестный оптимизатор: {opt_name}")


def get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-6):
    sn = (
        scheduler_name.lower()
        .replace("steplr", "step")
        .replace("cosineannealing", "cosine")
    )
    if sn == "poly":
        return None
    elif sn == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif sn == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif sn == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
    if sn not in ("none", ""):
        raise ValueError(f"Неизвестный scheduler: {scheduler_name}")
    return None


class PolyLRSchedulerIter:
    def __init__(self, optimizer, max_iter, power=0.9, min_lr=0.0):
        self.optimizer = optimizer
        self.max_iter = max(1, int(max_iter))
        self.power = float(power)
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        factor = max(0.0, (1.0 - self.current_iter / self.max_iter)) ** self.power
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = max(self.min_lr, self.base_lrs[i] * factor)

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)

    def state_dict(self):
        return self.ema.state_dict()


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def _pad_to_multiple(x, multiple=16):
    h, w = x.shape[-2], x.shape[-1]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return x, (h, w)
    x = torch.nn.functional.pad(x, (0, nw - w, 0, nh - h), mode="constant", value=0)
    return x, (h, w)


def _crop_to_hw(x, hw):
    h, w = hw
    return x[..., :h, :w]


def _forward_safe(model, x, use_amp, device, output_stride=None):
    """
    Патч 3: убран nan_to_num(nan=0.0) — он маскировал проблему,
    превращая nan-логиты в 0 → softmax([0,0]) = [0.5,0.5] → argmax = «всё фон».
    Теперь nan передаётся наружу, его ловит isfinite(loss) guard в цикле.
    """
    if output_stride is None:
        output_stride = getattr(model.encoder, "output_stride", 16)
    try:
        output_stride = int(output_stride)
    except (TypeError, ValueError):
        output_stride = 16

    padded, hw = _pad_to_multiple(x, output_stride)

    if use_amp and device.type == "cuda":
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(padded)
    else:
        out = model(padded)

    out = _crop_to_hw(out, hw)
    return out.float()


def predict_batch_multiscale(model, images, device, scales, use_amp):
    H, W = images.shape[2], images.shape[3]
    if not scales:
        return _forward_safe(model, images, use_amp, device)

    avg = None
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            scaled = images
        else:
            nh = max(16, int(round(H * s / 16)) * 16)
            nw = max(16, int(round(W * s / 16)) * 16)
            scaled = torch.nn.functional.interpolate(
                images, size=(nh, nw), mode="bilinear", align_corners=False
            )
        out = _forward_safe(model, scaled, use_amp, device)
        if out.shape[2:] != (H, W):
            out = torch.nn.functional.interpolate(
                out, size=(H, W), mode="bilinear", align_corners=False
            )
        avg = out if avg is None else avg + out
    return (avg / len(scales)).float()


def encoder_stages(encoder):
    """SMP get_stages may be a dict or contain nested lists. Cover every parameter."""
    if all(
        hasattr(encoder, key)
        for key in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
    ):
        return [
            nn.Sequential(encoder.conv1, encoder.bn1),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        ]
    try:
        raw = encoder.get_stages()
    except AttributeError:
        raw = list(encoder.children())

    def modules(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from modules(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from modules(nested)
        elif isinstance(value, nn.Module):
            yield value

    stages = list(modules(raw))
    return stages or [encoder]


def set_encoder_phase(model, *, frozen=False, last_n=None):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not frozen and last_n is None)
    if not frozen and last_n is not None:
        stages = encoder_stages(model.encoder)
        if last_n:
            for stage in stages[-last_n:]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(True)


def layerwise_optimizer(
    model, name, lr, momentum=0.9, weight_decay=0.0001, decay_factor=0.1
):
    groups = []
    seen = set()
    stages = encoder_stages(model.encoder)
    for index, stage in enumerate(stages):
        parameters = []
        for parameter in stage.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
        if parameters:
            groups.append(
                dict(
                    params=parameters, lr=lr * decay_factor ** (len(stages) - 1 - index)
                )
            )
    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if remaining:
        groups.append(dict(params=remaining, lr=lr))
    cls = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
    }.get(name.lower())
    if cls is None:
        raise ValueError(f"Неизвестный оптимизатор: {name}")
    options = dict(lr=lr, weight_decay=weight_decay)
    if cls is optim.SGD:
        options["momentum"] = momentum
    return cls(groups, **options)


def resolve_device(value):
    name = str(value).lower()
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.isdigit():
        name = "cuda:" + name
    if name in ("cuda", "gpu"):
        name = "cuda:0"
    device = torch.device(name)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index or 0) >= torch.cuda.device_count()
    ):
        raise ValueError(f"Устройство {device} недоступно.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS недоступен.")
    return device


def probe_batch_size(model, criterion, dataset, device, maximum=64, log=print):
    """Probe a real forward/backward without changing weights, BN buffers or RNG."""
    if device.type != "cuda":
        return min(2, len(dataset))
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    was_training = model.training
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    selected = 1
    try:
        image, mask = dataset[0]
        for batch in (1, 2, 4, 8, 16, 32, 64):
            if batch > min(maximum, len(dataset)):
                break
            try:
                model.train()
                model.zero_grad(set_to_none=True)
                with singleton_batchnorm(model, batch):
                    output = _forward_safe(
                        model,
                        image[None].repeat(batch, 1, 1, 1).to(device),
                        False,
                        device,
                    )
                    loss = criterion(output, mask[None].repeat(batch, 1, 1).to(device))
                loss.backward()
                selected = batch
                del output, loss
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise ValueError(
                        "Даже один снимок не помещается в GPU: уменьшите input_size."
                    )
                break
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        log(f"Auto batch: {selected}")
        return selected
    finally:
        model.load_state_dict(original)
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def seed_worker(worker_id):
    import random

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def singleton_batchnorm(model, batch_size):
    layers = (
        [
            m
            for m in model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.training
        ]
        if batch_size == 1
        else []
    )
    for layer in layers:
        layer.eval()
    try:
        yield
    finally:
        for layer in layers:
            layer.train()
