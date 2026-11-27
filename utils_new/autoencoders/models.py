"""Checkpoint-compatible AE/VAE architectures; model parameters have one owner."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 128
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.encoder = ConvEncoder(in_channels)
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoder(in_channels)

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        z = self.fc_encode(z)
        z = self.fc_decode(z)
        z = z.view(z.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(z), None


class ConvVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 128
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoder(in_channels)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(
            logvar,
            getattr(self, "logvar_min", -10.0),
            getattr(self, "logvar_max", 10.0),
        )
        if not self.training and getattr(self, "deterministic_eval", False):
            return mu, logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z, logvar_c = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(h_dec), (mu, logvar_c)


def _gn(channels, max_groups=8):
    """GroupNorm с автоматическим подбором числа групп — как в обучении."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class ConvEncoderGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            _gn(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=1),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoderGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=2),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            _gn(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ConvAutoencoderGN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 64
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.encoder = ConvEncoderGN(in_channels)
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoderGN(in_channels)

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        z = self.fc_encode(z)
        z = self.fc_decode(z)
        z = z.view(z.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(z), None


class ConvVAEGN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 64
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.latent_dim = latent_dim
        self.encoder = ConvEncoderGN(in_channels)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoderGN(in_channels)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(
            logvar,
            getattr(self, "logvar_min", -10.0),
            getattr(self, "logvar_max", 10.0),
        )
        if not self.training and getattr(self, "deterministic_eval", False):
            return mu, logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z, logvar_c = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(h_dec), (mu, logvar_c)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        b = base
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(b, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(b * 2, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(b * 4, b * 8, 3, padding=1),
            nn.BatchNorm2d(b * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 8, b * 8, 3, padding=1),
            nn.BatchNorm2d(b * 8),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(self.pool(f0))
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        return f0, f1, f2, f3


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, base=32):
        super().__init__()
        b = base
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(b * 4 + b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(b * 2 + b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(b * 2, b, 2, stride=2),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(b + b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h, f0, f1, f2):
        d = self.up2(h)
        d = torch.cat([d, f2], dim=1)
        d = self.dec2(d)
        d = self.up1(d)
        d = torch.cat([d, f1], dim=1)
        d = self.dec1(d)
        d = self.up0(d)
        d = torch.cat([d, f0], dim=1)
        d = self.dec0(d)
        return d


class UNetAE(nn.Module):
    def __init__(self, in_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.encoder = UNetEncoder(in_channels, base)
        self.compress = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(in_channels, base)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        z = self.compress(f3)
        h = self.expand(z)
        if getattr(self, "skip_mode", "full") == "none":
            f0, f1, f2 = (torch.zeros_like(f) for f in (f0, f1, f2))
        recon = self.decoder(h, f0, f1, f2)
        return recon, None


class UNetVAE(nn.Module):
    def __init__(self, in_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.encoder = UNetEncoder(in_channels, base)
        self.fc_mu = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(in_channels, base)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(
            logvar,
            getattr(self, "logvar_min", -10.0),
            getattr(self, "logvar_max", 10.0),
        )
        if not self.training and getattr(self, "deterministic_eval", False):
            return mu, logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        mu = self.fc_mu(f3)
        logvar = self.fc_logvar(f3)
        z, logvar_c = self.reparameterize(mu, logvar)
        h = self.expand(z)
        if getattr(self, "skip_mode", "full") == "none":
            f0, f1, f2 = (torch.zeros_like(f) for f in (f0, f1, f2))
        recon = self.decoder(h, f0, f1, f2)
        return recon, (mu, logvar_c)


def _norm_v2(c, kind="gn"):
    """GroupNorm с тем же подбором групп, что и в тренировочном скрипте."""
    if kind == "gn":
        g = min(8, c)
        while g > 1 and c % g != 0:
            g -= 1
        return nn.GroupNorm(g, c)
    return nn.BatchNorm2d(c)


def _act_v2():
    return nn.LeakyReLU(0.1, inplace=True)


class ConvBlockV2(nn.Module):
    def __init__(self, in_c, out_c, stride=1, n_convs=2, norm_type="gn"):
        super().__init__()
        layers = []
        for i in range(n_convs):
            s = stride if i == 0 else 1
            layers += [
                nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, stride=s, padding=1),
                _norm_v2(out_c, norm_type),
                _act_v2(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DilatedResidualV2(nn.Module):
    def __init__(self, c, dilations=(2, 4), norm_type="gn"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=d, dilation=d),
                    _norm_v2(c, norm_type),
                    _act_v2(),
                    nn.Conv2d(c, c, 3, padding=d, dilation=d),
                    _norm_v2(c, norm_type),
                )
                for d in dilations
            ]
        )
        self.act = _act_v2()

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return self.act(x)


class UNetEncoderV2(nn.Module):
    def __init__(self, in_channels, base=32, dilations=(2, 4), norm_type="gn"):
        super().__init__()
        b = base
        self.enc0 = ConvBlockV2(in_channels, b, stride=1, norm_type=norm_type)
        self.enc1 = ConvBlockV2(b, b * 2, stride=2, norm_type=norm_type)
        self.enc2 = ConvBlockV2(b * 2, b * 4, stride=2, norm_type=norm_type)
        self.enc3 = ConvBlockV2(b * 4, b * 8, stride=2, norm_type=norm_type)
        self.bottleneck = DilatedResidualV2(b * 8, dilations, norm_type=norm_type)

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f3 = self.bottleneck(f3)
        return f0, f1, f2, f3


class DecoderBlockV2(nn.Module):
    def __init__(self, in_c, skip_c, out_c, norm_type="gn"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.norm_up = _norm_v2(out_c, norm_type)
        self.conv1 = nn.Conv2d(out_c + skip_c, out_c, 3, padding=1)
        self.n1 = _norm_v2(out_c, norm_type)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.n2 = _norm_v2(out_c, norm_type)
        self.act = _act_v2()

    def forward(self, x, skip):
        x = self.act(self.norm_up(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class UNetDecoderV2(nn.Module):
    def __init__(self, out_channels, base=32, norm_type="gn"):
        super().__init__()
        b = base
        self.dec2 = DecoderBlockV2(b * 8, b * 4, b * 4, norm_type)
        self.dec1 = DecoderBlockV2(b * 4, b * 2, b * 2, norm_type)
        self.dec0 = DecoderBlockV2(b * 2, b, b, norm_type)
        self.head = nn.Sequential(
            nn.Conv2d(b, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h, f0, f1, f2):
        d = self.dec2(h, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, f0)
        return self.head(d)


class UNetAEV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        img_h,
        img_w,
        base=32,
        dilations=(2, 4),
        norm_type="gn",
    ):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.encoder = UNetEncoderV2(in_channels, base, dilations, norm_type)
        self.compress = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoderV2(out_channels, base, norm_type)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        z = self.compress(f3)
        h = self.expand(z)
        if getattr(self, "skip_mode", "full") == "none":
            f0, f1, f2 = (torch.zeros_like(f) for f in (f0, f1, f2))
        recon = self.decoder(h, f0, f1, f2)
        return recon, None


class UNetVAEV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        img_h,
        img_w,
        base=32,
        dilations=(2, 4),
        norm_type="gn",
        logvar_min=-10.0,
        logvar_max=10.0,
    ):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.encoder = UNetEncoderV2(in_channels, base, dilations, norm_type)
        self.fc_mu = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoderV2(out_channels, base, norm_type)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        if not self.training and getattr(self, "deterministic_eval", False):
            return mu, logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        mu = self.fc_mu(f3)
        logvar = self.fc_logvar(f3)
        z, logvar_c = self.reparameterize(mu, logvar)
        h = self.expand(z)
        if getattr(self, "skip_mode", "full") == "none":
            f0, f1, f2 = (torch.zeros_like(f) for f in (f0, f1, f2))
        recon = self.decoder(h, f0, f1, f2)
        return recon, (mu, logvar_c)


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_type):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        def norm():
            return (_gn(channels, min(8, max(1, channels // 2)))
                    if norm_type == 'gn' else nn.BatchNorm2d(channels))
        self.norm1 = norm()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = norm()

    def forward(self, x):
        return x + self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x)))))


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 GRN on channels-last features; identity at initialization."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x):
        response = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * normalized) + self.beta


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, channels * 4)
        self.grn = GlobalResponseNorm(channels * 4)
        self.project = nn.Linear(channels * 4, channels)

    def forward(self, x):
        h = self.norm(self.depthwise(x).permute(0, 2, 3, 1))
        h = self.project(self.grn(F.gelu(self.expand(h))))
        return x + h.permute(0, 3, 1, 2)


class SpatialAutoencoder(nn.Module):
    """Spatial AE/VAE: every reconstructed pixel passes through z, no U-Net skips.

    Custom reconstruction backbones using residual or ConvNeXt V2 blocks,
    not pretrained ImageNet classifiers or a reproduction of FCMAE training.
    """
    def __init__(self, cfg):
        super().__init__()
        self.model_type = cfg['model_type']
        self.logvar_min, self.logvar_max = cfg['logvar_min'], cfg['logvar_max']
        self.deterministic_eval = cfg['vae_use_mean']
        widths = [cfg['base'] * factor for factor in (1, 2, 4, 8)]
        def stage(channels):
            return nn.Sequential(*[
                ConvNeXtV2Block(channels) if cfg['arch'] == 'convnext_v2'
                else ResidualBlock(channels, cfg['norm_type'])
                for _ in range(cfg['blocks_per_stage'])])
        self.stem = nn.Conv2d(cfg['model_in_channels'], widths[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList([stage(c) for c in widths])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 2, stride=2) for a, b in zip(widths, widths[1:])])
        if self.model_type == 'vae':
            self.fc_mu = nn.Conv2d(widths[-1], cfg['latent_channels'], 1)
            self.fc_logvar = nn.Conv2d(widths[-1], cfg['latent_channels'], 1)
        else:
            self.compress = nn.Conv2d(widths[-1], cfg['latent_channels'], 1)
        self.expand = nn.Conv2d(cfg['latent_channels'], widths[-1], 1)
        self.up = nn.ModuleList([nn.Conv2d(a, b, 3, padding=1) for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.decoder_stages = nn.ModuleList([stage(c) for c in reversed(widths)])
        self.head = nn.Conv2d(widths[0], cfg['out_channels'], 3, padding=1)

    def encode(self, x):
        h = self.encoder_stages[0](self.stem(x))
        for down, stage in zip(self.down, self.encoder_stages[1:]):
            h = stage(down(h))
        if self.model_type == 'ae':
            return self.compress(h), None
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(self.logvar_min, self.logvar_max)
        z = mu if not self.training and self.deterministic_eval else mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, (mu, logvar)

    def decode(self, z):
        h = self.decoder_stages[0](self.expand(z))
        for up, stage in zip(self.up, self.decoder_stages[1:]):
            h = stage(up(F.interpolate(h, scale_factor=2, mode='nearest')))
        return self.head(h).sigmoid()

    def forward(self, x):
        z, statistics = self.encode(x)
        return self.decode(z), statistics


def detect_arch_from_state_dict(state_dict):
    """
    Определяет архитектуру по ключам state_dict.
    Приоритет: unet_v2 → unet → conv (BN) → conv_gn.
    """
    keys = list(state_dict.keys())
    if any(k.startswith("encoder_stages.") for k in keys):
        return "convnext_v2" if any(".depthwise." in k for k in keys) else "residual"
    if any(k.startswith("encoder.bottleneck.") for k in keys):
        return "unet_v2"
    if any(k.startswith("encoder.enc0.net.") for k in keys):
        return "unet_v2"
    if any(k.startswith("encoder.enc0.") for k in keys):
        return "unet"
    if any(k.endswith("running_mean") for k in keys):
        return "conv"  # старая архитектура (BatchNorm, enc_c=128)
    return "conv_gn"  # новая (GroupNorm, enc_c=64)


def detect_arch(cfg, state_dict=None):
    """Use actual weights first, then a versioned config's explicit architecture.

    Unversioned historical configs retain their marker-based fallback: some
    earlier V2 checkpoints were tagged as the original U-Net.
    """
    # 1) state_dict — истина в последней инстанции
    if state_dict is not None:
        detected = detect_arch_from_state_dict(state_dict)
        if detected in ("unet_v2", "unet", "conv", "conv_gn", "residual", "convnext_v2"):
            return detected

    # Modern explicit architectures never existed with historical U-Net tags.
    if cfg.get('arch') in ('residual', 'convnext_v2'):
        return cfg['arch']

    if cfg.get("preprocessing_version") and cfg.get("arch") in (
        "unet_v2",
        "unet",
        "conv",
        "conv_gn",
        "residual",
        "convnext_v2",
    ):
        return cfg["arch"]

    # 2) специфичные маркеры unet_v2 в cfg
    if (
        "use_edge_channel" in cfg
        or "bottleneck_dilations" in cfg
        or "model_in_channels" in cfg
        or "norm_type" in cfg
    ):
        return "unet_v2"

    # 3) явный arch
    arch = cfg.get("arch")
    if arch in ("unet_v2", "unet", "conv", "conv_gn", "residual", "convnext_v2"):
        return arch

    # 4) latent_channels без маркеров → старый unet
    if "latent_channels" in cfg:
        return "unet"

    # 5) дефолт
    return "conv"


def _build_model(cfg):
    """Construct an already validated configuration, without changing parameter names."""
    arch = cfg["arch"]
    model_type = cfg["model_type"]
    in_channels = cfg["in_channels"]
    img_h, img_w = cfg["img_h"], cfg["img_w"]

    if arch in ("residual", "convnext_v2"):
        return SpatialAutoencoder(cfg)

    # ---- unet_v2 (новый U-Net с edge-каналом и dilated bottleneck) ----
    if arch == "unet_v2":
        model_in_channels = int(cfg.get("model_in_channels", in_channels))
        out_channels = int(cfg.get("out_channels", in_channels))
        latent_channels = int(cfg["latent_channels"])
        norm_type = str(cfg.get("norm_type", "gn"))
        dilations = tuple(cfg.get("bottleneck_dilations", (2, 4)))
        base = int(cfg.get("base", 32))
        logvar_min = float(cfg.get("logvar_min", -10.0))
        logvar_max = float(cfg.get("logvar_max", 10.0))

        if model_type == "vae":
            return UNetVAEV2(
                model_in_channels,
                out_channels,
                latent_channels,
                img_h,
                img_w,
                base=base,
                dilations=dilations,
                norm_type=norm_type,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
            )
        return UNetAEV2(
            model_in_channels,
            out_channels,
            latent_channels,
            img_h,
            img_w,
            base=base,
            dilations=dilations,
            norm_type=norm_type,
        )

    # ---- старый U-Net (BatchNorm, MaxPool, без edge) ----
    if arch == "unet":
        latent_channels = int(cfg["latent_channels"])
        if model_type == "vae":
            return UNetVAE(in_channels, latent_channels, img_h, img_w)
        return UNetAE(in_channels, latent_channels, img_h, img_w)

    # ---- conv / conv_gn ----
    latent_dim = int(cfg["latent_dim"])

    if arch == "conv_gn":
        if model_type == "vae":
            return ConvVAEGN(in_channels, latent_dim, img_h, img_w)
        return ConvAutoencoderGN(in_channels, latent_dim, img_h, img_w)

    if model_type == "vae":
        return ConvVAE(in_channels, latent_dim, img_h, img_w)
    return ConvAutoencoder(in_channels, latent_dim, img_h, img_w)


def build_ae_model_from_cfg(cfg, arch=None):
    from utils_new.configuration.autoencoder import normalize_model_config

    cfg = normalize_model_config(cfg, arch=arch or detect_arch(cfg))
    model = _build_model(cfg)
    model.logvar_min, model.logvar_max = cfg["logvar_min"], cfg["logvar_max"]
    model.deterministic_eval = cfg["vae_use_mean"]
    model.skip_mode = cfg.get("skip_mode", "full")
    return model


class PatchDiscriminator(nn.Module):
    """Training-only PatchGAN; shares the V2 normalization implementation."""

    def __init__(self, in_channels, base=32, norm_type="gn"):
        super().__init__()

        def block(i, o, stride, norm=True):
            layers = [nn.Conv2d(i, o, 4, stride, 1)]
            if norm:
                layers.append(_norm_v2(o, norm_type))
            return nn.Sequential(*layers, nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(
            block(in_channels, base, 2, False),
            block(base, base * 2, 2),
            block(base * 2, base * 4, 2),
            block(base * 4, base * 4, 1),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, image):
        return self.net(image)
