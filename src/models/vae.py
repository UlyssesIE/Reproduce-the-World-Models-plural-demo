"""Variational Autoencoder (V) for the World Models reproduction.

Paper: Ha & Schmidhuber (2018), "World Models".
Specified by the paper: input is a 64x64x3 frame; latent z in R^32;
reconstruction is lossy; V and M are trained separately from C.

NOT specified by the paper (chosen here, see Deviation Log):
  * conv depth / channels / kernel size
  * reconstruction loss (Bernoulli vs Gaussian)
  * beta (KL weight) and whether it is annealed
  * presence of free bits
  * output activation

The reported parameter count for the paper's CarRacing V is 4,348,547.
The default config here yields ~3.5M -- an acknowledged gap, since the
paper's architecture is not described in enough detail to reproduce exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class VAEConfig:
    z_dim: int = 32
    in_ch: int = 3
    frame_size: int = 64                 # NEW: used for shape validation
    enc_widths: tuple = (32, 64, 128, 256)
    dec_widths: tuple = (256, 128, 64, 32)
    hidden: int = 256
    kernel: int = 4
    stride: int = 2
    padding: int = 1                     # NEW: k=4, s=2 -> p=1 gives exact /2
    recon: str = "bce"
    beta: float = 1.0
    free_bits: float = 0.0
    recon_reduction: str = "sum"


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _init(m: nn.Module) -> None:
    """Orthogonal init on conv/fc with small gain -- stabilises VAE training."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    """64x64x3 -> (mu, logvar) in R^z_dim."""

    """64x64x3 -> (mu, logvar) in R^z_dim."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        layers, c_in = [], cfg.in_ch
        for c_out in cfg.enc_widths:
            layers += [
                nn.Conv2d(c_in, c_out, cfg.kernel, cfg.stride, cfg.padding),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # Measure the real output shape instead of deriving it from a formula.
        # A wrong padding convention silently changes the spatial size and only
        # surfaces later as a Linear shape error.
        with torch.no_grad():
            probe = self.conv(
                torch.zeros(1, cfg.in_ch, cfg.frame_size, cfg.frame_size)
            )
        self.spatial = int(probe.shape[-1])
        self.flat_dim = int(probe.flatten(1).shape[1])
        expected = cfg.frame_size // (2 ** len(cfg.enc_widths))
        assert self.spatial == expected, (
            f"conv stack yields {self.spatial}x{self.spatial}, expected "
            f"{expected}x{expected}; with k={cfg.kernel}, s={cfg.stride} use "
            f"padding={cfg.kernel // 2 - 1}"
        )

        head_in = cfg.hidden
        self.fc = (
            nn.Sequential(nn.Linear(self.flat_dim, cfg.hidden), nn.ReLU(inplace=True))
            if cfg.hidden
            else nn.Identity()
        )
        if not cfg.hidden:
            head_in = self.flat_dim
        self.mu = nn.Linear(head_in, cfg.z_dim)
        self.logvar = nn.Linear(head_in, cfg.z_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig, spatial: int):
        super().__init__()
        self.cfg = cfg
        self.spatial = spatial
        self.first_ch = cfg.dec_widths[0]

        head_in = cfg.hidden or cfg.z_dim
        self.fc = (
            nn.Sequential(nn.Linear(cfg.z_dim, cfg.hidden), nn.ReLU(inplace=True))
            if cfg.hidden else nn.Identity()
        )
        self.expand = nn.Linear(head_in, self.first_ch * spatial * spatial)

        layers, chans = [], list(cfg.dec_widths)
        for i in range(len(chans) - 1):
            layers += [
                nn.ConvTranspose2d(chans[i], chans[i + 1], cfg.kernel, cfg.stride,
                                   cfg.padding),
                nn.ReLU(inplace=True),
            ]
        layers.append(
            nn.ConvTranspose2d(chans[-1], cfg.in_ch, cfg.kernel, cfg.stride,
                               cfg.padding)
        )
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = self.expand(h).view(-1, self.first_ch, self.spatial, self.spatial)
        out = self.deconv(h)
        # BCE -> return logits (stable w/ BCEWithLogitsLoss)
        # MSE -> return probabilities in [0,1]
        return out if self.cfg.recon == "bce" else torch.sigmoid(out)


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg, spatial=self.encoder.spatial)
        self.apply(_init)

        # Fail loudly here rather than 10 minutes into training.
        with torch.no_grad():
            x = torch.zeros(1, self.cfg.in_ch, self.cfg.frame_size, self.cfg.frame_size)
            mu, _ = self.encoder(x)
            out = self.decoder(mu)
        assert out.shape == x.shape, (
            f"VAE round-trip shape {tuple(out.shape)} != input {tuple(x.shape)}"
        )

    # -- core ops ---------------------------------------------------------- #
    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * sigma,  eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar, z

    # -- losses ------------------------------------------------------------ #
    def loss(self, x: torch.Tensor):
        recon, mu, logvar, _ = self.forward(x)
        rec, kl = self.recon_kl(recon, x, mu, logvar)
        total = rec + self.cfg.beta * kl
        return total.mean(), rec.mean(), kl.mean()

    def recon_kl(self, recon, x, mu, logvar, beta: float | None = None):
        """Return per-sample (recon, kl). beta is applied by the caller/beta-schedule."""
        n_px = x[0].numel()

        if self.cfg.recon == "bce":
            err = F.binary_cross_entropy_with_logits(recon, x, reduction="none")
        else:
            err = F.mse_loss(recon, x, reduction="none")
        err = err.flatten(1)
        rec = err.sum(1) if self.cfg.recon_reduction == "sum" else err.mean(1)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)

        if self.cfg.free_bits > 0:
            # Per-dimension floor to prevent posterior collapse, then sum.
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl = torch.clamp(kl_per_dim, min=self.cfg.free_bits).sum(1)

        return rec, kl

    # -- utilities --------------------------------------------------------- #
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction (uses mu, no sampling). Returns [0,1] images."""
        mu, _ = self.encode(x)
        out = self.decode(mu)
        return torch.sigmoid(out) if self.cfg.recon == "bce" else out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}
