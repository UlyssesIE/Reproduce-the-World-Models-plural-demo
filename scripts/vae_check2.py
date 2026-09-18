"""vae_check2.py — corrected VAE metrics.

Fixes two flaws in vae_diagnose.py:
  1. the per-channel VAE BCE was divided by an extra factor of n, so every
     channel printed 0.0003 and every "gain" printed 100%;
  2. "gain over the constant predictor" is a misleading headline here. With
     continuous [0,1] targets, BCE against a constant p has floor H(p) = 0.59-0.67
     per pixel, so even a perfect predictor gains little in percentage terms.

Reports BCE with correct units, plus explained variance R^2 and the MSE gain
over the per-pixel-mean baseline, which isolate the predictable (dynamic) part.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.frames import RolloutFrameDataset, split_files
from src.models.vae import VAE, VAEConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="runs/vae_pilot/best.pt")
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}  cuda={torch.cuda.is_available()}")

    ck = torch.load(args.vae, map_location=dev, weights_only=False)
    cfg = VAEConfig(**ck["cfg"])
    m = VAE(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    print(f"[model]  epoch={ck.get('epoch')} params={m.num_params():,} "
          f"recon={cfg.recon} z_dim={cfg.z_dim}")

    _, vf = split_files(args.data, 0.05, args.seed)
    ds = RolloutFrameDataset(files=vf, crop_bottom=12, size=64,
                             grayscale=(cfg.in_ch == 1))
    n = min(args.n, len(ds))
    idx = np.random.default_rng(args.seed).choice(len(ds), n, replace=False)
    xs = torch.stack([ds[int(i)] for i in idx])
    C, H, W = xs.shape[1], xs.shape[2], xs.shape[3]

    bce_sum = np.zeros(C)      # sum of BCE over frames*H*W, per channel
    se_sum = np.zeros(C)       # sum of squared error, per channel
    kl_dims = None
    nb = 0

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            x = xs[i:i + args.batch_size].to(dev)
            logits, mu, logvar, _ = m(x)

            err = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
            bce_sum += err.sum(dim=(0, 2, 3)).double().cpu().numpy()

            p = torch.sigmoid(logits) if cfg.recon == "bce" else logits
            se_sum += ((p - x) ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()

            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_dims = kld.sum(0) if kl_dims is None else kl_dims + kld.sum(0)
            nb += x.size(0)

    X = xs.numpy().astype(np.float64)
    npx = nb * H * W

    print()
    print("BCE (per pixel) and explained variance (MSE basis)")
    print(f"{'ch':>3}{'mean':>9}{'var':>10}"
          f"{'BCE_vae':>10}{'BCE_const':>11}{'BCE_pixmu':>11}"
          f"{'R2_pixmu':>10}{'R2_vae':>9}{'gain%':>8}")
    print("-" * 84)

    r2_vae_all, gain_all = [], []
    for c in range(C):
        xc = X[:, c]
        var = xc.var()
        mu_pix = xc.mean(axis=0)

        mse_pix = ((xc - mu_pix) ** 2).mean()
        mse_vae = se_sum[c] / npx

        r2_pix = 1.0 - mse_pix / var
        r2_vae = 1.0 - mse_vae / var
        gain = 100.0 * (1.0 - mse_vae / mse_pix)      # vs per-pixel mean

        pm = float(xc.mean())
        bce_const = float(-(xc * np.log(pm) + (1 - xc) * np.log(1 - pm)).mean())
        bce_pix = float(-(xc * np.log(mu_pix) + (1 - xc) * np.log(1 - mu_pix)).mean())

        r2_vae_all.append(r2_vae)
        gain_all.append(gain)
        print(f"{c:>3}{xc.mean():>9.4f}{var:>10.5f}"
              f"{bce_sum[c]/npx:>10.4f}{bce_const:>11.4f}{bce_pix:>11.4f}"
              f"{r2_pix:>10.3f}{r2_vae:>9.3f}{gain:>7.1f}%")

    print(f"\nmean R^2 over channels        : {np.mean(r2_vae_all):.3f}")
    print(f"mean MSE gain over per-pixel  : {np.mean(gain_all):.1f}%")

    kl_dims = (kl_dims / nb).cpu().numpy()
    active = int((kl_dims > 0.1).sum())
    print(f"\nKL total={kl_dims.sum():.2f}  active dims (KL>0.1)={active}/{len(kl_dims)}")

    print()
    r = float(np.mean(r2_vae_all))
    if r > 0.5:
        print("-> the VAE explains most of the frame variance: working well.")
    elif r > 0.2:
        print("-> the VAE captures a substantial part of the variance: usable.")
    elif r > 0.05:
        print("-> weak but non-trivial; the dynamic content is only partly captured.")
    else:
        print("-> essentially no variance explained: the VAE is not learning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
