"""vae_baseline.py — is the VAE actually reconstructing, or just outputting an
average image?

Compares the VAE's reconstruction loss to two trivial baselines:

  * global constant : predict the dataset-wide mean for every pixel/channel
  * per-pixel mean  : predict, for each pixel/channel, its mean over the dataset
                      (the best possible predictor that ignores the input)

If the VAE is not clearly better than per-pixel mean, it has collapsed: the
decoder emits an average frame regardless of z.

Also prints per-dimension KL to expose posterior collapse.

Usage:
  python scripts/vae_baseline.py --vae runs/vae_pilot/best.pt --data data/pilot
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
    ap.add_argument("--vae", required=True)
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  cuda_available={torch.cuda.is_available()}")

    ck = torch.load(args.vae, map_location=device)
    cfg = VAEConfig(**ck["cfg"])
    model = VAE(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"[model]  epoch={ck.get('epoch')} params={model.num_params():,} "
          f"recon={cfg.recon} beta={cfg.beta} reduction={cfg.recon_reduction}")
    if cfg.recon != "bce":
        print("!! baselines below assume BCE; numbers are not comparable")

    _, val_files = split_files(args.data, 0.05, args.seed)\
        if False else split_files(args.data, 0.05, args.seed)
    ds = RolloutFrameDataset(files=val_files, crop_bottom=12, size=64,
                             grayscale=(cfg.in_ch == 1))
    n = min(args.n, len(ds))
    idx = np.random.default_rng(args.seed).choice(len(ds), n, replace=False)
    xs = torch.stack([ds[int(i)] for i in idx])          # (n,C,H,W), values in [0,1]
    print(f"[data]   {n} validation frames  shape={tuple(xs.shape[1:])}")

    # ---------------- baselines (float64, exact) ---------------- #
    X = xs.double().numpy()
    eps = 1e-6
    H = lambda p: -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))

    p_global = X.mean(axis=(0, 2, 3), keepdims=True)     # (1,C,1,1)
    base_global = float(H(p_global).sum()) * n

    p_pixel = X.mean(axis=0, keepdims=True)              # (1,C,H,W)
    base_pixel = float(H(p_pixel).sum()) * n

    # ---------------- VAE ---------------- #
    tot_rec, kl_dims, nb = 0.0, None, 0
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            x = xs[i:i + args.batch_size].to(device)
            recon, mu, logvar, _ = model(x)
            rec = F.binary_cross_entropy_with_logits(recon, x, reduction="none")
            tot_rec += float(rec.flatten(1).sum(1).sum())
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_dims = kld.sum(0) if kl_dims is None else kl_dims + kld.sum(0)
            nb += x.size(0)

    vae_rec = tot_rec
    kl_dims = (kl_dims / nb).cpu().numpy()
    npx = n * int(np.prod(xs.shape[1:]))

    print()
    print(f"{'predictor':<24}{'BCE (sum)':>13}{'per-pixel':>12}")
    print("-" * 50)
    for name, v in [("VAE (reconstruction)", vae_rec),
                    ("per-pixel mean", base_pixel),
                    ("global constant", base_global)]:
        print(f"{name:<24}{v:>13.1f}{v / npx:>12.4f}")

    print()
    ratio = vae_rec / base_pixel if base_pixel else float("nan")
    print(f"VAE / per-pixel-mean = {ratio:.4f}")
    if ratio > 0.98:
        print("  -> NO better than the mean image: the VAE has collapsed.")
    elif ratio > 0.90:
        print("  -> only marginally better than the mean image: weak latents.")
    else:
        print("  -> clearly better than the mean image: the VAE is reconstructing.")

    active = int((kl_dims > 0.1).sum())
    print(f"\nKL per latent dim: total={kl_dims.sum():.2f}  "
          f"mean={kl_dims.mean():.3f}  max={kl_dims.max():.2f}")
    print(f"active dims (KL>0.1): {active} / {len(kl_dims)}")
    print("  dims:", np.array2string(kl_dims, precision=2, max_line_width=100))
    if active < 0.5 * len(kl_dims):
        print("  -> posterior collapse: most latent dimensions carry no information")
    return 0


if __name__ == "__main__":
    sys.exit(main())
