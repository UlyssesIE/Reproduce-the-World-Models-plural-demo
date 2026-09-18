"""bench_step.py — split per-step time into dataloader vs GPU compute."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.frames import RolloutFrameDataset, split_files
from src.models.vae import VAE, VAEConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  cuda={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    train_files, _ = split_files(args.data, 0.05, 0)
    ds = RolloutFrameDataset(files=train_files, crop_bottom=12, size=64,
                             cache_files=8)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0,
                        pin_memory=(device.type == "cuda"))
    it = iter(loader)

    # ---- 1. data only -------------------------------------------------- #
    first = next(it)                       # warm the workers / page cache
    t0 = time.time()
    for _ in range(args.iters):
        next(it)
    t_data = (time.time() - t0) / args.iters
    print(f"\n[data ] {t_data*1000:8.1f} ms / batch   ({t_data/args.batch_size*1000:.2f} ms / frame)")

    # ---- 2. compute only ----------------------------------------------- #
    model = VAE(VAEConfig()).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None
    x = first.to(device, non_blocking=True)

    for _ in range(3):                     # warmup
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            recon, mu, logvar, _ = model(x)
            rec, kl = model.recon_kl(recon, x, mu, logvar)
            loss = (rec + kl).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.iters):
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            recon, mu, logvar, _ = model(x)
            rec, kl = model.recon_kl(recon, x, mu, logvar)
            loss = (rec + kl).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_comp = (time.time() - t0) / args.iters

    print(f"[comp ] {t_comp*1000:8.1f} ms / batch   (precision={args.precision})")
    print(f"[total] {(t_data+t_comp)*1000:8.1f} ms / batch")
    print(f"\nshare: data {100*t_data/(t_data+t_comp):.0f}%  "
          f"compute {100*t_comp/(t_data+t_comp):.0f}%")
    if t_comp > 0.3:
        print("!! compute alone exceeds 300 ms/batch -- the GPU is not the limit;")
        print("   check autocast/device, or run --precision fp32 for comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
