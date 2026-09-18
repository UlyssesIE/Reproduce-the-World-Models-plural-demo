"""Visualise VAE reconstructions: original frames (top) vs. reconstructions (bottom).

python scripts/recon_grid.py --vae runs/vae_pilot/best.pt --data data/pilot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.data.frames import RolloutFrameDataset, split_files
from src.models.vae import VAE, VAEConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--out", default="artifacts/recon_grid.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.vae, map_location=device)
    cfg = VAEConfig(**ck["cfg"])
    model = VAE(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {args.vae}  (epoch {ck.get('epoch')}, params {model.num_params():,})")

    _, val_files = split_files(args.data, val_frac=0.05, seed=args.seed)
    ds = RolloutFrameDataset(files=val_files, crop_bottom=12, size=64,
                             grayscale=(cfg.in_ch == 1))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    xs = torch.stack([ds[int(i)] for i in idx]).to(device)
    with torch.no_grad():
        rec = model.reconstruct(xs)

    xs_np = xs.permute(0, 2, 3, 1).cpu().numpy()
    rec_np = rec.permute(0, 2, 3, 1).cpu().numpy()

    def to_uint8(a, grayscale):
        a = np.clip(a, 0, 1)
        if grayscale and a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        return (a * 255).astype(np.uint8)

    def up(a):
        im = Image.fromarray(a)
        return np.asarray(im.resize((a.shape[1] * args.scale, a.shape[0] * args.scale),
                                    Image.NEAREST))

    top = np.concatenate([up(to_uint8(f, cfg.in_ch == 1)) for f in xs_np], axis=1)
    bot = np.concatenate([up(to_uint8(f, cfg.in_ch == 1)) for f in rec_np], axis=1)
    sep = np.zeros((6, top.shape[1], 3), np.uint8)
    grid = np.concatenate([top, sep, bot], axis=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)

    rec_err = float(np.abs(xs_np - rec_np).mean())
    print(f"mean |orig - recon| = {rec_err:.4f}  (lower is better)")
    print(f"wrote {args.out}")
    print("  top    = original (validation frames)")
    print("  bottom = reconstruction (decoded from mu)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
