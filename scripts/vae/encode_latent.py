"""encode_latents.py — freeze the VAE, encode every frame, store latent sequences.

M is trained on (z_t, a_t, z_{t+1}) with rollout boundaries respected, so the
output keeps the same per-rollout segmentation as the frame cache.

The encoder's posterior mean mu is stored, not a sample: sampling would inject
noise that M is not responsible for modelling.

Outputs under --out:
    latents.npy   (N, z_dim) float16
    actions.npy   (N, 3)     float32
    index.json    per-rollout (name, start, count) + latent mean/std
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.frames import FlatFrameDataset
from src.models.vae import VAE, VAEConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/pilot_frames.npy")
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", default="data/pilot_latents")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--crop-bottom", type=int, default=12)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}")

    ck = torch.load(args.vae, map_location=dev, weights_only=False)
    cfg = VAEConfig(**ck["cfg"])
    vae = VAE(cfg).to(dev).eval()
    vae.load_state_dict(ck["model"])
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[vae]    {args.vae}  epoch={ck.get('epoch')} z_dim={cfg.z_dim} "
          f"params={vae.num_params():,}")

    files = sorted(Path(args.data).glob("rollout_*.npz"))
    if not files:
        raise SystemExit(f"no rollout_*.npz under {args.data}")

    index_path = Path(args.cache).with_suffix(".index.json")
    index = json.loads(index_path.read_text())
    by_name = {n: (s, c) for n, s, c in index["files"]}

    segments, action_list = [], []
    for f in files:
        if f.name not in by_name:
            raise KeyError(f"{f.name} missing from {index_path}")
        s, c = by_name[f.name]
        segments.append([f.name, s, c])
        with np.load(f) as d:
            a = d["act"].astype(np.float32)
        assert a.shape[0] == c, f"{f.name}: {a.shape[0]} actions vs {c} frames"
        action_list.append(a)

    total = int(sum(s[2] for s in segments))
    acts = np.concatenate(action_list, axis=0)
    print(f"[data]   {len(files)} rollouts, {total:,} frames, actions {acts.shape}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lat_dtype = np.float16 if args.dtype == "float16" else np.float32
    lat = np.lib.format.open_memmap(out / "latents.npy", mode="w+",
                                    dtype=lat_dtype, shape=(total, cfg.z_dim))

    ds = FlatFrameDataset(args.cache, files, crop_bottom=args.crop_bottom,
                          size=args.size, grayscale=(cfg.in_ch == 1))
    if len(ds) != total:
        raise SystemExit(f"cache holds {len(ds)} frames, index says {total}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.workers,
                        pin_memory=(dev.type == "cuda"),
                        persistent_workers=args.workers > 0)

    run_sum = np.zeros(cfg.z_dim, np.float64)
    run_sqsum = np.zeros(cfg.z_dim, np.float64)
    i, t0 = 0, time.time()
    with torch.no_grad():
        for k, x in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            mu, _ = vae.encode(x)
            m = mu.float().cpu().numpy()
            lat[i:i + m.shape[0]] = m
            run_sum += m.sum(0, dtype=np.float64)
            run_sqsum += (m.astype(np.float64) ** 2).sum(0)
            i += m.shape[0]
            if (k + 1) % args.log_every == 0:
                print(f"  {i:,}/{total:,} frames  ({time.time() - t0:.0f}s)", flush=True)
    lat.flush()
    if i != total:
        raise SystemExit(f"encoded {i} but expected {total}")

    mean = run_sum / total
    std = np.sqrt(np.maximum(run_sqsum / total - mean ** 2, 0.0))
    np.save(out / "actions.npy", acts)

    info = {"files": segments, "z_dim": cfg.z_dim, "total": total,
            "mean": mean.tolist(), "std": std.tolist(),
            "vae": str(args.vae), "epoch": ck.get("epoch"),
            "crop_bottom": args.crop_bottom, "size": args.size,
            "grayscale": bool(cfg.in_ch == 1), "dtype": args.dtype}
    (out / "index.json").write_text(json.dumps(info, indent=1))

    gb = total * cfg.z_dim * np.dtype(lat_dtype).itemsize / 1e9
    print(f"\nwrote {out}/latents.npy  ({total}, {cfg.z_dim}) {np.dtype(lat_dtype).name}, {gb:.3f} GB")
    print(f"      {out}/actions.npy  {acts.shape}")
    print(f"      {out}/index.json")
    print(f"  per-dim std: min={std.min():.3f} max={std.max():.3f} mean={std.mean():.3f}")
    print(f"  near-constant dims (std<0.05): {int((std < 0.05).sum())}/{len(std)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
