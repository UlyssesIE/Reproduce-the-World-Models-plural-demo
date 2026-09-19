"""Train the MDN-RNN (M model) on encoded latent sequences.

python scripts/train_mdnrnn.py --latents data/pilot_latents --out runs/mdn_pilot \
    --epochs 30 --batch-size 64 --seq-len 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequence import LatentSequenceDataset
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default="data/pilot_latents")
    p.add_argument("--out", default="runs/mdn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default=None)
    p.add_argument("--norm-min-std", type=float, default=0.0,
                   help="floor for the latent normalisation std; 0 = plain "
                   "normalisation (paper behaviour), 0.1 recommended")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr = LatentSequenceDataset(args.latents, args.seq_len, "train", seed=args.seed)
    va = LatentSequenceDataset(args.latents, args.seq_len, "val", seed=args.seed)
    print(f"[data] train {tr.stats()}")
    print(f"[data] val   {va.stats()}")

    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           drop_last=True, num_workers=args.workers,
                           pin_memory=(dev.type == "cuda"),
                           persistent_workers=args.workers > 0)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                           drop_last=False, num_workers=max(0, args.workers // 2),
                           pin_memory=(dev.type == "cuda"))

    cfg = MDNRNNConfig(z_dim=tr.z_dim, hidden=args.hidden,
                       n_components=args.components)
    model = MDNRNN(cfg).to(dev)
    print(f"[model] LSTM hidden={cfg.hidden} layers={cfg.num_layers} "
          f"mixtures={cfg.n_components} z_dim={cfg.z_dim} "
          f"params={model.num_params():,}")
    print(f"[model] predict_done={cfg.predict_done} tau_mode={cfg.tau_mode}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp = torch.bfloat16 if args.precision == "bf16" else None

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1; step = ck["step"]; best = ck.get("best", float("inf"))
        print(f"[resume] epoch {start_epoch}")

    log = out / "log.jsonl"
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for i, (z, a, zt) in enumerate(tr_loader):
            z, a, zt = z.to(dev), a.to(dev), zt.to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                pi, mu, ls, _, _ = model(z, a)
                nll = model.nll(pi, mu, ls, zt)          # (B,T)
                loss = nll.mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            tot += float(loss) * z.size(0); nb += z.size(0); step += 1
            if step % args.log_every == 0:
                print(f"ep {epoch:3d} step {step:6d} | nll {tot/nb:8.4f} | "
                      f"{time.time()-t0:6.0f}s", flush=True)
                with log.open("a") as fh:
                    fh.write(json.dumps({"epoch": epoch, "step": step,
                                         "split": "train", "nll": tot / nb}) + "\n")

        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for z, a, zt in va_loader:
                z, a, zt = z.to(dev), a.to(dev), zt.to(dev)
                with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                    pi, mu, ls, _, _ = model(z, a)
                    nll = model.nll(pi, mu, ls, zt)
                vt += float(nll.mean()) * z.size(0); vn += z.size(0)
        v = vt / max(vn, 1)
        print(f"  [val] ep {epoch}: nll {v:8.4f}  ({time.time()-t0:.0f}s)", flush=True)
        with log.open("a") as fh:
            fh.write(json.dumps({"epoch": epoch, "split": "val", "nll": v}) + "\n")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "step": step, "best": best, "cfg": cfg.__dict__,
              "latent_mean": tr.mean.tolist(), "latent_std": tr.std.tolist(),
              "args": vars(args)}
        torch.save(ck, out / "last.pt")
        if v < best:
            best = v; torch.save(ck, out / "best.pt")
            print(f"  [ckpt] new best val nll {best:.4f} -> best.pt")

    print(f"\ndone. best val nll {best:.4f}. checkpoints in {out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
