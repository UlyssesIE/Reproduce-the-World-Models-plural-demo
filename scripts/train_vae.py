"""Train the VAE (V model) on collected CarRacing frames.

Example
-------
python scripts/train_vae.py --data data/pilot --out runs/vae_pilot \
    --epochs 30 --batch-size 128 --beta 1.0
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

from src.data.frames import RolloutFrameDataset, split_files
from src.models.vae import VAE, VAEConfig

import json

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--crop-bottom", type=int, default=12)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--max-frames", type=int, default=0,
                   help="subsample this many training frames (0 = all)")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    # VAE specifics
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--recon", choices=["bce", "mse"], default="bce")
    p.add_argument("--recon-reduction", choices=["sum", "mean"], default="sum")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--free-bits", type=float, default=0.0)
    p.add_argument("--beta-warmup-steps", type=int, default=0,
                   help="linearly ramp beta from 0 over this many steps")
    # misc
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every-epochs", type=int, default=5)
    p.add_argument("--resume", default=None, help="path to checkpoint .pt")
    p.add_argument("--frame-cache", default=None,
                   help="path to a flat .npy built by build_frame_cache.py")
    return p.parse_args()


def beta_at(step: int, target: float, warmup: int) -> float:
    if warmup <= 0:
        return target
    return target * min(1.0, step / float(warmup))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda} | available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")
    else:
        print("[device] !! running on CPU -- this is ~20x slower than expected")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ---------------- #
    train_files, val_files = split_files(args.data, args.val_frac, args.seed)
    ds_kw = dict(crop_bottom=args.crop_bottom, size=args.size,
                 grayscale=args.grayscale)
    train_ds = RolloutFrameDataset(files=train_files, **ds_kw)
    val_ds = RolloutFrameDataset(files=val_files, **ds_kw)

    if args.max_frames and args.max_frames < len(train_ds):
        # keep it simple: subset by taking the first N global indices,
        # which corresponds to whole rollouts (no partial-file leakage concerns)
        sub = torch.utils.data.Subset(train_ds, list(range(args.max_frames)))
        train_ds = sub

    else:
        train_ds = RolloutFrameDataset(files=train_files, **ds_kw)
        val_ds = RolloutFrameDataset(files=val_files, **ds_kw)

##################################################################################
    if args.frame_cache:
        from src.data.frames import FlatFrameDataset
        train_ds = FlatFrameDataset(args.frame_cache, train_files,
                                    crop_bottom=args.crop_bottom,
                                    size=args.size, grayscale=args.grayscale)
        val_ds = FlatFrameDataset(args.frame_cache, val_files,
                                  crop_bottom=args.crop_bottom,
                                  size=args.size, grayscale=args.grayscale)
        print(f"[data] flat frame cache: {args.frame_cache}")
###################################################################################
    n_val_workers = max(0, args.workers // 2)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=n_val_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=n_val_workers > 0,
    )
##############################################################################
    # train_loader = DataLoader(
    #     train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    #     num_workers=args.workers, pin_memory=(device.type == "cuda"),
    #     persistent_workers=args.workers > 0,
    # )
    # val_loader = DataLoader(
    #     val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
    #     num_workers=max(1, args.workers // 2), pin_memory=(device.type == "cuda"),
    #     persistent_workers=args.workers > 0,
    # )

    print(f"[data] train frames={len(train_ds)}  val frames={len(val_ds)}")
    print(f"[data] preprocessing: crop_bottom={args.crop_bottom} size={args.size} "
          f"grayscale={args.grayscale}")

    # ---------------- model ---------------- #
    cfg = VAEConfig(
        z_dim=args.z_dim,
        in_ch=1 if args.grayscale else 3,
        recon=args.recon,
        beta=args.beta,
        free_bits=args.free_bits,
        recon_reduction=args.recon_reduction,
    )
    model = VAE(cfg).to(device)

    pb = model.param_breakdown()
    print(f"[model] params: encoder={pb['encoder']:,} decoder={pb['decoder']:,} "
          f"total={pb['total']:,}")
    print(f"[model] paper reports 4,348,547 for CarRacing -- "
          f"delta={pb['total'] - 4_348_547:+,} (architecture underspecified)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"[resume] from {args.resume} (epoch {start_epoch})")

    # ---------------- training ---------------- #
    log_path = out / "log.jsonl"
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        agg = {"total": 0.0, "rec": 0.0, "kl": 0.0, "n": 0}

        for i, x in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            beta = beta_at(global_step, args.beta, args.beta_warmup_steps)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                recon, mu, logvar, _ = model(x)
                rec, kl = model.recon_kl(recon, x, mu, logvar)
                loss = (rec + beta * kl).mean() / args.accum

            loss.backward()

            if (i + 1) % args.accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            bs = x.size(0)
            agg["total"] += float(loss) * args.accum * bs
            agg["rec"] += float(rec.mean()) * bs
            agg["kl"] += float(kl.mean()) * bs
            agg["n"] += bs
            global_step += 1

            if global_step % args.log_every == 0:
                n = max(agg["n"], 1)
                print(f"ep {epoch:3d} step {global_step:6d} | "
                      f"loss {agg['total']/n:9.1f} rec {agg['rec']/n:9.1f} "
                      f"kl {agg['kl']/n:7.2f} beta {beta:.3f} | "
                      f"{time.time()-t0:6.0f}s", flush=True)
                with log_path.open("a") as fh:
                    fh.write(json.dumps({
                        "epoch": epoch, "step": global_step, "split": "train",
                        "loss": agg["total"] / n, "rec": agg["rec"] / n,
                        "kl": agg["kl"] / n, "beta": beta,
                    }) + "\n")

        # ---------------- validation ---------------- #
        model.eval()
        v = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "n": 0}
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    recon, mu, logvar, _ = model(x)
                    rec, kl = model.recon_kl(recon, x, mu, logvar)
                    loss = (rec + args.beta * kl).mean()
                bs = x.size(0)
                v["loss"] += float(loss) * bs
                v["rec"] += float(rec.mean()) * bs
                v["kl"] += float(kl.mean()) * bs
                v["n"] += bs
        n = max(v["n"], 1)
        val_loss, val_rec, val_kl = v["loss"] / n, v["rec"] / n, v["kl"] / n
        print(f"  [val] ep {epoch}: loss {val_loss:9.1f} rec {val_rec:9.1f} "
              f"kl {val_kl:7.2f}  ({time.time()-t0:.0f}s)", flush=True)

        with log_path.open("a") as fh:
            fh.write(json.dumps({"epoch": epoch, "split": "val", "loss": val_loss,
                                 "rec": val_rec, "kl": val_kl}) + "\n")

        # ---------------- checkpoints ---------------- #
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "global_step": global_step, "best_val": best_val,
                "cfg": cfg.__dict__, "args": vars(args)}
        torch.save(ckpt, out / "last.pt")
        if (epoch + 1) % args.ckpt_every_epochs == 0:
            torch.save(ckpt, out / f"epoch{epoch:03d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out / "best.pt")
            print(f"  [ckpt] new best val loss {best_val:.1f} -> best.pt")

    print(f"\ndone. best val loss {best_val:.1f}. checkpoints in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
