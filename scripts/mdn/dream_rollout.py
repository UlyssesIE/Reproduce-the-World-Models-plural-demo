"""Autoregressive dream: run M on its own predictions and decode with V.

Two uses:

  1. Acceptance test for V and M jointly (Note2 §8): if the decoded sequence
     shows a coherent track, the world model is usable.
  2. Paper §3.4: put the evolved controller C *back into* the hallucinated
     environment.  Pass --controller with a best_theta.npz.

Top row = a real rollout. Bottom row = the dream, seeded with the same z_0.

python scripts/mdn/dream_rollout.py --vae runs/vae_pilot2/best.pt \
    --mdn runs/mdn_pilot/best.pt --latents data/pilot_latents \
    --steps 300 --tau 1.15 --out runs/dream/real.png

python scripts/mdn/dream_rollout.py --vae runs/vae_pilot2/best.pt \
    --mdn runs/mdn_pilot/best.pt --latents data/pilot_latents \
    --controller runs/controller_867/best_theta.npz \
    --steps 300 --tau 1.15 --out runs/dream/with_ctrl.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from src.data.frames import FlatFrameDataset          # noqa: F401  (kept: used by older callers)
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig
from src.models.vae import VAE, VAEConfig
# >>> CHANGE 1/4: controller import
from src.models.controller import Controller


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--mdn", required=True)
    ap.add_argument("--latents", default="data/pilot_latents")
    ap.add_argument("--cache", default="data/pilot_frames.npy")
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--rollout-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--action", choices=["repeat", "random", "real"], default="real")
    # >>> CHANGE 2/4: --controller overrides --action
    ap.add_argument("--controller", default=None,
                    help="best_theta.npz from scripts/controller/train_controller.py; "
                         "C drives the dream (paper §3.4). Overrides --action.")
    ap.add_argument("--out", default="artifacts/dream.png")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ---- V ---- #
    vck = torch.load(args.vae, map_location=dev, weights_only=False)
    vcfg = VAEConfig(**vck["cfg"])
    vae = VAE(vcfg).to(dev).eval(); vae.load_state_dict(vck["model"])

    # ---- M ---- #
    mck = torch.load(args.mdn, map_location=dev, weights_only=False)
    mcfg = MDNRNNConfig(**mck["cfg"])
    m = MDNRNN(mcfg).to(dev).eval(); m.load_state_dict(mck["model"])
    mean = np.asarray(mck["latent_mean"], np.float32)
    std = np.asarray(mck["latent_std"], np.float32)

    info = json.loads((Path(args.latents) / "index.json").read_text())
    Z = np.load(Path(args.latents) / "latents.npy", mmap_mode="r")
    A = np.load(Path(args.latents) / "actions.npy", mmap_mode="r")

    name, s0, cnt = info["files"][args.rollout_index]
    T = min(args.steps, cnt - 1)
    z_real = np.asarray(Z[s0:s0 + T + 1], np.float32)       # (T+1, D)
    a_real = np.asarray(A[s0:s0 + T], np.float32)           # (T, A)
    print(f"[data] rollout {name}  frames={cnt}  using T={T}")
    print(f"[dream] tau={args.tau} action={args.action} tau_mode={mcfg.tau_mode}")

    # >>> CHANGE 3/4: build C from the saved theta, inferring the arm from its size
    ctrl = None
    if args.controller:
        d = np.load(args.controller, allow_pickle=True)
        theta = np.asarray(d["theta"], np.float64).reshape(-1)
        z_dim, h_dim, a_dim = int(mcfg.z_dim), int(mcfg.hidden), int(mcfg.action_dim)

        n_full = (z_dim + h_dim) * a_dim + a_dim      # 867
        n_vonly = z_dim * a_dim + a_dim               # 99
        if theta.size == n_full:
            use_hidden = True
        elif theta.size == n_vonly:
            use_hidden = False
        else:
            raise SystemExit(
                f"{args.controller}: theta has {theta.size} params, but M implies "
                f"{n_full} (V+h) or {n_vonly} (V-only). Wrong checkpoint for this M?")

        ctrl = Controller(z_dim=z_dim, hidden=h_dim, action_dim=a_dim,
                          use_hidden=use_hidden).to(dev).eval()
        assert ctrl.n_params == theta.size, (ctrl.n_params, theta.size)
        ctrl.set_flat(theta)
        lo = ctrl.low.detach().cpu().numpy()
        hi = ctrl.high.detach().cpu().numpy()
        print(f"[ctrl] {args.controller}  params={theta.size} "
              f"use_hidden={use_hidden}  bounds lo={lo} hi={hi}")
        if args.action != "real":
            print("[ctrl] --controller given: --action is ignored for the dream")

    # ---- actions for the dream (only needed when C is absent) ---- #
    acts = None
    if ctrl is None:
        if args.action == "random":
            acts = np.random.default_rng(args.seed).uniform(
                [-1, 0, 0], [1, 1, 1], size=(T, mcfg.action_dim)).astype(np.float32)
        elif args.action == "repeat":
            acts = np.repeat(a_real[:1], T, axis=0)
        else:
            acts = a_real

    # ---- autoregressive rollout ---- #
    zn = (z_real - mean) / std                              # (T+1, D)
    h = m.init_hidden(1, dev)                               # reset per rollout
    z_cur = torch.from_numpy(zn[0:1]).to(dev).unsqueeze(0)  # (1,1,D)
    zs = [z_cur.squeeze(0).squeeze(0).cpu().numpy()]
    a_used = []
    with torch.no_grad():
        for t in range(T):
            # >>> CHANGE 4/4: a_t from C(z_norm, h_t) -- same order as src/rollout.py
            if ctrl is not None:
                z_np = z_cur.squeeze(0).squeeze(0)                 # (D,)
                if ctrl.use_hidden:
                    h_np = h[0].reshape(-1)                        # (H,) output vector
                    a_np = ctrl.act(z_np, h_np)
                else:
                    a_np = ctrl.act(z_np)
                a_np = a_np.detach().cpu().numpy().reshape(-1).astype(np.float32)
            else:
                a_np = np.asarray(acts[t], np.float32).reshape(-1)

            a_t = torch.from_numpy(a_np).to(dev).view(1, 1, -1)     # (1,1,A)
            pi, mu, ls, _, h = m(z_cur, a_t, h)                     # -> h_{t+1}
            z_next = m.sample(pi[:, -1], mu[:, -1], ls[:, -1],
                              temperature=args.tau)                 # (1,D)
            zs.append(z_next.squeeze(0).cpu().numpy())
            z_cur = z_next.unsqueeze(1)
            a_used.append(a_np)

    z_dream = np.stack(zs) * std + mean                     # (T+1, D), un-normalised
    a_used = np.stack(a_used)                               # (T, A)

    if ctrl is not None:
        print("[ctrl] action stats (dream):")
        for j, nm in enumerate(("steer", "gas", "brake")[:a_used.shape[1]]):
            print(f"    {nm:>5}: min {a_used[:, j].min():+.3f}  "
                  f"max {a_used[:, j].max():+.3f}  mean {a_used[:, j].mean():+.3f}")

    # ---- decode both ---- #
    def decode(z_np):
        out = []
        with torch.no_grad():
            for i in range(0, len(z_np), 256):
                zb = torch.from_numpy(z_np[i:i + 256]).to(dev)
                img = vae.decode(zb)
                img = torch.sigmoid(img) if vcfg.recon == "bce" else img
                out.append(img.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    rec_real, rec_dream = decode(z_real), decode(z_dream)

    def strip(imgs, every=8):
        sel = imgs[::every]
        return np.concatenate([np.repeat(np.repeat(f, args.scale, 0),
                                         args.scale, 1) for f in sel], axis=1)

    top, bot = strip(rec_real), strip(rec_dream)
    sep = np.zeros((6, top.shape[1], 3), np.float32)
    grid = np.concatenate([top, sep, bot], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).save(args.out)

    # ---- numeric summary ---- #
    d_real = np.abs(np.diff(rec_real, axis=0)).mean()
    d_dream = np.abs(np.diff(rec_dream, axis=0)).mean()
    dz_real = np.abs(np.diff(z_real, axis=0)).mean()
    dz_dream = np.abs(np.diff(z_dream, axis=0)).mean()
    print(f"\nframe-to-frame change, real  : {d_real:.4f}")
    print(f"frame-to-frame change, dream : {d_dream:.4f} "
          f"(ratio to real {d_dream / max(d_real, 1e-9):.3f})")
    print(f"latent |dz|, real {dz_real:.4f}  dream {dz_dream:.4f} "
          f"(ratio {dz_dream / max(dz_real, 1e-9):.3f})")
    print(f"wrote {args.out}   (top = real, bottom = dream)")
    print("  image ratio >> 1 : dream is noisier than reality (tau too high?)")
    print("  image ratio << 1 : dream is frozen (M learned a static transition)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
