"""Synthetic latents in which the action GENUINELY drives the step (positive control).

    z_{t+1} = z_t + rho*(mean - z_t) + c * (a_t @ V) + eps

Same file layout as the real dataset; actions copied verbatim; generated in the REAL
per-dim scale so the metric's denominator E|z_t+1 - z_t| is comparable.

    python scripts/mdn/make_synthetic_latents.py --src data/pilot_latents \
        --dst data/pilot_latents_synth --drive-frac 0.7 --noise-frac 0.3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/pilot_latents")
ap.add_argument("--dst", default="data/pilot_latents_synth")
ap.add_argument("--drive-frac", type=float, default=0.7,
                help="share of the real mean|dz| to be carried by the action term")
ap.add_argument("--noise-frac", type=float, default=0.3)
ap.add_argument("--rho", type=float, default=0.9, help="mean reversion")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
info = json.loads((src / "index.json").read_text())
mean = np.asarray(info["mean"], np.float64)
std = np.asarray(info["std"], np.float64)
rng = np.random.default_rng(a.seed)

A = np.load(src / "actions.npy")                       # (N, 3) verbatim
L = int(info["total"]); D = int(info["z_dim"]); Adim = A.shape[1]
Z = np.load(src / "latents.npy", mmap_mode="r")
real_step = float(np.abs(np.diff(np.asarray(Z[:2000], np.float32), axis=0)).mean())

# fixed random 3 -> D map, scaled per dimension to the real latents' scale
V = rng.standard_normal((Adim, D)) * std[None, :]

# choose c so the action term carries drive-frac of the real mean step size
unit = A[:2000] @ V
c = a.drive_frac * real_step / max(float(np.abs(unit).mean()), 1e-12)

Zout = np.zeros((L, D), dtype=np.float64)
comp = {"drive": 0.0, "noise": 0.0, "drift": 0.0}
i = 0
for _name, s0, cnt in info["files"]:
    z = mean + std * rng.standard_normal(D)
    Zout[i] = z; i += 1
    for t in range(s0, s0 + cnt - 1):
        drift = a.rho * (mean - z)
        drive = c * (A[t] @ V)
        eps = std * a.noise_frac * rng.standard_normal(D)
        comp["drift"] += np.abs(drift).mean(); comp["drive"] += np.abs(drive).mean()
        comp["noise"] += np.abs(eps).mean()
        z = z + drift + drive + eps
        Zout[i] = z; i += 1
nn = max(i, 1)
np.save(dst / "latents.npy", Zout.astype(np.float16))
np.save(dst / "actions.npy", A)
shutil.copy2(src / "index.json", dst / "index.json")

step = float(np.abs(np.diff(Zout[:2000], axis=0)).mean())
print(f"[synth] {dst}  latents {Zout.shape} float16  actions {A.shape} (verbatim)")
print(f"[synth] mean|dz|  real {real_step:.5f}  synth {step:.5f}  (ratio {step/real_step:.2f})")
print(f"[synth] per-step mean|component|  drive {comp['drive']/nn:.5f}  "
      f"noise {comp['noise']/nn:.5f}  drift {comp['drift']/nn:.5f}")
print(f"[synth] c = {c:.6g}   rho = {a.rho}   index.json copied verbatim")
