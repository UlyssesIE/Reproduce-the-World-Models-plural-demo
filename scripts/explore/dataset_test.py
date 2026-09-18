"""Full-dataset sanity check. Run this before training anything."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--montage", default="artifacts/dataset_montage.png")
    ap.add_argument("--n-montage", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.data)
    files = sorted(root.glob("rollout_*.npz"))
    if not files:
        raise SystemExit(f"no rollout_*.npz under {root}")
    print(f"found {len(files)} rollouts under {root}\n")

    n_frames = 0
    steps, returns, outcomes = [], [], []
    shapes, dtypes = set(), set()
    gmin, gmax = 255, 0
    sample_frames = []

    for i, f in enumerate(files):
        with np.load(f) as d:
            obs, act = d["obs"], d["act"]
        shapes.add(obs.shape[1:]); dtypes.add(str(obs.dtype))
        n_frames += obs.shape[0]
        steps.append(obs.shape[0]); returns.append(float(d["total_reward"]))
        outcomes.append(str(d["outcome"]))
        assert obs.shape[0] == act.shape[0], f"{f.name}: frames/actions length mismatch"
        assert act.shape[1] == 3, f"{f.name}: action dim {act.shape[1]}"
        assert not np.isnan(act).any(), f"{f.name}: NaN in actions"
        if obs.size:
            gmin = min(gmin, int(obs.min())); gmax = max(gmax, int(obs.max()))
        if len(sample_frames) < args.n_montage:
            sample_frames.append(obs[len(obs) // 2])

    steps = np.array(steps)

    print(f"rollouts        : {len(files)}")
    print(f"total frames    : {n_frames:,}")
    print(f"frame shapes    : {shapes}")
    print(f"frame dtypes    : {dtypes}")
    print(f"pixel range     : [{gmin}, {gmax}]  (expect [0, 255])")
    print(f"episode steps   : mean={steps.mean():.1f} std={steps.std():.1f} "
          f"min={steps.min()} max={steps.max()}")
    print(f"mean return     : {np.mean(returns):.2f}")
    print(f"terminated      : {outcomes.count('terminated')} / {len(outcomes)}")
    print(f"truncated       : {outcomes.count('truncated')} / {len(outcomes)}")
    print(f"est. disk (GB)  : {n_frames * 96 * 96 * 3 / 1e9:.1f}")

    if gmax == 0 or gmin == 255:
        print("!! WARNING: frames are constant -- collection is broken")
    if steps.mean() < 20:
        print("!! WARNING: episodes end almost immediately -- check env config")
    if outcomes.count("truncated") > 0.5 * len(outcomes):
        print("!! WARNING: most episodes hit the step limit -- max_episode_steps "
              "may be too large, or the car is not going off-track as expected")

    if sample_frames:
        scale = 3
        imgs = [np.asarray(Image.fromarray(f).resize(
            (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST))
            for f in sample_frames]
        row = np.concatenate(imgs, axis=1)
        Path(args.montage).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(row).save(args.montage)
        print(f"\nmiddle frame of {len(imgs)} rollouts -> {args.montage}")
        print("open it: should look like varied race tracks (different per rollout)")


if __name__ == "__main__":
    main()
