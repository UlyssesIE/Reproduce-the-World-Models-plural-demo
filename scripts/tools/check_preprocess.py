# scripts/check_preprocess.py
"""Render raw vs. preprocessed frames side by side. Eyeball this file."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import gymnasium as gym

import sys
from pathlib import Path

# Make the project root importable regardless of how this script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # test/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))


from src.data.preprocess import preprocess_frame


def upsample(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """Nearest-upsample only for VIEWING, so we can see pixel-level artefacts."""
    im = Image.fromarray(img)
    return np.asarray(im.resize((img.shape[1] * scale, img.shape[0] * scale),
                                resample=Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/preprocess_check.png")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--crop-bottom", type=int, default=12)
    args = ap.parse_args()

    env = gym.make("CarRacing-v3", continuous=True)
    obs, _ = env.reset(seed=123)
    env.action_space.seed(123)

    raws, procs = [], []
    for _ in range(args.n):
        raws.append(obs)
        procs.append(preprocess_frame(obs, crop_bottom=args.crop_bottom))
        obs, *_ = env.step(env.action_space.sample())
    env.close()

    top = np.concatenate([upsample(r) for r in raws], axis=1)
    bot = np.concatenate([upsample(p) for p in procs], axis=1)

    # pad the shorter row (processed is 64*3=192 wide, raw is 96*3=288)
    w = max(top.shape[1], bot.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)),
                           constant_values=255)
    grid = np.concatenate([pad(top), np.full((8, w, 3), 0, np.uint8), pad(bot)], axis=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"wrote {args.out}\n  top    = raw 96x96x3 (upsampled 3x)\n"
          f"  bottom = preprocessed {bot.shape[2]}x{procs[0].shape[1]}x{procs[0].shape[2]}")


if __name__ == "__main__":
    main()
