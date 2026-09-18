"""diagnose_diversity.py — is the collected data meaningfully diverse?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

def run(data_dir: str, n_files: int = 40, n_pairs: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    files = sorted(Path(data_dir).glob("rollout_*.npz"))
    if not files:
        raise SystemExit(f"no rollouts under {data_dir}")
    files = files[:: max(1, len(files) // n_files)][:n_files]

    consec, distant, speeds, tiles = [], [], [], []
    all_frames = []

    for f in files:
        with np.load(f) as d:
            obs, act = d["obs"], d["act"]
        frames = obs[::5].astype(np.float32)          # subsample for speed
        all_frames.append(frames)
        # consecutive-frame difference (how much changes step to step)
        diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
        consec.append(diff.mean())
        # speed proxy: L2 of steering/action changes is not speed; use frame diff only
        tiles.append(int((np.abs(act[:, 1] - act[:, 2]) < 0.5).sum()))  # rough

    pool = np.concatenate([f[:200] for f in all_frames])   # (N, 64?, 96, 96, 3)
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(pool), 2)
        if i != j:
            distant.append(np.abs(pool[i] - pool[j]).mean())

    print(f"rollouts sampled        : {len(files)}")
    print(f"mean |frame_t - frame_t+1| : {np.mean(consec):8.2f}   (consecutive)")
    print(f"mean |frame_i - frame_j|   : {np.mean(distant):8.2f}   (random pair)")
    ratio = np.mean(consec) / max(np.mean(distant), 1e-9)
    print(f"ratio consec/random     : {ratio:8.3f}")
    if ratio > 0.5:
        print("  -> frames change almost as much as unrelated frames: GOOD diversity")
    elif ratio > 0.15:
        print("  -> moderate redundancy")
    else:
        print("  -> HIGHLY redundant: consecutive frames nearly identical")

    # pixel-level variance across the pool (a second view of the same thing)
    print(f"pool pixel std          : {pool.std():8.2f}  (uniform noise would be ~74)")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
