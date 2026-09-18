"""Batch-collect random-policy rollouts. Safe to Ctrl+C and re-run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.collect import make_env, run_rollout


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--num-rollouts", type=int, default=10_000)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift all rollout seeds; change this for extra data")
    ap.add_argument("--max-episode-steps", type=int, default=1000)
    ap.add_argument("--compress", action="store_true",
                    help="np.savez_compressed; ~2x smaller but notably slower")
    ap.add_argument("--log-every", type=int, default=25)
    return ap.parse_args()


def rollout_path(out: Path, idx: int) -> Path:
    return out / f"rollout_{idx:06d}.npz"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(max_episode_steps=args.max_episode_steps)
    save = np.savez_compressed if args.compress else np.savez

    steps_hist, reward_hist = [], []
    t_start = time.time()
    done_count = 0

    try:
        for idx in range(args.num_rollouts):
            path = rollout_path(out, idx)

            # ---- resume: skip anything already on disk ----
            if path.exists():
                done_count += 1
                continue

            seed = args.seed_offset + idx
            frames, actions, meta = run_rollout(env, seed=seed)

            # atomic-ish write: tmp then rename, so Ctrl+C never leaves a half file
            # tmp = path.with_suffix(".npz.tmp")
            tmp = path.with_name(f"{path.stem}.tmp.npz")
            with open(tmp, "wb") as fh:
                save(fh, obs=frames, act=actions,
                    seed=np.int64(meta.seed), steps=np.int64(meta.steps),
                    total_reward=np.float32(meta.total_reward),
                    outcome=np.array(meta.outcome))
            tmp.replace(path)

            steps_hist.append(meta.steps)
            reward_hist.append(meta.total_reward)

            n = len(steps_hist)
            if n % args.log_every == 0:
                el = time.time() - t_start
                per = el / n
                remain = per * (args.num_rollouts - idx - 1)
                print(f"[{idx + 1:>6}/{args.num_rollouts}] "
                      f"steps={np.mean(steps_hist):6.1f} "
                      f"return={np.mean(reward_hist):7.1f} "
                      f"| {per:.2f}s/rollout | ETA {remain / 60:5.1f} min",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is safe, re-run the same command to resume")
    finally:
        env.close()

    manifest = {
        "requested": args.num_rollouts,
        "seed_offset": args.seed_offset,
        "max_episode_steps": args.max_episode_steps,
        "on_disk": len(list(out.glob("rollout_*.npz"))),
        "mean_steps": float(np.mean(steps_hist)) if steps_hist else None,
        "mean_return": float(np.mean(reward_hist)) if reward_hist else None,
        "raw_frame_shape": [96, 96, 3],
        "frame_dtype": "uint8",
        "policy": "uniform random (env.action_space.sample)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n" + json.dumps(manifest, indent=2))
    print(f"\nresumed/skipped: {done_count}")


if __name__ == "__main__":
    main()
