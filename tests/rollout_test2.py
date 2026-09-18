"""measure_rollout.py — quantify how much the car actually moves."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.collect import make_env



env = make_env()
print(f"{'seed':>4} {'steps':>6} {'outcome':>10} {'tiles':>7} "
      f"{'max_dist':>9} {'final_dist':>10} {'mean_spd':>9} {'max_spd':>8}")

for seed in range(5):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    start = np.array(env.unwrapped.car.hull.position, dtype=float)

    positions, speeds, n_tiles, total_r, steps = [], [], 0, 0.0, 0
    while True:
        a = env.action_space.sample()
        _, r, term, trunc, _ = env.step(a)
        steps += 1
        total_r += r
        if r > -0.05:                                  # a tile was visited
            n_tiles += 1
        positions.append(np.array(env.unwrapped.car.hull.position, dtype=float))
        speeds.append(np.linalg.norm(env.unwrapped.car.hull.linearVelocity))
        if term or trunc:
            outcome = "terminated" if term else "truncated"
            break

    positions, speeds = np.array(positions), np.array(speeds)
    dist = np.linalg.norm(positions - start, axis=1)
    print(f"{seed:>4} {steps:>6} {outcome:>10} {n_tiles:>7} "
          f"{dist.max():>9.1f} {dist[-1]:>10.1f} "
          f"{speeds.mean():>9.2f} {speeds.max():>8.2f}")

env.close()
