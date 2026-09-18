"""probe test.py — compare random action distributions for data collection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.collect import make_env

DT = 1.0 / 50.0        # CarRacing physics timestep

def uniform(rng):
    return np.array([rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(0, 1)], np.float32)

def full_gas(rng):
    """Control: full gas, no brake. Upper bound on achievable speed."""
    return np.array([0.0, 1.0, 0.0], np.float32)

def signed_throttle(rng, lo=-0.5, hi=1.0):
    """Gas and brake derive from ONE signed value -> they can never both be positive."""
    t = rng.uniform(lo, hi)
    return np.array([rng.uniform(-1, 1), max(t, 0.0), max(-t, 0.0)], np.float32)

STRATEGIES = {
    "uniform":         uniform,
    "full_gas":        full_gas,
    "signed(-0.5,1)":  lambda rng: signed_throttle(rng, -0.5, 1.0),
    "signed(-0.2,1)":  lambda rng: signed_throttle(rng, -0.2, 1.0),
}

def probe(env, policy, rng, seed, max_steps=1000):
    env.reset(seed=seed)
    start = np.array(env.unwrapped.car.hull.position, dtype=float)
    pos, spd, tiles, done = [], [], 0, "truncated"
    while True:
        _, r, term, trunc, _ = env.step(policy(rng))
        if r > -0.05:
            tiles += 1
        pos.append([env.unwrapped.car.hull.position.x, env.unwrapped.car.hull.position.y])
        spd.append(np.linalg.norm(env.unwrapped.car.hull.linearVelocity))
        if term or trunc:
            done = "terminated" if term else "truncated"
            break
    pos, spd = np.array(pos), np.array(spd)
    path = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    return dict(steps=len(spd), outcome=done, tiles=tiles,
                mean_spd=spd.mean(), max_spd=spd.max(),
                path=path, net=np.linalg.norm(pos[-1] - pos[0]),
                net_over_path=np.linalg.norm(pos[-1] - pos[0]) / max(path, 1e-9),
                speed_per_step=spd.mean() * DT)

env = make_env()
rng = np.random.default_rng(0)
print(f"{'strategy':>16} {'steps':>6} {'outcome':>10} {'tiles':>6} "
      f"{'mean_spd':>9} {'max_spd':>8} {'net_dist':>9} {'net/path':>9}")
for name, pol in STRATEGIES.items():
    rows = [probe(env, pol, rng, seed=s) for s in range(5)]
    agg = lambda k: np.mean([r[k] for r in rows])
    print(f"{name:>16} {agg('steps'):>6.0f} {rows[0]['outcome']:>10} "
          f"{agg('tiles'):>6.1f} {agg('mean_spd'):>9.2f} {agg('max_spd'):>8.2f} "
          f"{agg('net'):>9.1f} {agg('net_over_path'):>9.2f}")
env.close()
