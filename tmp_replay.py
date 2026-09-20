import numpy as np, sys; sys.path.insert(0, ".")
from src.data.collect import make_env
with np.load("data/raw/rollout_000000.npz") as d:
    act = d["act"].astype(np.float32); seed = int(d["seed"]); st = float(d["total_reward"])
env = make_env(1000); env.reset(seed=seed)
tot, t = 0.0, 0
while t < len(act):
    _, r, term, trunc, _ = env.step(act[t]); tot += float(r); t += 1
    if term or trunc: break
print(f"replay steps={t} return={tot:.3f} (stored {st:.3f}) d={tot-st:+.3f}")
env.close()
