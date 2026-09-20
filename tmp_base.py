import glob, numpy as np
fs = sorted(glob.glob("data/raw/rollout_*.npz"))
for tag, sel in (("first100", fs[:100]), ("all2000", fs)):
    r = np.array([float(np.load(f)["total_reward"]) for f in sel])
    s = np.array([int(np.load(f)["steps"]) for f in sel])
    print(f"[{tag}] n={len(r)} return {r.mean():.2f}+/-{r.std():.2f} steps {s.mean():.0f} max {s.max()}")
