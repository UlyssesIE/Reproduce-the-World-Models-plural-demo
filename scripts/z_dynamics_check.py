"""z_dynamics_check.py — is there anything in z for M to predict?

M learns z_{t+1} from (z_t, a_t, h_t). If z is nearly constant within an episode
and varies only between episodes, z is encoding "which track am I on" rather than
"where is the car on the track" -- M would then learn a static dream.

Variance of z is split into:
    within : variance over time inside a rollout, averaged over rollouts
    across : variance of the per-rollout means
    ratio  : within / (within + across)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="data/pilot_latents")
    args = ap.parse_args()

    root = Path(args.latents)
    info = json.loads((root / "index.json").read_text())
    Z = np.load(root / "latents.npy", mmap_mode="r").astype(np.float32)
    z_dim = info["z_dim"]
    print(f"latents {Z.shape}  {len(info['files'])} rollouts  z_dim={z_dim}")

    within = np.zeros(z_dim, np.float64)
    means = []
    for name, start, count in info["files"]:
        seg = np.asarray(Z[start:start + count], dtype=np.float64)
        within += seg.var(axis=0, ddof=0)
        means.append(seg.mean(axis=0))
    within /= len(info["files"])
    means = np.array(means)
    across = means.var(axis=0, ddof=0)

    total_var = within + across
    ratio = np.divide(within, total_var,
                      out=np.zeros_like(within), where=total_var > 0)

    print()
    print(f"{'dim':>4}{'std_total':>11}{'std_within':>12}{'std_across':>12}{'within%':>9}")
    for d in range(z_dim):
        st = np.sqrt(total_var[d]); sw = np.sqrt(within[d]); sa = np.sqrt(across[d])
        print(f"{d:>4}{st:>11.4f}{sw:>12.4f}{sa:>12.4f}{100*ratio[d]:>8.1f}%")

    act = total_var > 1e-6
    w = float(within[act].sum())
    t = float(total_var[act].sum())
    print(f"\naggregate over {int(act.sum())} informative dims")
    print(f"  within-rollout variance : {w:.4f}")
    print(f"  total variance          : {t:.4f}")
    print(f"  within share            : {100*w/max(t,1e-12):.1f}%")

    print()
    if act.sum() == 0:
        print("-> z is frozen: the encoder is not producing a usable code.")
    elif w / max(t, 1e-12) > 0.5:
        print("-> z varies mainly WITHIN a rollout: it encodes dynamics. "
              "M has something to predict.")
    elif w / max(t, 1e-12) > 0.15:
        print("-> mixed: z carries both track identity and within-episode change.")
    else:
        print("-> z varies mainly BETWEEN rollouts: it encodes track identity, "
              "not the car's motion. The dream will be static.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
