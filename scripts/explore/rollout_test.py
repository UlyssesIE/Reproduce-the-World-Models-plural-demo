import numpy as np
import sys
from pathlib import Path

# Make the project root importable regardless of how this script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # test/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))


from src.data.collect import make_env, run_rollout
from src.data.preprocess import preprocess_episode, to_float

import sys

env = make_env()
frames, actions, meta = run_rollout(env, seed=0)
env.close()

print(meta)
print("frames :", frames.shape, frames.dtype, frames.min(), frames.max())
print("actions:", actions.shape, actions.dtype,
      "\n  steering range:", actions[:, 0].min(), actions[:, 0].max(),
      "\n  gas      range:", actions[:, 1].min(), actions[:, 1].max(),
      "\n  brake    range:", actions[:, 2].min(), actions[:, 2].max())

# --- assertions: catch the silent failures now, not after 4 hours of training ---
assert frames.ndim == 4 and frames.shape[1:] == (96, 96, 3), frames.shape
assert frames.dtype == np.uint8, frames.dtype
assert actions.shape == (len(frames), 3), actions.shape
assert actions[:, 0].min() >= -1.0 and actions[:, 0].max() <= 1.0
assert actions[:, 1:].min() >= 0.0 and actions[:, 1:].max() <= 1.0
assert meta.steps > 10, "episode died almost immediately -- suspicious"
assert not np.isnan(actions).any()

# determinism: same seed must give the same episode
env = make_env(); f2, a2, m2 = run_rollout(env, seed=0); env.close()
assert m2.steps == meta.steps and np.allclose(a2, actions), "not reproducible!"
print("\nall checks passed; episode length =", meta.steps)

