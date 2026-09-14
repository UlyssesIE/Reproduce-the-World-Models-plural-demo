"""Random-policy rollout collection for CarRacing."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym


@dataclass
class RolloutMeta:
    seed: int
    steps: int
    total_reward: float
    outcome: str            # "terminated" | "truncated"


def make_env(seed: int | None = None, max_episode_steps: int = 1000):
    """Continuous-3-action CarRacing, matching the paper's action space."""
    return gym.make(
        "CarRacing-v3",
        continuous=True,             # Box([-1,0,0], [1,1,1]) -> steering/gas/brake
        lap_complete_percent=0.95,   # align with v0 lap semantics
        max_episode_steps=max_episode_steps,
    )


def run_rollout(env, seed: int) -> tuple[np.ndarray, np.ndarray, RolloutMeta]:
    """One random-policy episode.

    Returns
    -------
    frames  : (T, 96, 96, 3) uint8   -- RAW frames, exactly as the env emits them
    actions : (T, 3) float32         -- a_t paired with frames[t]
    meta    : RolloutMeta

    Note: T frames and T actions are stored; the terminal observation returned by
    the final step() is deliberately dropped, so training pairs (z_t, a_t, z_{t+1})
    can be formed for t = 0 .. T-2.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)          # explicit seeding -> reproducible policy

    frames, actions = [], []
    total_reward = 0.0
    outcome = "truncated"

    while True:
        action = env.action_space.sample().astype(np.float32)

        frames.append(np.asarray(obs, dtype=np.uint8))   # x_t
        actions.append(action)                           # a_t

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated:
            outcome = "terminated"
            break
        if truncated:
            outcome = "truncated"
            break

    meta = RolloutMeta(seed=seed, steps=len(frames),
                       total_reward=total_reward, outcome=outcome)
    return (np.stack(frames), np.stack(actions), meta)
