"""One CarRacing episode driven by C, in the REAL environment.

Step t, matching exactly how M was trained (train_mdnrnn + encode_latent):

    z_t      = VAE.encode(x_t).mu                  deterministic, no sampling
    z_norm   = (z_t - mean) / std                  mean/std from M's checkpoint
    a_t      = C(z_norm, h_t)                      tanh + affine, clipped
    x_{t+1}  = env.step(a_t)
    h_{t+1}  = M(z_norm, a_t, h_t)[-1]             Per-episode h starts at zeros
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.collect import make_env
from src.data.preprocess import preprocess_frame, to_float


def build_env(max_episode_steps: int = 1000):
    """Same factory the training data was collected with (gymnasium v3)."""
    return make_env(max_episode_steps=max_episode_steps)


class Agent:
    """Bundles V (encode), M (hidden state) and M's normalisation stats."""

    def __init__(self, vae, mdn, ctrl, mean, std, device="cpu",
                 crop_bottom: int = 12, size: int = 64, grayscale: bool = False):
        self.vae, self.mdn, self.ctrl = vae, mdn, ctrl
        self.dev = torch.device(device)
        self.mean = np.asarray(mean, np.float32).reshape(-1)
        self.std = np.asarray(std, np.float32).reshape(-1)
        self.crop_bottom, self.size, self.grayscale = crop_bottom, size, grayscale
        lo = self.ctrl.low.detach().cpu().numpy()
        hi = self.ctrl.high.detach().cpu().numpy()
        self.norm_stats_source = None
        # these two are what the env loop needs; kept as numpy for speed
        self._lo, self._hi = lo, hi

    # ---------------- V ---------------- #
    def encode(self, obs) -> np.ndarray:
        img = preprocess_frame(obs, crop_bottom=self.crop_bottom,
                               size=self.size, grayscale=self.grayscale)
        x = to_float(img)                                     # (H,W,C) [0,1]
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        with torch.no_grad():
            mu, _ = self.vae.encode(t.unsqueeze(0).to(self.dev))
        z = mu[0].float().cpu().numpy()
        return (z - self.mean) / np.maximum(self.std, 1e-8)

    # ---------------- M ---------------- #
    def init_hidden(self):
        return self.mdn.init_hidden(1, self.dev)

    def hidden_step(self, h, z_norm, a):
        zt = torch.as_tensor(z_norm, dtype=torch.float32, device=self.dev).view(1, 1, -1)
        at = torch.as_tensor(np.asarray(a, np.float32), device=self.dev).view(1, 1, -1)
        with torch.no_grad():
            _, _, _, _, h_new = self.mdn(zt, at, h)
        return h_new

    @staticmethod
    def h_np(h) -> np.ndarray:
        return h[0].reshape(-1).detach().cpu().numpy()        # output vector only

    # ---------------- C ---------------- #
    def act(self, z_norm, h) -> np.ndarray:
        zt = torch.as_tensor(z_norm, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            if self.ctrl.use_hidden:
                ht = torch.as_tensor(self.h_np(h), device=self.dev)
                y = self.ctrl.act(zt, ht)
            else:
                y = self.ctrl.act(zt)
        return y.detach().cpu().numpy().astype(np.float32)


def reset_env(env, seed):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def run_episode(env, agent, seed=None, max_steps: int = 1000) -> dict:
    obs = reset_env(env, seed)
    z = agent.encode(obs)
    h = agent.init_hidden()
    total, steps = 0.0, 0
    while steps < max_steps:
        a = agent.act(z, h)
        obs, r, terminated, truncated, _ = env.step(a)
        steps += 1
        total += float(r)
        h = agent.hidden_step(h, z, a)
        z = agent.encode(obs)
        if terminated or truncated:
            break
    return {"return": total, "steps": steps, "seed": seed}
