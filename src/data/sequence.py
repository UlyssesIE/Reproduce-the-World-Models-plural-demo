"""Sequence dataset over encoded latents for MDN-RNN training.

Windows never cross a rollout boundary: M must not be asked to predict the
first latent of one episode from the last latent of another.

Latents are normalised on the fly with the mean/std stored in index.json --
the paper normalises z before feeding it to M, and without it the mixture
Gaussians have to cope with very different per-dimension scales.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    def __init__(self, latents_dir, seq_len: int = 32, mode: str = "train",
                 train_frac: float = 0.95, seed: int = 0, normalize: bool = True):
        root = Path(latents_dir)
        info = json.loads((root / "index.json").read_text())
        self.files = info["files"]
        self.z_dim = int(info["z_dim"])
        self.seq_len = int(seq_len)
        self.normalize = normalize

        self.Z = np.load(root / "latents.npy", mmap_mode="r")
        self.A = np.load(root / "actions.npy", mmap_mode="r")

        mean = np.asarray(info["mean"], np.float32)
        std = np.asarray(info["std"], np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        self.mean, self.std = mean, std

        # split by rollout, not by window
        n = len(self.files)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(round(n * train_frac))
        sel = perm[:n_train] if mode == "train" else perm[n_train:]
        rollout_ids = sorted(sel.tolist())

        # valid window starts: need seq_len+1 consecutive frames in one rollout
        stride = 1 if mode == "train" else max(1, self.seq_len)
        starts = []
        for r in rollout_ids:
            _, s0, cnt = self.files[r]
            last = cnt - (self.seq_len + 1)
            if last < 0:
                continue
            starts.extend(range(s0, s0 + last + 1, stride))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.rollouts = rollout_ids
        if len(self.starts) == 0:
            raise RuntimeError("no valid windows; is seq_len too large?")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        t = self.seq_len
        s = int(self.starts[i])
        z = np.asarray(self.Z[s:s + t + 1], dtype=np.float32)     # (t+1, D)
        a = np.asarray(self.A[s:s + t], dtype=np.float32)         # (t, A)
        if self.normalize:
            z = (z - self.mean) / self.std
        return (torch.from_numpy(z[:t]),      # z_t
                torch.from_numpy(a),          # a_t
                torch.from_numpy(z[1:]))      # z_{t+1}

    def stats(self) -> dict:
        return {"rollouts": len(self.rollouts), "windows": len(self.starts),
                "seq_len": self.seq_len, "z_dim": self.z_dim,
                "normalize": self.normalize}
