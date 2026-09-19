"""Sequence dataset over encoded latents for MDN-RNN training.

Windows never cross a rollout boundary: M must not be asked to predict the
first latent of one episode from the last latent of another.

Normalisation
-------------
z is normalised with the mean/std computed over all frames (stored in
index.json), because the mixture Gaussians otherwise have to cope with
per-dimension scales spanning two orders of magnitude.

That plain division has a side effect: 26 of the 32 latent dims have raw std
0.015-0.044, so dividing by their own std amplifies them 23-70x, turning them
into amplified noise. Measured on the first trained M, those dims absorbed 75%
of the NLL. `norm_min_std` floors the *divisor* so each dim's normalised scale
stays proportional to its actual content.

    norm_min_std = 0.0   plain normalisation (paper behaviour, default)
    norm_min_std = 0.1   recommended; caps the amplification at ~5x for the
                         smallest dims

`stats` lets a caller supply explicit (mean, std) -- used by scripts/mdn/
mdn_probe.py, which reads the statistics stored in a checkpoint so that a
model is never evaluated under a different normalisation than it was trained
with. When `stats` is given it overrides everything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    """Windowed (z_t, a_t, z_t+1) samples drawn from a latent dataset.

    Parameters
    ----------
    latents_dir : directory containing latents.npy, actions.npy, index.json
    seq_len : length T of each window; each item yields T+1 latents
    mode : "train" | "val" -- split is by *rollout*, so no rollout contributes
        windows to both sides
    train_frac : fraction of rollouts used for training
    seed : controls the rollout permutation (the split)
    normalize : apply the z normalisation
    norm_min_std : floor applied to the normalisation divisor (see module docstring)
    stats : optional (mean, std) overriding those from index.json, e.g. taken
        from a checkpoint
    """

    def __init__(self, latents_dir, seq_len: int = 32, mode: str = "train",
                 train_frac: float = 0.95, seed: int = 0, normalize: bool = True,
                 norm_min_std: float = 0.0, stats=None):
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        root = Path(latents_dir)
        info = json.loads((root / "index.json").read_text())
        self.files = info["files"]
        self.z_dim = int(info["z_dim"])
        self.seq_len = int(seq_len)
        self.normalize = bool(normalize)
        self.latents_dir = str(root)

        # NOTE: assigned to self below and re-opened per worker in __setstate__.
        # np.memmap serialises its whole buffer through pickle, which on Windows
        # spawn would push the entire dataset through a pipe for every worker.
        self._z_path = str(root / "latents.npy")
        self._a_path = str(root / "actions.npy")
        self.Z = None
        self.A = None
        self._open()

        mean = np.asarray(info["mean"], np.float32)
        std = np.asarray(info["std"], np.float32)
        std = np.where(std < 1e-6, 1.0, std)

        # Explicit assignment, NOT getattr(self, ..., default): reading an
        # unassigned attribute through a default silently disables the floor.
        # That exact mistake made an earlier retrain reproduce the previous
        # checkpoint bit for bit.
        self.norm_min_std = float(norm_min_std)
        if self.norm_min_std > 0.0:
            std = np.maximum(std, self.norm_min_std)

        if stats is not None:
            mean = np.asarray(stats[0], np.float32)
            std = np.asarray(stats[1], np.float32)
        self.stats_source = "checkpoint" if stats is not None else "index.json"
        self.mean, self.std = mean, std

        if self.mean.shape != (self.z_dim,) or self.std.shape != (self.z_dim,):
            raise ValueError(
                f"normalisation stats have shape {self.mean.shape}/"
                f"{self.std.shape}, expected ({self.z_dim},)"
            )

        # ---- split by rollout, not by window ---- #
        n = len(self.files)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(round(n * train_frac))
        sel = perm[:n_train] if mode == "train" else perm[n_train:]
        rollout_ids = sorted(sel.tolist())
        if not rollout_ids:
            raise RuntimeError(
                f"{mode} split is empty (n_rollouts={n}, train_frac={train_frac})"
            )

        # valid window starts: need seq_len+1 consecutive frames in one rollout.
        # val uses non-overlapping windows, so its windows are not correlated.
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
            raise RuntimeError(
                f"no valid windows for seq_len={self.seq_len}; "
                f"shortest rollout has {min(self.files[r][2] for r in rollout_ids)} frames"
            )

    # ------------------------------------------------------------------ #
    def _open(self):
        if self.Z is None:
            self.Z = np.load(self._z_path, mmap_mode="r")
        if self.A is None:
            self.A = np.load(self._a_path, mmap_mode="r")
        return self.Z, self.A

    # ---- Windows spawn: never ship the memmaps through the pickle pipe ---- #
    def __getstate__(self):
        state = self.__dict__.copy()
        state["Z"] = None
        state["A"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.Z = None
        self.A = None
        self._open()

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        t = self.seq_len
        s = int(self.starts[i])
        Z, A = self._open()

        z = np.asarray(Z[s:s + t + 1], dtype=np.float32)      # (t+1, D)
        a = np.asarray(A[s:s + t], dtype=np.float32)          # (t, A)
        if self.normalize:
            z = (z - self.mean) / self.std

        # .copy(): z[:t] and z[1:] are overlapping views sharing one buffer.
        # The collate function would copy anyway; being explicit avoids handing
        # out aliased tensors and costs ~4 KB per sample.
        return (torch.from_numpy(z[:t].copy()),      # z_t
                torch.from_numpy(a.copy()),          # a_t
                torch.from_numpy(z[1:].copy()))      # z_{t+1}

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Everything needed to confirm the configuration actually took effect."""
        return {
            "rollouts": len(self.rollouts),
            "windows": len(self.starts),
            "seq_len": self.seq_len,
            "z_dim": self.z_dim,
            "normalize": self.normalize,
            "norm_min_std": self.norm_min_std,
            "stats_source": self.stats_source,
            "std_min": float(self.std.min()),
            "std_max": float(self.std.max()),
        }
