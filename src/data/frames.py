"""Frame dataset for VAE training.

Reads the RAW rollout files produced by scripts/collect_data.py
(obs stored as (T, 96, 96, 3) uint8) and applies the preprocessing
pipeline from src.data.preprocess on the fly.

File-level LRU cache avoids re-opening .npz archives on every access.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocess import preprocess_frame, to_float

import json

class RolloutFrameDataset(Dataset):
    """Random access over all frames of a directory of rollout_*.npz files.

    Parameters
    ----------
    data_dir : directory containing rollout_*.npz
    files : explicit list of files (used to build disjoint train/val splits)
    crop_bottom, size, grayscale : forwarded to preprocess_frame
    cache_files : how many decoded .npz archives to keep in RAM
    """

    def __init__(self, data_dir=None, files=None, crop_bottom: int = 12,
                 size: int = 64, grayscale: bool = False, cache_files: int = 8):
        if files is None:
            if data_dir is None:
                raise ValueError("pass either data_dir or files")
            files = sorted(Path(data_dir).glob("rollout_*.npz"))
        self.files = [Path(f) for f in files]
        if not self.files:
            raise FileNotFoundError(f"no rollout_*.npz found (data_dir={data_dir})")

        self.crop_bottom = crop_bottom
        self.size = size
        self.grayscale = grayscale
        self.cache_files = cache_files
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # ---- index: (file_idx, frame_idx) -> global idx ---------------- #
        self.counts = []
        for f in self.files:
            with np.load(f) as d:
                self.counts.append(int(d["obs"].shape[0]))
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.total = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total

    # ---- internals ------------------------------------------------------- #
    def _get_obs(self, file_idx: int) -> np.ndarray:
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]
        with np.load(self.files[file_idx]) as d:
            obs = d["obs"]
        self._cache[file_idx] = obs
        if len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return obs

    # ---- API ------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        file_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        frame_idx = idx - int(self.offsets[file_idx])
        raw = self._get_obs(file_idx)[frame_idx]

        img = preprocess_frame(raw, crop_bottom=self.crop_bottom,
                               size=self.size, grayscale=self.grayscale)
        x = to_float(img)                                  # [0,1] float32
        if self.grayscale and x.ndim == 2:
            x = x[..., None]
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()   # (C,H,W)

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "frames": self.total,
            "shape": (self.size, self.size, 1 if self.grayscale else 3),
            "crop_bottom": self.crop_bottom,
        }


def split_files(data_dir, val_frac: float = 0.05, seed: int = 0):
    """File-level split -- keeps whole rollouts on one side (no frame leakage)."""
    files = sorted(Path(data_dir).glob("rollout_*.npz"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_frac)) if files else 0
    val = [files[i] for i in perm[:n_val]]
    train = [files[i] for i in perm[n_val:]]
    return sorted(train), sorted(val)

class FlatFrameDataset(Dataset):
    """Random access over a memmapped frame cache, with a file-level split."""

    def __init__(self, npy_path, files, crop_bottom: int = 12, size: int = 64,
                 grayscale: bool = False):
        npy_path = Path(npy_path)
        index_path = npy_path.with_suffix(".index.json")
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found -- run scripts/build_frame_cache.py first"
            )
        index = json.loads(index_path.read_text())
        by_name = {name: (start, count) for name, start, count in index["files"]}

        self.npy_path = str(npy_path)
        self.crop_bottom = crop_bottom
        self.size = size
        self.grayscale = grayscale

        segments = []
        for f in files:
            name = Path(f).name
            if name not in by_name:
                raise KeyError(f"{name} is not present in {index_path}")
            segments.append(by_name[name])
        self.segments = segments
        self.offsets = np.concatenate([[0], np.cumsum([c for _, c in segments])])
        self.total = int(self.offsets[-1])

        self.arr = None
        self._open()

    def _open(self):
        """Open (or re-open) the memmap. Cheap; pages are shared via the OS."""
        if self.arr is None:
            self.arr = np.load(self.npy_path, mmap_mode="r")
        return self.arr

    # ---- Windows spawn: never ship the memmap through the pickle pipe ---- #
    def __getstate__(self):
        state = self.__dict__.copy()
        state["arr"] = None          # numpy would serialise the whole buffer
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.arr = None
        self._open()                 # each worker maps the file itself

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        arr = self._open()
        seg = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        start, _ = self.segments[seg]
        local = idx - int(self.offsets[seg])
        img = preprocess_frame(np.asarray(arr[start + local]),
                               crop_bottom=self.crop_bottom,
                               size=self.size, grayscale=self.grayscale)
        x = to_float(img)
        if self.grayscale and x.ndim == 2:
            x = x[..., None]
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()

