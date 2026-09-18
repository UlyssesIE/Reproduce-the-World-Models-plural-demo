"""build_frame_cache.py — flatten rollout_*.npz into one memmapped .npy.

Why: RolloutFrameDataset reads a ~26 MB archive on every cache miss, and each
DataLoader worker keeps its own small LRU (default 8 of ~100 files), so random
access is dominated by disk reads. Measured with scripts/bench_step.py:

    data  983.9 ms / batch
    comp   10.7 ms / batch      (bf16, RTX 3060 Laptop)

99% of training time was I/O wait. A single memmapped array is shared through
the OS page cache by all workers and never re-read from disk.

Writes:
    <out>.npy          raw frames, uint8, shape (N, 96, 96, 3)
    <out>.index.json   per-file (start, count), so splits can slice segments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--out", default="data/pilot_frames.npy")
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("rollout_*.npz"))
    if not files:
        raise SystemExit(f"no rollout_*.npz under {args.data}")

    counts, shapes, dtypes = [], set(), set()
    for f in files:
        with np.load(f) as d:
            o = d["obs"]
        counts.append(int(o.shape[0]))
        shapes.add(o.shape[1:])
        dtypes.add(str(o.dtype))
    assert len(shapes) == 1, f"inconsistent frame shapes: {shapes}"
    assert len(dtypes) == 1, f"inconsistent dtypes: {dtypes}"
    shape, dtype = shapes.pop(), dtypes.pop()
    total = int(sum(counts))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=dtype,
                                    shape=(total,) + shape)

    entries, i = [], 0
    for k, f in enumerate(files):
        with np.load(f) as d:
            o = d["obs"]
        arr[i:i + o.shape[0]] = o
        entries.append([f.name, i, int(o.shape[0])])
        i += o.shape[0]
        if (k + 1) % 20 == 0 or k + 1 == len(files):
            print(f"  {k+1}/{len(files)} files, {i}/{total} frames", flush=True)
    arr.flush()
    del arr

    index = {"data": str(args.data), "shape": list(shape), "dtype": dtype,
             "total": total, "files": entries}
    index_path = out.with_suffix(".index.json")
    index_path.write_text(json.dumps(index, indent=1))

    gb = total * int(np.prod(shape)) * np.dtype(dtype).itemsize / 1e9
    print(f"\nwrote {out}  ({gb:.2f} GB)")
    print(f"      {index_path}")
    print(f"  frames {total:,}  dtype {dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
