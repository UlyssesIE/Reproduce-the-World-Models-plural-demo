"""Per-t decomposition of metric_validation's counterfactual: which t drives NEW?

Replicates metric_validation.measure() exactly (h from init, teacher-forced on the
TRUE actions, global shuffle with seed 0) but reports per t:

    mean|D|        per-element mean absolute mixture-mean difference
    ||D||_2        same difference, L2 over the 32 dims
    L2/step        <-- directly comparable to dream_controllability's divergence
    NEW_t          32*mean|D|/step   (the metric SUMS over dims: NEW = 32*mean|D|/step)

    python scripts/mdn/metric_dz_profile.py --mdn runs/mdn_pilot/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.sequence import LatentSequenceDataset
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", default=r"runs\mdn_pilot\best.pt")
    ap.add_argument("--latents", default=r"data\pilot_latents")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.mdn, map_location=dev, weights_only=False)
    cfg = MDNRNNConfig(**ck["cfg"])
    m = MDNRNN(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    stats = (np.asarray(ck["latent_mean"], np.float32),
             np.asarray(ck["latent_std"], np.float32))
    ds = LatentSequenceDataset(a.latents, seq_len=a.seq_len, mode=a.split,
                               seed=a.seed, normalize=True, stats=stats)
    Zt, A, Zn = (torch.stack(t).to(dev) for t in zip(*[ds[i] for i in range(len(ds))]))
    N, T, D = Zt.shape
    g = torch.Generator().manual_seed(a.seed)
    As = A.reshape(N * T, -1)[torch.randperm(N * T, generator=g)].reshape(N, T, -1)
    step = float((Zn - Zt).abs().mean())
    mm = lambda pi, mu: (F.log_softmax(pi, -1).exp().unsqueeze(-1) * mu).sum(-2)

    h = m.init_hidden(N, dev)
    rows = []
    with torch.no_grad():
        for t in range(T):
            pa, mua, _, _, h_new = m(Zt[:, t:t + 1], A[:, t:t + 1], h)
            pb, mub, _, _, _ = m(Zt[:, t:t + 1], As[:, t:t + 1], h)
            diff = (mm(pa, mua) - mm(pb, mub)).squeeze(1)          # (N, D)
            mean_abs = float(diff.abs().mean())
            rms = float((diff ** 2).mean().sqrt())
            rows.append((mean_abs, np.sqrt(D) * rms, rms / max(mean_abs, 1e-12)))
            h = h_new
    arr = np.array(rows)

    print(f"[cfg ] {a.mdn}   {a.split}: {N} windows  T {T}  D {D}  seed {a.seed}")
    print(f"[data] E|dz| = {step:.5f}")
    print(f"\n  t   mean|D|    ||D||2   L2/step   NEW_t")
    for t in [0, 1, 2, 4, 8, 16, 31]:
        print(f"{t:3d}  {arr[t,0]:.5f}   {arr[t,1]:.5f}  {arr[t,1]/step:7.3f}  "
              f"{32*arr[t,0]/step:7.3f}")
    for tag, sl in (("ALL ", slice(None)), ("t>=1", slice(1, None))):
        s = arr[sl]
        print(f"{tag} {s[:,0].mean():.5f}   {s[:,1].mean():.5f}  "
              f"{s[:,1].mean()/step:7.3f}  {32*s[:,0].mean()/step:7.3f}")
    print(f"\n(distribution shape rms/mean|D|: t=0 {arr[0,2]:.2f}, "
          f"mean {arr[:,2].mean():.2f}  [5.66 = constant, 7.09 = gaussian])")
    print(f"[check] NEW(ALL) should reproduce metric_validation: {32*arr[:,0].mean()/step:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
