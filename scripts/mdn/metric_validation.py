"""Validate the action-sensitivity metric BEFORE using it as evidence.

    python scripts/mdn/metric_validation.py --mdn runs/mdn_pilot/best.pt

Four controls on the SAME data (denominator fixed at E|z_t+1 - z_t|):

  1. IDENTITY   pred(A) vs pred(A)                     -> must be exactly 0.0
  2. NULL       a-slice of lstm.weight_ih_l0 := 0      -> must be exactly 0.0
  3. LAMBDA     a-slice *= lam in {.25,.5,1,2,4,8}     -> the metric must MOVE
  4. NOISE      lam=1, 5 shuffle seeds                 -> the metric's own spread

Both aggregations are printed:
  OLD  |sum D| / (N*T) / step     (mdn_probe.py: |difference of means|)
  NEW  sum |D| / (N*T) / step     (element-wise; no sign cancellation)

PASS/FAIL is printed for 1 and 2.  Do not read 3/4 until both PASS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.sequence import LatentSequenceDataset
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig


def mix_mean(pi_logit, mu):
    """(B,T,K), (B,T,K,D) -> (B,T,D)"""
    return (F.log_softmax(pi_logit, -1).exp().unsqueeze(-1) * mu).sum(-2)


def a_slice(model, z_dim, action_dim):
    """Columns of weight_ih_l0 that receive a (x = cat([z, a]) -> a is last)."""
    W = model.lstm.weight_ih_l0
    assert W.shape[1] == z_dim + action_dim, (tuple(W.shape), z_dim, action_dim)
    return W[:, z_dim:z_dim + action_dim]


def measure(model, Zt, Zn, A_a, A_b, dev):
    """h teacher-forced on A_a and shared across the two branches."""
    N, T, D = Zt.shape
    h = model.init_hidden(N, dev)
    pa, pb, na, nb = [], [], 0.0, 0.0
    with torch.no_grad():
        for t in range(T):
            z_t = Zt[:, t:t + 1, :]
            tgt = Zn[:, t:t + 1, :]
            pia, mua, lsa, _, h_new = model(z_t, A_a[:, t:t + 1, :], h)
            pib, mub, lsb, _, _ = model(z_t, A_b[:, t:t + 1, :], h)   # same h
            pa.append(mix_mean(pia, mua))
            pb.append(mix_mean(pib, mub))
            na += float(model.nll(pia, mua, lsa, tgt).sum())
            nb += float(model.nll(pib, mub, lsb, tgt).sum())
            h = h_new
    P, Q = torch.cat(pa, 1), torch.cat(pb, 1)
    D = P - Q
    cnt = N * T
    return dict(step=float((Zn - Zt).abs().mean()),
                old=abs(float(D.sum())) / cnt,
                new=float(D.abs().sum()) / cnt,
                dnll=(nb - na) / cnt, D=D)


def report(tag, m):
    r_old = m["old"] / m["step"]
    r_new = m["new"] / m["step"]
    print(f"  {tag:<26} OLD {r_old:7.4f}   NEW {r_new:7.4f}   "
          f"NEW/OLD {r_new / max(r_old, 1e-12):5.2f}   dNLL/dim {m['dnll']:+7.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", required=True)
    ap.add_argument("--latents", default="data/pilot_latents")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.mdn, map_location=dev, weights_only=False)
    z_dim = int(ck["cfg"]["z_dim"]); act_dim = int(ck["cfg"]["action_dim"])
    m = MDNRNN(MDNRNNConfig(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    stats = (np.asarray(ck["latent_mean"], np.float32),
             np.asarray(ck["latent_std"], np.float32))
    ds = LatentSequenceDataset(a.latents, seq_len=a.seq_len, mode=a.split,
                              seed=a.seed, normalize=True, stats=stats)
    n = len(ds) if a.max_windows <= 0 else min(a.max_windows, len(ds))
    Zt, A, Zn = zip(*[ds[i] for i in range(n)])
    Zt, A, Zn = (torch.stack(t).to(dev) for t in (Zt, A, Zn))
    N, T, _ = Zt.shape
    print(f"[cfg] {a.split}: {n}/{len(ds)} windows  seq_len={T}  z_dim={z_dim}  "
          f"act_dim={act_dim}  stats=checkpoint")

    orig = a_slice(m, z_dim, act_dim).detach().clone()
    g = torch.Generator().manual_seed(a.seed)

    def shuffled(seed=None):
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(N * T, generator=g)
        return A.reshape(N * T, -1)[perm].reshape(N, T, -1)

    A_shuf = shuffled()

    print("\n[1] identity null   (pred(A) vs pred(A))")
    report("identity", measure(m, Zt, Zn, A, A, dev))
    id_new = (measure(m, Zt, Zn, A, A, dev))["new"]
    print(f"      -> {'PASS' if id_new == 0.0 else 'FAIL (non-deterministic path!)'}")

    print("\n[2] weight null     (a-slice := 0)")
    with torch.no_grad():
        a_slice(m, z_dim, act_dim).zero_()
    m2 = measure(m, Zt, Zn, A, A_shuf, dev)
    report("a-weights zeroed", m2)
    print(f"      -> {'PASS' if m2['new'] == 0.0 else 'FAIL (a leaks through another path)'}")

    print("\n[3] lambda ladder   (a-slice *= lam; lam=1 is the trained model)")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        with torch.no_grad():
            a_slice(m, z_dim, act_dim).copy_(orig * lam)
        report(f"lam={lam:g}", measure(m, Zt, Zn, A, A_shuf, dev))
    with torch.no_grad():
        a_slice(m, z_dim, act_dim).copy_(orig)

    print(f"\n[4] resolution      (lam=1, {a.noise_seeds} shuffle seeds)")
    vals = []
    for k in range(a.noise_seeds):
        v = measure(m, Zt, Zn, A, shuffled(seed=1000 + k), dev)
        vals.append(v["new"] / v["step"])
        print(f"      seed {1000 + k}: NEW {vals[-1]:.4f}")
    v = np.array(vals)
    print(f"      mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"range [{v.min():.4f}, {v.max():.4f}]")

    print("\n[sanity] per-dim at lam=1 (is the numerator dominated by a few dims?)")
    d = measure(m, Zt, Zn, A, A_shuf, dev)["D"].abs().mean(dim=(0, 1)).numpy()
    top = np.argsort(-d)[:6]
    print("      top6: " + ", ".join(f"d{int(i)}={d[i]:.5f}" for i in top)
          + f"   median {np.median(d):.5f}  max {d.max():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
