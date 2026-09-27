"""Validate the action-sensitivity metric BEFORE using it as evidence.

Core question: does
        NEW = mean|pred(a_true) - pred(a_shuf)| / E|dz|
measure "how much M uses the action", or does it mostly measure "how large the
a-columns of lstm.weight_ih_l0 happen to be"?

Controls (numerator, denominator and both aggregations always printed):

  1 identity       pred(A) vs pred(A)                     must be exactly 0
  2 weight-null    a-slice := 0                           must be exactly 0
  3 lam ladder     a-slice *= {0,.25,.5,1,2,4,8}          must move monotonically
  4 resolution     5 shuffle seeds at lam=1               spread << signal
  5 RANDOM-INIT    same architecture, never trained       <-- decisive control
  6 A-REINIT       trained model, a-slice re-drawn from the init distribution

MDNRNN._init_weights uses nn.init.orthogonal_(gain=1.0) on weight_ih_l0, so the
a-columns start at O(1).  If an untrained model scores >= the trained one, NEW is
dominated by weight scale, and a cross-model comparison is only valid like-for-like
(same data, same architecture, same training budget) -- i.e. report a RATIO against
an architecture-matched control, never an absolute number.

    python scripts/mdn/metric_validation.py --mdn runs/mdn_pilot/best.pt --split val
    python scripts/mdn/metric_validation.py --mdn runs/mdn_pilot_synth/best.pt --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sequence import LatentSequenceDataset
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig


# --------------------------------------------------------------------------- #
def mix_mean(pi_logit, mu):
    """(B,T,K), (B,T,K,D) -> (B,T,D)"""
    return (F.log_softmax(pi_logit, -1).exp().unsqueeze(-1) * mu).sum(-2)


def a_slice(model, z_dim, action_dim):
    """Columns of weight_ih_l0 receiving a (x = cat([z, a]) -> a is last)."""
    W = model.lstm.weight_ih_l0
    assert W.shape[1] == z_dim + action_dim, (tuple(W.shape), z_dim, action_dim)
    return W[:, z_dim:z_dim + action_dim]


def a_norm(model, z_dim, action_dim):
    return float(a_slice(model, z_dim, action_dim).detach().norm())


def measure(model, Zt, Zn, A_a, A_b, dev):
    """h teacher-forced on A_a and shared across both branches (one-step counterfactual)."""
    N, T, _ = Zt.shape
    h = model.init_hidden(N, dev)
    pa, pb = [], []
    na = nb = 0.0
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
    D = torch.cat(pa, 1) - torch.cat(pb, 1)
    cnt = N * T
    return dict(step=float((Zn - Zt).abs().mean()),
                new_num=float(D.abs().sum()) / cnt,      # mean |D|
                old_num=abs(float(D.sum())) / cnt,       # |mean D|
                nll_true=na / cnt,
                dnll=(nb - na) / cnt,
                D=D)


def row(tag, m, wnorm=None, extra=""):
    r_new = m["new_num"] / m["step"]
    r_old = m["old_num"] / m["step"]
    w = f"  ||Wa|| {wnorm:7.3f}" if wnorm is not None else ""
    print(f"  {tag:<22} |D|bar {m['new_num']:.5f} / step {m['step']:.5f}"
          f" = NEW {r_new:9.3f} | OLD {r_old:9.6f}"
          f" | cancel {r_new / max(r_old, 1e-12):8.1f}x"
          f" | NLL {m['nll_true']:8.3f}{w}{extra}")
    return r_new


def reinit_a_slice(model, z_dim, action_dim, seed):
    """Replace the a-columns with a fresh draw from the SAME init distribution as
    training used (orthogonal_, gain=1), leaving the z-columns untouched."""
    W = model.lstm.weight_ih_l0
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    fresh = torch.empty_like(W)
    nn.init.orthogonal_(fresh, gain=1.0)
    torch.set_rng_state(state)
    with torch.no_grad():
        W[:, z_dim:z_dim + action_dim].copy_(fresh[:, z_dim:z_dim + action_dim])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", required=True)
    ap.add_argument("--latents", default="data/pilot_latents")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seeds", type=int, default=5)
    ap.add_argument("--init-seed", type=int, default=12345,
                    help="seed for the untrained model and the a-slice re-draw")
    ap.add_argument("--skip-controls", action="store_true",
                    help="only run the comparative block (conditions 5-6)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.mdn, map_location=dev, weights_only=False)
    cfg = MDNRNNConfig(**ck["cfg"])
    z_dim, act_dim = int(cfg.z_dim), int(cfg.action_dim)

    trained = MDNRNN(cfg).to(dev).eval()
    trained.load_state_dict(ck["model"])
    trained.eval()

    # normalisation ALWAYS from the checkpoint, so every condition sees identical data
    stats = (np.asarray(ck["latent_mean"], np.float32),
             np.asarray(ck["latent_std"], np.float32))
    ds = LatentSequenceDataset(a.latents, seq_len=a.seq_len, mode=a.split,
                               seed=a.seed, normalize=True, stats=stats)
    n = len(ds) if a.max_windows <= 0 else min(a.max_windows, len(ds))
    Zt, A, Zn = zip(*[ds[i] for i in range(n)])
    Zt, A, Zn = (torch.stack(t).to(dev) for t in (Zt, A, Zn))
    N, T, _ = Zt.shape

    g = torch.Generator().manual_seed(a.seed)

    def shuffled(seed=None):
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(N * T, generator=g)
        return A.reshape(N * T, -1)[perm].reshape(N, T, -1)

    A_shuf = shuffled()
    m_one = measure(trained, Zt, Zn, A, A_shuf, dev)
    print(f"[cfg]  {a.mdn}")
    print(f"[data] {a.split}: {n}/{len(ds)} windows  seq_len={T}  z_dim={z_dim}  "
          f"act_dim={act_dim}  stats=checkpoint")
    print(f"[data] denominator  E|z_t+1 - z_t| = {m_one['step']:.5f}  "
          f"(data-only; identical in every condition)")

    # ------------------------------------------------------------------ #
    if not a.skip_controls:
        orig = a_slice(trained, z_dim, act_dim).detach().clone()

        print("\n[1] identity null  (pred(A) vs pred(A))")
        m = measure(trained, Zt, Zn, A, A, dev)
        row("identity", m)
        print(f"      -> {'PASS' if m['new_num'] == 0.0 else 'FAIL (non-deterministic path)'}")

        print("\n[2] weight null    (a-slice := 0)")
        with torch.no_grad():
            a_slice(trained, z_dim, act_dim).zero_()
        m = measure(trained, Zt, Zn, A, A_shuf, dev)
        row("a-weights zeroed", m)
        print(f"      -> {'PASS' if m['new_num'] == 0.0 else 'FAIL (a leaks elsewhere)'}")
        with torch.no_grad():
            a_slice(trained, z_dim, act_dim).copy_(orig)

        print("\n[3] lambda ladder  (a-slice *= lam; lam=1 is the trained model)")
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            with torch.no_grad():
                a_slice(trained, z_dim, act_dim).copy_(orig * lam)
            m = measure(trained, Zt, Zn, A, A_shuf, dev)
            row(f"lam={lam:g}", m, wnorm=a_norm(trained, z_dim, act_dim))
        with torch.no_grad():
            a_slice(trained, z_dim, act_dim).copy_(orig)

        print(f"\n[4] resolution     (lam=1, {a.noise_seeds} shuffle seeds)")
        vals = []
        for k in range(a.noise_seeds):
            m = measure(trained, Zt, Zn, A, shuffled(seed=1000 + k), dev)
            vals.append(m["new_num"] / m["step"])
            print(f"      seed {1000 + k}: NEW {vals[-1]:.4f}")
        v = np.array(vals)
        print(f"      mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
              f"range [{v.min():.4f}, {v.max():.4f}]")

    # ------------------------------------------------------------------ #
    print("\n[5/6] comparative block at lam=1  (the decisive test)")

    rand = MDNRNN(cfg).to(dev).eval()          # fresh init, WRONG weights for the data
    state = torch.get_rng_state()
    torch.manual_seed(a.init_seed)
    rand.apply(lambda mod: mod.reset_parameters()
               if hasattr(mod, "reset_parameters") else None)
    rand._init_weights()                        # same distribution as training used
    torch.set_rng_state(state)

    areinit = MDNRNN(cfg).to(dev).eval()
    areinit.load_state_dict(ck["model"])
    reinit_a_slice(areinit, z_dim, act_dim, a.init_seed)

    table = {}
    for tag, mdl in (("trained", trained), ("random-init", rand), ("a-reinit", areinit)):
        m = measure(mdl, Zt, Zn, A, A_shuf, dev)
        table[tag] = (row(tag, m, wnorm=a_norm(mdl, z_dim, act_dim)), m)

    r_tr, r_ra = table["trained"][0], table["random-init"][0]
    r_ar = table["a-reinit"][0]
    print("\n  ---- verdict ----")
    print(f"  NEW(random-init) / NEW(trained) = {r_ra / max(r_tr, 1e-12):.2f}")
    print(f"  NEW(a-reinit)    / NEW(trained) = {r_ar / max(r_tr, 1e-12):.2f}")
    if r_ra >= r_tr * 0.8:
        print("  => the metric is dominated by the SCALE of the a-weights, not by")
        print("     whether M learned to use the action.  Quote it only as a RATIO")
        print("     against an architecture-matched control, never as an absolute.")
    elif r_ra < r_tr * 0.5:
        print("  => the trained model scores well above an untrained one with the same")
        print("     architecture: the metric does carry learned action usage.")
        print("     Report NEW_trained / NEW_random-init as the calibrated quantity.")
    else:
        print("  => intermediate: report the ratio and state the control's score.")

    print("\n[sanity] per-dim mean|D| at lam=1 (trained model)")
    d = table["trained"][1]["D"].abs().mean(dim=(0, 1)).numpy()
    top = np.argsort(-d)[:6]
    print("      top6: " + ", ".join(f"d{int(i)}={d[i]:.5f}" for i in top)
          + f"   median {np.median(d):.5f}  max {d.max():.5f}"
          + f"   max/median {d.max() / max(np.median(d), 1e-12):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
