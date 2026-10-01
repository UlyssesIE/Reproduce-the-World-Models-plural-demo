"""Dream controllability instrument (Note9, task 2).

Question: inside a dream, is the effect of the ACTION smaller than the dream's
own SAMPLING noise?

Arms (identical warm-up => identical h_t and identical starting z_t):
  A  action effect  : a -> globally shuffled a, sampler seed HELD FIXED
  B  sampling noise : a HELD FIXED, sampler seed -> another seed
  D  deterministic  : mixture mean instead of sampling (pure response)

divergence(t) = ||z_arm(t) - z_ref(t)||_2 / E|dz|        ("typical step" units)

Instrument checks that must pass before any reading is trusted:
  [1] identity       -> same arm twice must be EXACTLY 0
  [2] a-slice zeroed -> arm A must be EXACTLY 0

All randomness is seeded: the action permutation uses its own Generator (one draw
per rep), the sampler is seeded per rollout. Two identical invocations must give
bit-identical output.

    python scripts/mdn/dream_controllability.py --mdn runs/mdn_pilot/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.models.mdn_rnn import MDNRNN, MDNRNNConfig

PERM_BASE = 31337


def mix_mean(pi_logit, mu):
    return (F.log_softmax(pi_logit, -1).exp().unsqueeze(-1) * mu).sum(-2)


def shuffled_actions(a_seq: torch.Tensor, rep: int) -> torch.Tensor:
    """Global permutation of the action sequence, seeded per rep (independent Generator)."""
    gen = torch.Generator().manual_seed(PERM_BASE + rep)
    return a_seq[torch.randperm(a_seq.shape[0], generator=gen)]


def a_slice(model, z_dim, action_dim):
    W = model.lstm.weight_ih_l0
    assert W.shape[1] == z_dim + action_dim, (tuple(W.shape), z_dim, action_dim)
    return W[:, z_dim:z_dim + action_dim]


def e_abs_dz(latents, k=2000):
    """E|z_t+1 - z_t| in NORMALISED space, first k frames."""
    root = Path(latents)
    info = json.loads((root / "index.json").read_text())
    mean = np.asarray(info["mean"], np.float64)
    std = np.asarray(info["std"], np.float64)
    Z = np.load(root / "latents.npy", mmap_mode="r")[:k]
    z = (np.asarray(Z, np.float64) - mean) / std
    return float(np.abs(np.diff(z, axis=0)).mean())


def load_episode(latents, rollout, steps, warmup, mean, std, dev):
    root = Path(latents)
    info = json.loads((root / "index.json").read_text())
    _, s0, cnt = info["files"][rollout]
    n = min(steps + warmup, cnt)
    Z = np.load(root / "latents.npy", mmap_mode="r")[s0:s0 + n]
    A = np.load(root / "actions.npy", mmap_mode="r")[s0:s0 + n]
    z = (np.asarray(Z, np.float32) - mean) / std
    return (torch.tensor(z, device=dev), torch.tensor(np.asarray(A, np.float32), device=dev))


@torch.no_grad()
def warm_hidden(model, z_pre, a_pre):
    _, _, _, _, h = model(z_pre[None], a_pre[None])
    return (h[0].clone(), h[1].clone())


@torch.no_grad()
def rollout(model, z0, a_seq, h0, seed, tau, mode="sample"):
    z = z0.clone()
    h = (h0[0].clone(), h0[1].clone())
    out = []
    if mode == "sample":
        torch.manual_seed(seed)
    for t in range(a_seq.shape[0]):
        pi, mu, ls, _, h = model(z.view(1, 1, -1), a_seq[t].view(1, 1, -1), h)
        if mode == "sample":
            z = model.sample(pi.view(-1), mu.view(-1, mu.shape[-1]),
                             ls.view(-1, ls.shape[-1]), temperature=tau).view(-1)
        else:
            z = mix_mean(pi, mu).view(-1)
        out.append(z)
    return torch.stack(out)


def div_curve(x, ref, eabs):
    d = torch.linalg.vector_norm(x - ref, dim=-1)
    return (d / eabs).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", required=True)
    ap.add_argument("--latents", default="data/pilot_latents")
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--taus", default="0.1,0.5,1.0")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.mdn, map_location=dev, weights_only=False)
    cfg = MDNRNNConfig(**ck["cfg"])
    z_dim, act_dim = int(cfg.z_dim), int(cfg.action_dim)
    model = MDNRNN(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])

    mean = np.asarray(ck["latent_mean"], np.float32)
    std = np.asarray(ck["latent_std"], np.float32)
    eabs = e_abs_dz(a.latents)
    z, A = load_episode(a.latents, a.rollout, a.steps, a.warmup, mean, std, dev)
    z0, a_seq = z[a.warmup], A[a.warmup:a.warmup + a.steps]
    # h0 = warm_hidden(model, z[:a.warmup], A[:a.warmup])
    ##########################################################################
    if a.warmup > 0:
        h0 = warm_hidden(model, z[:a.warmup], A[:a.warmup])
    else:
        h0 = model.init_hidden(1, dev)
        print("[warn] warmup=0: h0 = zeros (metric_validation's t=0 condition)")
    ##########################################################################
    steps = int(a_seq.shape[0])
    # print(f"[cfg ] {a.mdn}   rollout {a.rollout}  steps {steps}  warmup {a.warmup}")
    print(f"[cfg ] {a.mdn}  rollout {a.rollout} steps {steps} warmup {a.warmup} "
      f"reps {a.reps} taus {a.taus} E|dz| {eabs:.5f}")
    print(f"[data] E|dz| = {eabs:.5f}   (normalised space, same convention as the metric)")

    # ---------------- instrument checks ---------------------------------- #
    r1 = rollout(model, z0, a_seq, h0, 1000, 1.0)
    r2 = rollout(model, z0, a_seq, h0, 1000, 1.0)
    print(f"[1] identity      : max|diff| = {float((r1-r2).abs().max()):.3e}  (must be 0)")

    a_shuf0 = shuffled_actions(a_seq, 0)
    moved = float((a_shuf0 != a_seq).any(dim=1).float().mean())
    print(f"    permutation moved {moved*100:.1f}% of the rows (must be > 0, else vacuous)")

    probe = MDNRNN(cfg).to(dev).eval()
    probe.load_state_dict(ck["model"])
    with torch.no_grad():
        a_slice(probe, z_dim, act_dim).zero_()
    q1 = rollout(probe, z0, a_seq, h0, 1000, 1.0)
    q2 = rollout(probe, z0, a_shuf0, h0, 1000, 1.0)
    znull = float((q1 - q2).abs().max())
    print(f"[2] a-slice = 0   : arm A max|diff| = {znull:.3e}  (must be exactly 0)")
    if znull != 0.0:
        print("    -> INSTRUMENT INVALID, stop here")
        return 1

    # ---------------- arms ------------------------------------------------ #
    idx = [min(t, steps) - 1 for t in (1, 2, 4, 8, 16, 32, 64, 128, 200) if t <= steps]
    print(f"\n{'tau':>5} {'t':>5} {'A(stoch)':>16} {'B(noise)':>16} {'A/B':>7} {'D(det)':>10}")
    for tau in [float(s) for s in a.taus.split(",")]:
        acc = {k: [] for k in ("A", "B", "D")}
        for rep in range(a.reps):
            a_shuf = shuffled_actions(a_seq, rep)          # one seeded draw per rep
            base = 2000 + 137 * rep
            ref = rollout(model, z0, a_seq, h0, base, tau)
            armA = rollout(model, z0, a_shuf, h0, base, tau)          # same seed
            armB = rollout(model, z0, a_seq, h0, base + 7919, tau)    # other seed
            armD = rollout(model, z0, a_shuf, h0, base, tau, mode="det")
            det0 = rollout(model, z0, a_seq, h0, base, tau, mode="det")
            acc["A"].append(div_curve(armA, ref, eabs))
            acc["B"].append(div_curve(armB, ref, eabs))
            acc["D"].append(div_curve(armD, det0, eabs))
        for k in acc:
            acc[k] = np.stack(acc[k])
        # for tt in idx:
        #     # A_m, A_s = acc["A"][:, tt].mean(), acc["A"][:, tt].std(ddof=1)
        #     A_m, A_s = acc["A"][:, tt].mean(), acc["A"][:, tt].std(ddof=1)
        #     A_med, B_med = np.median(acc["A"][:, tt]), np.median(acc["B"][:, tt])
        #     B_m, B_s = acc["B"][:, tt].mean(), acc["B"][:, tt].std(ddof=1)
        #     D_m = acc["D"][:, tt].mean()
        #     print(f"{tau:5.2f} {tt+1:5d} {A_m:9.2f}±{A_s:5.2f} {B_m:9.2f}±{B_s:5.2f} "
        #           f"{A_m/max(B_m, 1e-9):7.2f} {D_m:10.2f}"
        #           f" | medA {A_med:8.2f} medB {B_med:8.2f}")
        for tt in idx:
            av, bv, dv = acc["A"][:, tt], acc["B"][:, tt], acc["D"][:, tt]
            A_m, A_s = av.mean(), av.std(ddof=1)
            B_m, B_s = bv.mean(), bv.std(ddof=1)
            A_med, B_med = np.median(av), np.median(bv)
            r_med = np.median(av / np.maximum(bv, 1e-9))        # paired per rep
            print(f"{tau:5.2f} {tt+1:5d} {A_m:9.2f}±{A_s:5.2f} {B_m:9.2f}±{B_s:5.2f} "
                  f"{A_m/max(B_m,1e-9):7.2f} {dv.mean():10.2f} | "
                  f"medA {A_med:8.2f} medB {B_med:8.2f} medRatio {r_med:6.2f}")

    print("\n(units: 'typical steps' = E|dz|; A/B > 1 => action matters MORE than the")
    print(" dream's own sampling noise at that horizon and tau)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
