"""mdn_probe.py — post-hoc diagnostics on a trained MDN-RNN.

Reports four quantities that a scalar validation NLL cannot:

  [1] per-dimension NLL and per-dimension R^2 over persistence
  [2] fitted sigma per dimension, to expose overconfidence
  [3] gain over the trivial "copy z_t" predictor
  [4] action sensitivity: prediction change under shuffled / random a_t

Design notes
------------
* Every metric is a per-batch MEAN weighted by batch size, accumulated with a
  single counter `w`. An earlier version kept separate counters for elements
  (B*T, B*T*D) alongside raw sums, and a mismatch between them silently turned
  every derived number into 0 or nan -- a data-independent failure that looked
  like a data problem. Per-batch means cannot get out of step with their weight.
* Statistics come from the checkpoint, never index.json, so the model is never
  evaluated under a normalisation it was not trained with.
* No "share of total NLL": a share of a sum of NLLs is scale-dependent, and the
  per-dimension gradient is invariant to the normalisation divisor anyway
  (notes/Note4.md s8). Per-dim R^2 is the scale-invariant substitute.
* Degenerate input is reported as an explicit error with a hint, never as nan.

Usage
-----
python scripts/mdn/mdn_probe.py --mdn runs/mdn_pilot/best.pt \
    --latents data/pilot_latents [--json runs/mdn_pilot/probe.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _project_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve().parent
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").is_file() or (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"project root not found above {p}")


sys.path.insert(0, str(_project_root()))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from torch.utils.data import DataLoader                     # noqa: E402

from src.data.sequence import LatentSequenceDataset        # noqa: E402
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig         # noqa: E402

UNIT_NLL = 0.5 * math.log(2.0 * math.pi * math.e)           # 1.4189 nats
EPS = 1e-12


def mixture_mean(pi_logit, mu):
    return (F.softmax(pi_logit, dim=-1).unsqueeze(-1) * mu).sum(-2)


def mixture_log_prob(pi_logit, mu, log_sigma, target):
    t = target.unsqueeze(-2)
    sigma = torch.exp(log_sigma)
    log_comp = -0.5 * (((t - mu) / sigma) ** 2 + 2.0 * log_sigma
                       + math.log(2.0 * math.pi))
    return torch.logsumexp(F.log_softmax(pi_logit, dim=-1).unsqueeze(-1)
                           + log_comp, dim=-2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", required=True)
    ap.add_argument("--latents", default="data/pilot_latents")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"[device] {dev} | torch {torch.__version__}")

    ck = torch.load(args.mdn, map_location=dev, weights_only=False)
    cfg = MDNRNNConfig(**ck["cfg"])
    model = MDNRNN(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    D = cfg.z_dim
    print(f"[model] {args.mdn}")
    print(f"[model] epoch={ck.get('epoch')} hidden={cfg.hidden} K={cfg.n_components} "
          f"z_dim={D} params={model.num_params():,} tau_mode={cfg.tau_mode}")
    print(f"[model] paper reports 422,368 for CarRacing -- "
          f"delta={model.num_params() - 422_368:+,} (head not specified)")

    stats = (ck["latent_mean"], ck["latent_std"])
    ds = LatentSequenceDataset(args.latents, args.seq_len, args.split,
                               seed=args.seed, stats=stats)
    print(f"[norm]  stats_source=checkpoint  "
          f"norm_min_std={ck.get('norm_min_std', 'not recorded')}")
    print(f"[data]  {ds.stats()}")

    # ---- degenerate-input guard: fail loudly, name the cause ---- #
    z0, _, zt0 = ds[0]
    d0 = float((zt0 - z0).abs().max())
    print(f"[check] first window: max|z_t+1 - z_t| = {d0:.6f}  "
          f"equal={bool(torch.equal(z0, zt0))}")
    if d0 < 1e-6:
        print("\nERR the dataset returns z_t == z_t+1 to float32 precision.")
        print("    Every metric below is scaled by z_t+1 - z_t, so none of them")
        print("    would be meaningful. Check, in order:")
        print("      1. normalisation std (should be ~0.02-1.0, not huge)")
        print("      2. latents.npy vs index.json consistency")
        print("      3. checkpoint latent_mean/std vs the file")
        return 2

    std = np.asarray(ck["latent_std"], np.float32)
    active = std > 0.3
    print(f"[dims]  raw std: min={std.min():.4f} max={std.max():.4f} "
          f"mean={std.mean():.4f} | active(>0.3): {int(active.sum())}/{D}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    n_total = len(loader)
    n_use = min(n_total, args.max_batches) if args.max_batches else n_total
    print(f"[eval]  {n_use} of {n_total} batches, {len(ds)} windows, "
          f"seq_len={args.seq_len}")

    # ---- accumulate per-batch MEANS, weighted by batch size ----
    w = 0                                            # the single counter
    acc = {
        "nll_dim": np.zeros(D), "resid": np.zeros(D), "persist": np.zeros(D),
        "step": 0.0, "sens_shuf": 0.0, "sens_rand": 0.0,
        "sigma": 0.0, "floor": 0.0,
    }
    sig_min, sig_max = float("inf"), 0.0

    with torch.no_grad():
        for bi, (z, a, zt) in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            z, a, zt = z.to(dev), a.to(dev), zt.to(dev)
            bs = z.size(0)

            pi, mu, ls, _, _ = model(z, a)
            lp = mixture_log_prob(pi, mu, ls, zt)              # (B,T,D)
            pred = mixture_mean(pi, mu)

            acc["nll_dim"] += (-lp.mean(dim=(0, 1))).double().cpu().numpy() * bs
            acc["resid"] += ((pred - zt) ** 2).mean(dim=(0, 1)).double().cpu().numpy() * bs
            acc["persist"] += ((z - zt) ** 2).mean(dim=(0, 1)).double().cpu().numpy() * bs
            acc["step"] += float((zt - z).abs().mean()) * bs

            sigma = ls.exp()
            acc["sigma"] += float(sigma.mean()) * bs
            acc["floor"] += float((sigma < 2 * cfg.sigma_min).float().mean()) * bs
            sig_min = min(sig_min, float(sigma.min()))
            sig_max = max(sig_max, float(sigma.max()))

            a_perm = a[torch.randperm(bs, device=dev)]
            a_rand = torch.stack([
                torch.empty_like(a[..., 0]).uniform_(-1.0, 1.0),
                torch.empty_like(a[..., 0]).uniform_(0.0, 1.0),
                torch.empty_like(a[..., 0]).uniform_(0.0, 1.0),
            ], dim=-1)
            for key, aa in (("sens_shuf", a_perm), ("sens_rand", a_rand)):
                pi2, mu2, _, _, _ = model(z, aa)
                acc[key] += float((mixture_mean(pi2, mu2) - pred).abs().mean()) * bs

            w += bs

    if w == 0:
        print("ERR no batches evaluated")
        return 1

    nll_dim = acc["nll_dim"] / w
    resid = acc["resid"] / w
    persist = acc["persist"] / w
    step_scale = acc["step"] / w
    sigma_mean = acc["sigma"] / w
    floor_frac = acc["floor"] / w
    sens = {"shuffled": acc["sens_shuf"] / w, "random": acc["sens_rand"] / w}

    r2_dim = np.where(persist > EPS, 1.0 - resid / np.maximum(persist, EPS), np.nan)
    mse_model = float(resid.mean())
    mse_persist = float(persist.mean())
    r2_all = 1.0 - mse_model / max(mse_persist, EPS)
    ratio = {k: v / step_scale for k, v in sens.items()} if step_scale > EPS \
        else {k: float("nan") for k in sens}

    # ---- report ---- #
    print(f"\n[1] per-dimension NLL and R^2 over persistence")
    print(f"    total NLL = {nll_dim.sum():.2f}   per-dim mean = {nll_dim.mean():.3f}")
    print(f"    reference: unit-variance Gaussian = {UNIT_NLL:.3f}/dim "
          f"({UNIT_NLL * D:.2f} for {D} dims)")
    print(f"    NLL is scale dependent; residuals are printed so a degenerate "
          f"split cannot hide behind n/a:")
    print(f"\n    {'dim':>4}{'nll':>9}{'resid':>11}{'persist':>11}{'R2':>8}"
          f"{'raw_std':>10}{'':>8}")
    for i in np.argsort(-nll_dim)[:12]:
        r2 = r2_dim[i]
        r2s = f"{r2:.3f}" if not math.isnan(r2) else "n/a"
        tag = "ACTIVE" if active[i] else ""
        print(f"    {i:>4}{nll_dim[i]:>9.3f}{resid[i]:>11.3e}{persist[i]:>11.3e}"
              f"{r2s:>8}{std[i]:>10.4f}   {tag}")

    def masked_mean(x, m):
        s = (~np.isnan(x)) & m
        return float(x[s].mean()) if s.any() else float("nan")

    if active.any() and (~active).any():
        print("    ---")
        print(f"    mean NLL  active({int(active.sum())})="
              f"{masked_mean(nll_dim, active):>8.3f}   "
              f"near-constant({int((~active).sum())})="
              f"{masked_mean(nll_dim, ~active):>8.3f}")
        print(f"    mean R^2  active={masked_mean(r2_dim, active):>8.3f}   "
              f"near-constant={masked_mean(r2_dim, ~active):>8.3f}")

    print(f"\n[2] fitted sigma (normalised units)")
    print(f"    mean={sigma_mean:.4f}  min={sig_min:.2e}  max={sig_max:.4f}  "
          f"sigma_min={cfg.sigma_min:.1e}")
    print(f"    fraction at the floor: {100 * floor_frac:.2f}%"
          f"{'   <-- overconfident: NLL uninformative' if floor_frac > 0.5 else ''}")

    print(f"\n[3] next-step prediction vs 'copy z_t' (all dims, mean over {w} windows)")
    print(f"    mean |z_t+1 - z_t|         : {step_scale:.4f}")
    print(f"    MSE model                  : {mse_model:.6e}")
    print(f"    MSE persistence (copy z_t) : {mse_persist:.6e}")
    print(f"    R^2 over persistence       : {r2_all:.4f}")

    print(f"\n[4] action sensitivity")
    print(f"    |mean(a_true) - mean(a_shuffled)| : {sens['shuffled']:.5f}")
    print(f"    |mean(a_true) - mean(a_random)  | : {sens['random']:.5f}")
    print(f"    scale: mean |z_t+1 - z_t|          : {step_scale:.5f}")
    for k in ("shuffled", "random"):
        r = ratio[k]
        print(f"    ratio ({k + ' / step'})" + " " * max(1, 20 - len(k))
              + f": {r:.3f}" if not math.isnan(r) else
              f"    ratio ({k} / step) : n/a")

    print("\n--- reading the numbers ---")
    print(f" [3] R^2 over persistence = {r2_all:.3f}: "
          + ("M is essentially copying z_t." if r2_all < 0.1
             else "M predicts beyond z_t."))
    print(f" [4] ratio = {ratio['shuffled']:.3f}: "
          + ("the prediction barely uses a_t (action-blind)."
             if ratio['shuffled'] < 0.3 else "the prediction depends on a_t."))
    if floor_frac > 0.5:
        print(" [2] sigma is pinned at the floor: treat the NLL as uninformative.")

    if args.json:
        out = {
            "checkpoint": args.mdn, "epoch": ck.get("epoch"),
            "hidden": cfg.hidden, "n_components": cfg.n_components, "z_dim": D,
            "params": model.num_params(), "split": args.split,
            "windows": len(ds), "batches_used": n_use, "windows_used": int(w),
            "norm_min_std": ck.get("norm_min_std"),
            "total_nll": float(nll_dim.sum()),
            "nll_per_dim": nll_dim.tolist(),
            "unit_gaussian_nll": UNIT_NLL,
            "resid_per_dim": resid.tolist(), "persist_per_dim": persist.tolist(),
            "mse_model": mse_model, "mse_persistence": mse_persist,
            "r2_over_persistence": r2_all, "r2_per_dim": r2_dim.tolist(),
            "step_scale": step_scale,
            "action_sensitivity": sens, "action_ratio": ratio,
            "sigma_mean": sigma_mean, "sigma_min_observed": sig_min,
            "sigma_max_observed": sig_max, "sigma_floor_fraction": floor_frac,
            "latent_std": std.tolist(),
        }
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
