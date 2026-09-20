"""CMA-ES training of the linear controller C in the REAL CarRacing env.

    python scripts/controller/train_controller.py \
        --vae runs/vae_pilot2/best.pt --mdn runs/mdn_pilot/best.pt \
        --workers 8 --popsize 64 --search-episodes 8

Ctrl+C is handled by the parent only; workers ignore SIGINT so the shutdown
cannot deadlock.  Use --serial for a single-process run (smoke / debugging).

Always launch in its own console so the whole tree can be closed:
    Start-Process powershell -ArgumentList '-NoExit','-Command','...'
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.models.controller import Controller
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig
from src.models.vae import VAE, VAEConfig
from src.rollout import Agent, build_env, run_episode

_W: dict = {}


# --------------------------------------------------------------------- load
def load_vae(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    vae = VAE(VAEConfig(**ck["cfg"])).to(dev).eval()
    vae.load_state_dict(ck["model"])
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ck


def load_mdn(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    mdn = MDNRNN(MDNRNNConfig(**ck["cfg"])).to(dev).eval()
    mdn.load_state_dict(ck["model"])
    return mdn, ck


def resolve_stats(mdn_ck, latents_dir, verbose=True):
    """M's own normalisation stats are authoritative; cross-check when possible."""
    mean = np.asarray(mdn_ck["latent_mean"], np.float32)
    std = np.asarray(mdn_ck["latent_std"], np.float32)
    idx = Path(latents_dir) / "index.json"
    if idx.exists():
        info = json.loads(idx.read_text())
        m2 = np.asarray(info["mean"], np.float32)
        s2 = np.asarray(info["std"], np.float32)
        dm = float(np.abs(mean - m2).max())
        ds = float(np.abs(std - s2).max())
        if verbose:
            print(f"[stats] checkpoint vs {idx}: max|dmean|={dm:.3e} max|dstd|={ds:.3e}",
                  flush=True)
        if max(dm, ds) > 1e-3:
            print("[stats] !! MISMATCH -- M was trained under different stats. "
                  "Using the checkpoint's (authoritative for M).", flush=True)
    return mean, std


def make_agent(cfg, dev, verbose_stats=None):
    """V + M + normalisation stats + C + env, ready to roll out."""
    vae, _ = load_vae(cfg["vae"], dev)
    mdn, ck = load_mdn(cfg["mdn"], dev)
    mean, std = resolve_stats(
        ck, cfg["latents_dir"],
        verbose=(cfg.get("stats_verbose", True) if verbose_stats is None
                 else verbose_stats))
    ctrl = Controller(use_hidden=cfg["use_hidden"])
    env = build_env(cfg["max_steps"])
    ctrl.set_bounds(env.action_space.low, env.action_space.high)   # before any act
    agent = Agent(vae, mdn, ctrl, mean, std, device=cfg["device"],
                  crop_bottom=cfg["crop_bottom"], size=cfg["size"],
                  grayscale=cfg["grayscale"])
    return agent, env


# ---------------------------------------------------------------- workers
def _init_worker(cfg):
    # The parent owns shutdown.  If workers died on Ctrl+C while a pool.map was
    # waiting for their results, the parent would block forever: that is the
    # "spinning, uninterruptible" behaviour we saw.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    dev = torch.device(cfg["device"])
    agent, env = make_agent(cfg, dev, verbose_stats=True)
    _W.update(cfg=cfg, agent=agent, env=env)


def _fitness_one(theta, cfg, agent, env, label=""):
    agent.ctrl.set_flat(theta)
    rets = []
    for s in cfg["search_seeds"]:
        r = run_episode(env, agent, seed=s, max_steps=cfg["max_steps"])
        rets.append(r["return"])
        if cfg["verbose"]:
            print(f"    {label} seed={s} steps={r['steps']:4d} "
                  f"return={r['return']:8.1f}", flush=True)
    return -float(np.mean(rets))          # cma minimises


def _fitness_idx(job):
    i, theta = job
    return i, _fitness_one(theta, _W["cfg"], _W["agent"], _W["env"], label=f"cand{i}")


def _eval_one(job):
    theta, seed = job
    ag = _W["agent"]
    ag.ctrl.set_flat(theta)
    return run_episode(_W["env"], ag, seed=seed, max_steps=_W["cfg"]["max_steps"])["return"]


# ------------------------------------------------------------------ args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--mdn", required=True)
    p.add_argument("--latents-dir", default="data/pilot_latents")
    p.add_argument("--out", default="runs/controller_cma")
    p.add_argument("--device", default="cpu",
                   help="cpu (recommended for many workers) or cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--serial", action="store_true",
                   help="single process, no pool (smoke / debugging)")
    p.add_argument("--popsize", type=int, default=64)          # paper: 64
    p.add_argument("--sigma0", type=float, default=0.05)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--search-episodes", type=int, default=16)  # paper: 16
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=25)       # paper: 25
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--no-hidden", action="store_true",
                   help="V-only ablation: C sees z_t only (99 params)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-episode lines in workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="skip CMA-ES; load best_theta.npz and write eval.json")
    return p.parse_args()




def main() -> int:
    a = parse_args()
    import cma

    ctrl = Controller(use_hidden=not a.no_hidden)
    n = ctrl.check_spec(expect_paper=not a.no_hidden)
    print(f"[C] params={n} in_dim={ctrl.in_dim} use_hidden={ctrl.use_hidden}", flush=True)

    if a.smoke:
        a.popsize, a.search_episodes = min(a.popsize, 4), 1
        a.eval_episodes, a.generations = min(a.eval_episodes, 2), 1
    verbose = not a.quiet
    if a.serial:
        a.workers = 0
        verbose = True

    cfg = dict(vae=a.vae, mdn=a.mdn, latents_dir=a.latents_dir, device=a.device,stats_verbose=False,
               use_hidden=not a.no_hidden, max_steps=a.max_steps,
               crop_bottom=12, size=64, grayscale=False, verbose=verbose,
               search_seeds=list(range(1000, 1000 + a.search_episodes)),
               eval_seeds=list(range(20000, 20000 + a.eval_episodes)))

    dev = torch.device(a.device)

    # ----- parent-side preflight: fails fast, in-process, with a traceback -----
    print("[preflight] building V, M, C, env in the parent ...", flush=True)
    agent, env = make_agent(cfg, dev)
    print(f"[preflight] action_space low={env.action_space.low} "
          f"high={env.action_space.high}", flush=True)
    obs = env.reset(seed=0)[0]
    z = agent.encode(obs)
    h = agent.init_hidden()
    print(f"[preflight] obs{obs.shape} -> z{z.shape} | h[0]{tuple(h[0].shape)} "
          f"| a={agent.act(z, h)}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", buffering=1)

    # ---------------- one generation, serial or pooled ---------------- #
    def fitness(sols):
        if a.workers <= 0:
            return [_fitness_one(th, cfg, agent, env, label=f"cand{i}")
                    for i, th in enumerate(sols)]
        fits = [None] * len(sols)
        for i, f in pool.imap_unordered(_fitness_idx, list(enumerate(sols)),
                                        chunksize=1):
            fits[i] = f
            done = sum(x is not None for x in fits)
            print(f"  [gen] {done}/{len(sols)} candidates "
                  f"(last mean_return={-f:8.1f}) {time.time()-t0:.0f}s", flush=True)
        return fits

    def evaluate(theta):
        if a.workers <= 0:
            rets = [run_episode(env, agent, seed=s, max_steps=cfg["max_steps"])["return"]
                    for s in cfg["eval_seeds"]]
        else:
            rets = list(pool.imap_unordered(_eval_one,
                                            [(theta, s) for s in cfg["eval_seeds"]],
                                            chunksize=1))
        return float(np.mean(rets)), float(np.std(rets, ddof=1))

    # ---------------- training loop ---------------- #
        # ---------------- training loop ---------------- #
    pool, t0 = None, time.time()
    best_theta, best_f = ctrl.get_flat(), np.inf
    theta_path = out / "best_theta.npz"
    interrupted = False

    try:
        if a.workers > 0:
            import multiprocessing as mp
            print(f"[pool] spawning {a.workers} workers (device={a.device}) ...",
                  flush=True)
            pool = mp.get_context("spawn").Pool(
                a.workers, initializer=_init_worker, initargs=(cfg,))
            print("[pool] workers ready", flush=True)

        if a.eval_only:
            # ---------- no CMA-ES: re-evaluate a saved theta ----------
            assert theta_path.exists(), (
                f"--eval-only needs {theta_path}; run training first")
            best_theta = np.asarray(np.load(theta_path)["theta"], np.float64)
            assert best_theta.size == ctrl.n_params, (
                f"{theta_path} holds {best_theta.size} params but C is "
                f"configured for {ctrl.n_params} -- did you forget "
                f"--no-hidden (99) vs 867?")
            print(f"[eval-only] loaded {theta_path} "
                  f"({best_theta.size} params) -- skipping CMA-ES", flush=True)

        else:
            # ---------- CMA-ES ----------
            es = cma.CMAEvolutionStrategy(ctrl.get_flat(), a.sigma0,
                                          {"popsize": a.popsize, "seed": a.seed,
                                           "verbose": -9})
            print(f"[cma] popsize={a.popsize} dim={ctrl.n_params} "
                  f"search={len(cfg['search_seeds'])} eps/cand", flush=True)

            for g in range(a.generations):
                tg = time.time()
                sols = es.ask()
                fits = fitness(sols)
                es.tell(sols, fits)

                rec = {"gen": g, "search_mean_return": -float(np.mean(fits)),
                       "search_best_return": -float(np.min(fits)),
                       "search_std_return": float(np.std(fits)),
                       "secs": round(time.time() - t0, 1)}
                if es.result.fbest < best_f:
                    best_f = float(es.result.fbest)
                    best_theta = np.asarray(es.result.xbest).copy()
                    np.savez(theta_path, theta=best_theta)
                    rec["new_best"] = True

                if (g + 1) % a.eval_every == 0 or a.smoke:
                    t_ev = time.time()
                    print(f"[gen {g}] evaluating best on {len(cfg['eval_seeds'])} "
                          f"episodes ...", flush=True)
                    em, esd = evaluate(best_theta)
                    rec.update(eval_mean_return=em, eval_std_return=esd,
                               eval_n=len(cfg["eval_seeds"]))
                    print(f"[gen {g:4d}] search {rec['search_mean_return']:8.2f} | "
                          f"eval {em:8.2f} +/- {esd:6.2f} "
                          f"({rec['secs']:.0f}s, gen {time.time()-tg:.0f}s, "
                          f"eval {time.time()-t_ev:.0f}s)", flush=True)
                else:
                    print(f"[gen {g:4d}] search {rec['search_mean_return']:8.2f} | "
                          f"({rec['secs']:.0f}s)", flush=True)
                log.write(json.dumps(rec) + "\n")

        # ---- FINAL EVAL: pool is still alive here -- this is the fix ---- #
        print(f"\n[eval] final: {len(cfg['eval_seeds'])} episodes on the best "
              f"individual ...", flush=True)
        em, esd = evaluate(best_theta)
        (out / "eval.json").write_text(json.dumps(
            {"eval_mean_return": em, "eval_std_return": esd,
             "n_episodes": len(cfg["eval_seeds"]), "params": int(ctrl.n_params),
             "use_hidden": ctrl.use_hidden}, indent=1))
        print(f"\nFINAL  {em:.2f} +/- {esd:.2f} over {len(cfg['eval_seeds'])} "
              f"episodes ({'867' if ctrl.use_hidden else '99'} params)", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[main] Ctrl+C -- stopping (workers ignore SIGINT; "
              "shutting the pool down)", flush=True)
    finally:
        # ONLY teardown lives here -- nothing here may need the pool
        if pool is not None:
            pool.terminate()
            pool.join()
        log.close()
        env.close()

    if interrupted:
        print(f"[main] best search return so far: {-best_f:.1f} "
              f"({theta_path.name} saved)", flush=True)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
