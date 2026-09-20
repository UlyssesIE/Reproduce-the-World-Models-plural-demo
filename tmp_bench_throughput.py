"""Clean rollout-throughput benchmark: no CMA-ES, no final eval.

    python tmp_bench_throughput.py --workers 8
    $env:SDL_VIDEODRIVER="dummy"; python tmp_bench_throughput.py --workers 8
"""
from __future__ import annotations

import argparse, os, signal, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from src.models.controller import Controller
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig
from src.models.vae import VAE, VAEConfig
from src.rollout import Agent, build_env, run_episode

_W: dict = {}


def load_vae(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = VAE(VAEConfig(**ck["cfg"])).to(dev).eval(); m.load_state_dict(ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def load_mdn(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = MDNRNN(MDNRNNConfig(**ck["cfg"])).to(dev).eval(); m.load_state_dict(ck["model"])
    return m, ck


def _init_worker(cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    dev = torch.device("cpu")
    vae, _ = load_vae(cfg["vae"], dev)
    mdn, ck = load_mdn(cfg["mdn"], dev)
    mean = np.asarray(ck["latent_mean"], np.float32)
    std = np.asarray(ck["latent_std"], np.float32)
    ctrl = Controller(use_hidden=cfg["use_hidden"])
    env = build_env(cfg["max_steps"])
    ctrl.set_bounds(env.action_space.low, env.action_space.high)
    ag = Agent(vae, mdn, ctrl, mean, std, device="cpu")
    run_episode(env, ag, seed=0, max_steps=cfg["max_steps"])   # warm-up (untimed)
    _W.update(ag=ag, env=env, ms=cfg["max_steps"])


def _job(seed):
    t = time.time()
    r = run_episode(_W["env"], _W["ag"], seed=seed, max_steps=_W["ms"])
    return r["steps"], time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="runs/vae_pilot2/best.pt")
    ap.add_argument("--mdn", default="runs/mdn_pilot/best.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=0, help="0 = 6 x workers")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--use-hidden", action="store_true")
    a = ap.parse_args()
    import multiprocessing as mp

    cfg = dict(vae=a.vae, mdn=a.mdn, max_steps=a.max_steps,
               use_hidden=a.use_hidden)
    n = a.episodes or 6 * a.workers
    seeds = list(range(50000, 50000 + n))

    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker,
                                      initargs=(cfg,)) as p:
        t_ready = time.time()
        res = p.map(_job, seeds, chunksize=1)
        t_done = time.time()

    steps = sum(s for s, _ in res)
    per = [t for _, t in res]
    wall = t_done - t_ready
    print(f"[bench] workers={a.workers}  "
          f"SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', '<unset>')}")
    print(f"[bench] pool build + warm-up : {t_ready - t0:.1f}s")
    print(f"[bench] {n} episodes / {steps} steps in {wall:.1f}s")
    print(f"[bench] **THROUGHPUT {n/wall:.3f} eps/s | {steps/wall:.0f} steps/s**")
    print(f"[bench] wall per episode (worker view): mean {np.mean(per):.2f}s  "
          f"median {np.median(per):.2f}s  max {np.max(per):.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())
