# Reproducibility guide

This document states, in one place, **what a reader can re-run, what they cannot, why**,
and the exact commands to do it — the things reviewers actually check. It is written
against the state of the repository at the time of writing and complements `README.md`,
whose four-line Quickstart is the entry point; this guide carries the full pipeline.

Everything below was **measured**, not assumed. Where a claim rests on a run, the command
and its exit code are given.

---

## Environment

| item | value |
| --- | --- |
| Python | 3.10 or 3.11 (vizdoom has no 3.12 wheels); developed on 3.11.16 |
| PyTorch | 2.5.1 |
| NumPy | 2.4.6 |
| Gymnasium | 1.3.0 |
| Key deps | `torch`, `gymnasium[box2d]`, `numpy`, `pillow`, `cma` (full list in `pyproject.toml`) |
| Hardware used | AMD Ryzen 9 5900HX laptop + RTX 3060 Laptop (6 GB) |

Dependencies are declared in **`pyproject.toml`** (mirrored in `requirements.txt`); install
with `pip install -r requirements.txt`, after installing the CUDA torch wheels:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

---

## Setup & install (verified on a clean Ubuntu 22.04, nothing preinstalled)

```bash
# 2 GB of shared memory is required — Docker's 64 MB default kills DataLoader workers
docker run --gpus all --shm-size=2g -it --rm ubuntu:22.04 bash

apt-get update && apt-get install -y python3 python3-pip python3-venv git
git clone https://github.com/UlyssesIE/Reproduce-the-World-Models-plural-demo.git /app
cd /app && export PYTHONPATH=/app      # makes `from src…` importable; there is no build step

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

No GPU? Drop `--gpus all`; every script falls back to CPU automatically (~20× slower). On
WSL2 install a WSL-capable NVIDIA driver so `nvidia-smi` works; in Docker the host needs
`nvidia-container-toolkit`.

## Verification status

| stage | run at full scale | re-verified in a clean container |
| --- | --- | --- |
| container + dependencies | — | ✅ |
| collect 100 rollouts (`data/raw`) | ✅ | ✅ |
| frame cache (`data/raw`) | ✅ | ✅ |
| V — full VAE training (30 epochs, default) | ✅ | ✅ |
| V — `encode_latent` | ✅ | ✅ (smoke scale) |
| M — MDN-RNN training | ✅ | ✅ (smoke scale) |
| C — CMA-ES controller | ✅ | ✅ (smoke scale) |
| Tier 1 (table recomputation) | ✅ | ✅ |

The reported numbers come from the full-scale runs; the clean-container column shows which
stages were re-exercised there, and at what scale.

---

## Smoke test — minutes, one line at a time

The path after `--cache` / `--frame-cache` must stay on its line. On the 2 GB container
above no special flags are needed, but `--workers 0` keeps it safe on a smaller `/dev/shm`:

```bash
python3 scripts/tools/check_env.py --skip-env
# On a CPU-only host this prints "[FAIL] cuda.is_available() False" and exits 1. That is
# expected — no GPU is required — and does not block the CPU pipeline below.

python3 scripts/data/collect_data.py      --out data/smoke --num-rollouts 20 --max-episode-steps 200
python3 scripts/data/build_frame_cache.py --data data/smoke --out data/smoke_frames.npy
python3 scripts/vae/train_vae.py     --data data/smoke --frame-cache data/smoke_frames.npy --out runs/smoke_vae --epochs 1 --workers 0
python3 scripts/vae/encode_latent.py --vae runs/smoke_vae/best.pt --cache data/smoke_frames.npy --data data/smoke --out data/smoke_latents --workers 0
python3 scripts/mdn/train_mdnrnn.py  --latents data/smoke_latents --out runs/smoke_mdn --epochs 1 --workers 0
python3 scripts/controller/train_controller.py --vae runs/smoke_vae/best.pt --mdn runs/smoke_mdn/best.pt --latents-dir data/smoke_latents --smoke --serial --max-steps 200
```

**Known pitfalls** (all first hit from a clean container):

| symptom | cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | the repo root is not on the path — `export PYTHONPATH=/app` |
| `python: command not found` | only `python3` is installed — use `python3` |
| `check_env.py`: `cuda.is_available() False` | CPU-only host; expected, and the CPU pipeline runs fine |
| `--cache` / `--frame-cache`: `expected one argument` | the path was lost to a line break — keep the command on one line |
| `Bus error` / `DataLoader worker killed` | container `/dev/shm` too small — start Docker with `--shm-size=2g`, or add `--workers 0` |
| `val split is empty (n_rollouts=…)` | the split is by rollout at `train_frac=0.95`, so **fewer than 11 rollouts leaves no validation data** — the smoke test collects 20 |

---


## Full pipeline — one line per run (the paper used 10,000 rollouts)

```bash
# 1. collect random-policy rollouts from CarRacing-v3
python3 scripts/data/collect_data.py --out data/raw --num-rollouts 100

# 2. V — train the ConvVAE, then freeze it and encode latent sequences
python3 scripts/data/build_frame_cache.py --data data/raw --out data/pilot_frames.npy
python3 scripts/vae/train_vae.py --data data/raw --frame-cache data/pilot_frames.npy --out runs/vae_pilot2
python3 scripts/vae/encode_latent.py --vae runs/vae_pilot2/best.pt --cache data/pilot_frames.npy --data data/raw --out data/pilot_latents

# 3. M — train the MDN-RNN on the latent sequences
python3 scripts/mdn/train_mdnrnn.py --latents data/pilot_latents --out runs/mdn_pilot

# 4. C — evolve the controller with CMA-ES in the real environment
#    (launch in its own console; never Ctrl+C a spawn pool — close the window)
python3 scripts/controller/train_controller.py --vae runs/vae_pilot2/best.pt --mdn runs/mdn_pilot/best.pt --workers 8 --popsize 16 --search-episodes 8 --generations 100
#   --no-hidden   the paper's 99-parameter V-only ablation
#   --eval-only   re-evaluate a saved best_theta.npz (add --dump-returns for per-rollout scores)

# 5. probe the checkpoint (~1,250 steps), or re-render the frozen tables
python3 scripts/mdn/dream_controllability.py --mdn runs/mdn_pilot/best.pt --latents data/pilot_latents --rollout 0 --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0
python3 scripts/mdn/frozen_table_v5.py     # 15-row dose-response table (needs checkpoints)
python3 scripts/mdn/seed_spread.py         # seed-axis error bars (needs checkpoints)
python3 scripts/mdn/frozen_report.py      # rebuilds notes/Note11.md from notes/artifacts/ alone
python3 scripts/mdn/n18_audit.py          # merged audit; notes/artifacts/ alone
```

---

## Tier 1 — recomputable from a clean clone (no checkpoints)

These read only files tracked in `notes/artifacts/`. They run in a fresh `git clone`
with **no `runs/*.pt` and no repository-root `tmp_*` files**: `frozen_report.py`,
`n18_audit.py` (step 5 above).

**Verified.** Exported `HEAD` with `git archive` into an empty directory (0 checkpoints,
0 repository-root `tmp_*`), then ran both: **exit code 0** for each.
`frozen_report.py` rewrote `notes/Note11.md` (257 lines); `n18_audit.py` completed all
four stages. Facts these commands rest on:

* `frozen_report.py` reads 16 credentials; **all 16 are present** in `notes/artifacts/`.
* It imports **no** `torch` and touches **no** checkpoint.
* It re-checks its own source anchors (citations into `notes/Note9.md`, `Note8.md`,
  `Note10.md`) on every run; a moved line is reported as a failed anchor, not silently
  ignored.

> `n18_audit.py` hydrates `notes/artifacts/*` into the repository root before running its
> stages. Those hydrated files match the `.gitignore` `tmp_*` rule, so they never appear
> in `git status`. The audit's stage-1 `tmp_v5/` section reads the raw per-rollout dumps,
> which are **not shipped**; when the directory is absent the stage now reports this
> explicitly instead of raising `FileNotFoundError` (see *Fixed*, below).

---

## Tier 2 — reproducible by retraining (checkpoints not shipped)

**No checkpoint is tracked in the repository** — `runs/**/*.pt` is git-ignored; only the
training curves (`runs/**/log.jsonl`) are tracked. The checkpoint working set is
**≈ 429 MB** (mostly Adam state). A fresh clone therefore **cannot** run the probe, the
two table scripts, or the off-pipeline families without retraining first (steps 3 and 5
above; the families use `train_mdnrnn.py --latents data/pilot_latents_shuf`,
`train_mdnrnn_norm.py`, and `n24_gate1_2x2.py`).

This is a deliberate size trade-off, not an oversight: shipping the checkpoint set would
add ~429 MB to a repository whose tracked content is already **~138 MB of probe arrays**
(`data/*/{latents,actions}.npy`, 19 directories). The retraining path and its cost are
given above; every run reports its own generations and search episodes.

> **Size note for supplementary material.** `data/` alone is **≈ 138 MB**, above TMLR's
> **100 MB** supplementary-material limit (`jmlr.org/tmlr/author-guide.html`). Any
> packaged supplement must therefore either exclude `data/` or split it; the arrays are
> small individually (≈ 7 MB per probe directory) and could be subset to the probes the
> paper actually cites.

---

## Tier 3 — not reconstructable (declared, not hidden)

* **The `synth3pXX` dose generator is not preserved.** All seven doses ship frozen and
  share one byte-identical `latents.npy` (md5 `f5433cf8e0de`); the dose lives entirely in
  `actions.npy` as a keep fraction `0.10 … 0.90` at mean reversion `rho ≈ 0.021`. The
  random draws used (the action-to-latent map `V` and the per-dose row permutation) were
  never persisted, so the arrays **cannot be regenerated byte-for-byte** and no such
  regeneration is claimed. This is restated in `notes/Note11.md` §14.
* **`checkpoint.args` does not record every setting.** `--norm-target` (used for the
  norm-pinned control checkpoints) and `--drive-frac` / `rho` / `seed` for the synthetic
  latents are **not written into `checkpoint.args` or `index.json`**, so a checkpoint's
  probe conditions cannot be recovered from the checkpoint alone. Where those settings
  matter, they are documented beside the reading (e.g. `notes/Note11.md` §11).

---

## What is asserted vs. what is recomputed

The central dose–response headline is **recomputed, not asserted**: `notes/Note11.md` is
generated by `frozen_report.py` from `notes/artifacts/tmp_v5_table.csv`, and the
τ-resolved crossover is interpolated on the **measured** `x` column
(`x = NLL(λ=0) − NLL(λ=1)`), not on any hand-entered axis label. Re-running Tier 1
reproduces the note byte-for-byte.


---

## Audit trail

| command | needs | exit code (clean clone) |
| --- | --- | --- |
| `scripts/mdn/frozen_report.py` | `notes/artifacts/` only | **0** |
| `scripts/mdn/n18_audit.py` | `notes/artifacts/` only | **0** |
| `scripts/mdn/frozen_table_v5.py` | `runs/*/best.pt` | not run (Tier 2) |
| `scripts/mdn/seed_spread.py` | `runs/*/best.pt` | not run (Tier 2) |
| `scripts/mdn/dream_controllability.py` | `runs/*/best.pt` | not run (Tier 2) |
