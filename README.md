下面是 README 的第一版，按「教授会在 60 秒内扫完」的密度来写：结果前置、偏差压缩成一张表、长论证全部留给 Notes。已按你的要求预留更新位（状态行、roadmap 勾选框、changelog）。

---

# World Models — from-scratch reproduction (CarRacing)

A clean-room reproduction of **Ha & Schmidhuber (2018), *Recurrent World Models
Facilitate Policy Evolution***, built **only from the paper** — no official or
third-party implementation was cloned or consulted. Every place where the paper is
underspecified, or where this reproduction knowingly differs, is recorded in a
numbered **deviation log** with the measurement that decided it.


---

## Results at a glance

| | this reproduction | paper (CarRacing-v0) |
| --- | --- | --- |
| Random-policy baseline | **−32.78** (2000 rollouts) | — |
| **V (ConvVAE)** reconstruction | **R² = 0.916**, 87.6 % MSE gain | — |
| **M (MDN-RNN)** density vs persistence | **ΣΔNLL = −94.2, 32/32 dims better** | — |
| **C (99 params, V-only)** | **525.15 ± 165.50** (100 rollouts) | 632 ± 251 |
| C (867 params, V + `h_t`) | _pending_ | 906 ± 21 |

Full traces: `notes/Note1.md … Note5.md`. Every number above is reproducible from the
scripts in this repository.

> **Note on scope.** This is a reproduction, not a re-implementation of the reported
> score: it uses gymnasium **CarRacing-v3** (the paper used v0) and **100** collected
> rollouts (the paper used 10,000), on a laptop CPU. Scores are therefore reported
> against *this* environment's own random-policy baseline.

---

## Pipeline

```
                  ┌─────┐        ┌──────────────┐        ┌──────────────┐
  frame x_t ────► │  V  │ ──z_t─►│      M       │        │      C       │──► a_t
                  │ VAE │        │  MDN-RNN     │──h_t──►│ linear 867 p │
                  └─────┘        └──────────────┘        └──────────────┘
                  32-d latent      P(z_{t+1}|z_t,a_t,h_t)    CMA-ES, real env
```

* **V** — 4-layer conv VAE, `z ∈ R³²`, BCE reconstruction, β = 1.0.
* **M** — LSTM (256 units) + 5-component Gaussian mixture head; trained by NLL.
* **C** — single linear layer `a_t = W_c [z_t; h_t] + b_c` (**867 parameters**),
  evolved by CMA-ES **in the real environment** (M is used only for visualisation).

---

## Deviations from the paper

Each is a one-line summary; the full argument, measurement and decision are in the
linked note.

| # | Deviation | Decided in |
| --- | --- | --- |
| 1 | Data collected under **CarRacing-v3**, not v0. No off-track termination exists in either version; episodes end by a self-imposed 1000-step limit. | Note1, Note2 §2–§4 |
| 2 | Dataset sized by **frame budget**: **100** rollouts (95 train / 5 val) instead of 10,000. | Note2 §8, Note4 §13 |
| 3 | Under a uniform-random policy the car crawls at 3.13 units/s, so **actions carry almost no information** → M is **action-blind** (ratio 0.111). A property of the data, not of M. | Note2 §5–§6, Note4 §7 |
| 4 | VAE: **3,506,435** params vs the paper's 4,348,547; **M**: 383,557 vs 422,368. The paper underspecifies both heads; deltas are printed on every run. | Note3 §9, Note4 §2 |
| 5 | **The metric first used to accept M (`R² over persistence = 0.4724`) is retracted.** It is a pooled statistic dominated by 26 near-constant latent dims and flips sign with the evaluation subset. Replaced by explained variance **0.906** and **ΣΔNLL = −94.2 (32/32 dims)**. | Note5 §3–§6 |
| 6 | `--norm-min-std` (a proposed fix) was **wrong on mathematical grounds** and is withdrawn; the code path remains, defaults to off. | Note4 §8 |

---

## Repository layout

```
src/
  data/       collect.py  preprocess.py  frames.py  sequence.py
  models/     vae.py  mdn_rnn.py  controller.py
  rollout.py  Agent + episode loop (V → normalise → M → C → env)
scripts/
  data/       collect_data.py  build_frame_cache.py
  vae/        train_vae.py  encode_latents.py  vae_check2.py
  mdn/        train_mdnrnn.py  mdn_probe.py  dream_rollout.py  z_dynamics_check.py
  controller/ train_controller.py            # CMA-ES
  tools/      bench_step.py  check_env.py
notes/        Note1.md … Note5.md             # the deviation log
runs/         checkpoints, logs, per-run JSON
data/         raw rollouts (npz), frame cache, latents
```

---

## Quickstart

```bash
# 0. environment (repo is driven by sys.path; a pyproject.toml is still pending)
pip install torch gymnasium[box2d] numpy pillow cma

# 1. V — encode frames to latents
python scripts/vae/train_vae.py    --data data/raw --out runs/vae_pilot2
python scripts/vae/encode_latents.py --vae runs/vae_pilot2/best.pt --out data/pilot_latents

# 2. M — train the dynamics model
python scripts/mdn/train_mdnrnn.py --latents data/pilot_latents --out runs/mdn_pilot

# 3. C — CMA-ES in the real environment (launch in its own console; never Ctrl+C a
#    spawn pool — close the window instead)
python scripts/controller/train_controller.py \
    --vae runs/vae_pilot2/best.pt --mdn runs/mdn_pilot/best.pt \
    --workers 8 --popsize 16 --search-episodes 8 --generations 100
#   add --no-hidden for the paper's 99-parameter V-only ablation
#   add --eval-only to re-evaluate a saved best_theta.npz without retraining
```

---

## Verification before trusting any number

Three checks that separated "the code runs" from "the number is real" (Note5 §10):

* `obs → z` reproduces the stored latents to **float16 rounding** (≈3e-4).
* Replaying a rollout's stored actions reproduces its reward exactly (**d = 0.000**).
* Evaluation is **bit-for-bit deterministic** for a fixed θ and seed set.

---

## Roadmap

- [x] V accepted — `R² = 0.916` at the dataset's information ceiling
- [x] M accepted — corrected evidence (EV 0.906, ΔNLL −94.2); action-blind, as the data dictates
- [x] C, V-only arm (99 params) — **525.15 ± 165.50**
- [ ] C, full arm (867 params) — the number to beat is **525.15 ± 165.50**
- [ ] `pyproject.toml` (ends the `sys.path` juggling that caused one silent failure)
- [ ] Probe provenance: record `argv`, hashes and normalisation source in every JSON
- [ ] Scale data to 2000 existing rollouts — the only lever on action-blindness

---

## Changelog

| date | change |
| --- | --- |
| 2026-09-20 | C module added; V-only arm trained (525.15 ± 165.50); R²-over-persistence retracted (Note5) |
| 2026-09-19 | M trained and closed; `--norm-min-std` fix retracted (Note4) |
| 2026-09-17 | V accepted at R² 0.916 after three I/O and shape fixes (Note3) |
| 2026-09-16 | Data investigation corrected Note1's root cause (Note2) |
| 2026-09-14 | Initial deviation log: v3 termination semantics (Note1) |

---

## Disclosure

* **Data:** 100 random-policy rollouts, `CarRacing-v3`, `continuous=True`,
  `lap_complete_percent=0.95`, 1000-step limit. The paper used 10,000 rollouts.
* **Compute:** AMD Ryzen 9 5900HX laptop + RTX 3060 Laptop 6 GB. Rollout throughput is
  **CPU-bound at ~250 steps/s** (the environment's renderer is ~11.5 of 13 ms per step),
  so the paper's own CMA-ES configuration (~1.84 M rollouts) is out of reach here.
  Every run reports its own generations and search episodes.
* **Action sensitivity = 0.111** is disclosed with every controller result: it bounds
  what any dream-based experiment could achieve on this dataset.
* **The retracted `0.4724` figure** remains in Note4 (with its original wording) as the
  record of the error; Note5 §5–§6 supersedes it.
* **Hardware note:** all timings are from a *laptop* GPU, not a desktop 3060.

---

## Citation

```bibtex
@incollection{ha2018worldmodels,
  title     = {Recurrent World Models Facilitate Policy Evolution},
  author    = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages     = {2451--2463},
  year      = {2018}
}
```

No code from the official implementation or any third-party reproduction was used.

