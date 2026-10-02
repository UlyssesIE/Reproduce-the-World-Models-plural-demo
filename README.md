# World Models — from-scratch reproduction (CarRacing)

A clean-room reproduction of **Ha & Schmidhuber (2018), *Recurrent World Models
Facilitate Policy Evolution***, built **only from the paper** — no official or
third-party implementation was cloned or consulted. Where the paper is
underspecified, or where this reproduction knowingly differs, a numbered
**deviation log** records the measurement that decided it.

On top of the paper's three modules — **V** (vision), **M** (memory), **C**
(controller) — this repository adds a line of work the paper does not contain: a
**dose–response study of the action channel**, measuring how much of M's dream
behaviour is driven by `a_t` versus M's own dynamics.
See [Action-channel study](#action-channel-study).

---

## Results

| | this reproduction | paper (CarRacing-v0) |
| --- | --- | --- |
| Random-policy baseline | **−32.78** (2000 rollouts) | — |
| **V (ConvVAE)** reconstruction | **R² = 0.916**, 87.6 % MSE gain | — |
| **M (MDN-RNN)** density vs persistence | **ΣΔNLL = −94.2**, 32/32 dims better | — |
| **C, V-only** (99 params) | **525.15 ± 165.50** (100 rollouts) | 632 ± 251 |
| **C, full** (867 params) | **619.65** (SE 17.96; 95 % CI [584, 655]) | 906 ± 21 |
| ↳ paired gain from `h_t` | **+94.50**, 95 % CI [+53, +136], t = 4.50 | — |

**Scope.** This is a reproduction of the *design*, not of the reported score. It uses
gymnasium **CarRacing-v3** (the paper used v0) and **100** collected rollouts (the paper
used 10,000), on a laptop CPU. All scores are reported against *this* environment's own
random-policy baseline; the paper's 906 is **not** claimed as reproduced.

---

## Reading the score

* **Real but short of the paper.** 619.65 vs a random baseline of −32.78 is **+652**, or
  ≈ 36 standard errors — the controller learned to drive. Against the paper's 906 it is
  ≈ 70 % of the *above-baseline* gain.
* **The gap is consistency, not peak ability.** The distribution is left-skewed
  (median **668** > mean **620**): 13 % of rollouts score ≥ 800, one completes a full lap
  (**907.8**), while another 13 % collapse at ≤ 400 (bottom-decile mean 250) and drag the
  mean down. The paper's σ = **21** means nearly *every* episode finishes; ours is
  **179.6**. We do not have "a slower driver" — we have "a driver that fails one time in
  eight".
* **Not converged.** Generation 99 still set a new best, and the search used
  **12,800 rollouts against the paper's ≈ 1.84 M (1/144)**. The curve had not flattened.
* **What that implies.** Repairing the left tail alone is worth ≈ **+50** (mean → ≈ 670).
  Reaching ~800 requires both more search and a better latent representation — the latter
  capped by the 100-rollout dataset, independently of compute. See `notes/Note6.md`.

---

## Pipeline

```
  frame x_t ─►[ V: ConvVAE ]─ z_t ─►[ M: MDN-RNN ]─ h_t ─►[ C: linear 867 p ]─► a_t
                 32-d latent          P(z_{t+1}|z_t,a_t,h_t)      CMA-ES, real env
```

* **V** — 4-layer conv VAE, `z ∈ R³²`, BCE reconstruction, β = 1.0.
* **M** — LSTM (256 units) + 5-component Gaussian mixture head; trained by NLL.
* **C** — single linear layer `a_t = W_c [z_t; h_t] + b_c` (867 params), evolved by
  CMA-ES **in the real environment**; M is used only for visualisation.

---

## How this reproduction differs from the paper

| # | Difference | Note |
| --- | --- | --- |
| 1 | Environment: **CarRacing-v3** (paper: v0). | 1, 2 §2–4 |
| 2 | Dataset sized by a **frame budget**: **100** rollouts (95/5 train–val) vs **10,000**. | 2 §8, 4 §13 |
| 3 | **Parameter counts differ** where the paper underspecifies the heads: V **3,506,435** vs 4,348,547; M **383,557** vs 422,368. | 3 §9, 4 §2 |
| 4 | The metric first used to accept M (**R² over persistence = 0.4724**) is **retracted** — a pooled statistic dominated by 26 near-constant dims. Replaced by EV 0.906 and ΣΔNLL −94.2 (32/32 dims). | 5 §3–6 |
| 5 | A proposed normalisation fix (`--norm-min-std`) was **wrong on mathematical grounds** and is withdrawn; the code path remains, defaulting to off. | 4 §8 |
| 6 | Episodes end at a **1000-step limit**; no off-track termination exists in v0 or v3, so this is the only stopping rule. | 1, 2 §2–4 |
| 7 | **The "action-blind" claim is withdrawn.** Under a random policy the car crawls at 3.13 units/s and steering cancels out, so the *training* data carries little action information — but a randomly re-drawn a-slice responds *more* than the trained one, so the ratio 0.111 lost its evidential status. The action channel is quantified by the dose–response study below. | 2 §5–6, 8 §4, 11 |
| 8 | Paper §3.4 (C driving inside M's dream) is **implemented and executed**; the dream's response to the action channel is the subject of the dose–response study. | 7 §2–3, 11 |

---

## Quickstart

### 1. Train the pipeline

```bash
pip install torch gymnasium[box2d] numpy pillow cma      # pyproject.toml still pending

python scripts/vae/train_vae.py       --data data/raw --out runs/vae_pilot2
python scripts/vae/encode_latent.py   --vae runs/vae_pilot2/best.pt --out data/pilot_latents
python scripts/mdn/train_mdnrnn.py    --latents data/pilot_latents --out runs/mdn_pilot
```

### 2. Evolve the controller

```bash
# CMA-ES in the real environment. Launch in its own console; never Ctrl+C a spawn
# pool — close the window instead.
python scripts/controller/train_controller.py \
    --vae runs/vae_pilot2/best.pt --mdn runs/mdn_pilot/best.pt \
    --workers 8 --popsize 16 --search-episodes 8 --generations 100
#   --no-hidden   the paper's 99-parameter V-only ablation
#   --eval-only   re-evaluate a saved best_theta.npz (add --dump-returns for
#                 per-rollout scores, which enable paired comparisons)
```

Layout: `src/{data,models}` + `src/rollout.py`; `scripts/{data,vae,mdn,controller,tools}`;
`notes/Note1…11.md`; `runs/` and `data/`.

---

## Reproducibility

What can be re-run from a clean clone, what needs retraining, and what is
declared non-reconstructable: see [`REPRODUCE.md`](REPRODUCE.md).

---

## Action-channel study

The paper never measures how much of M's dream depends on the action `a_t`; this
repository does. For a checkpoint and a probe directory the script rolls the dream
forward and reports, per `(τ, t)`: `A(stoch)` (response to the real action sequence),
`B(noise)` (response to a permuted one), the printed `A/B`, and `D(det)` (response of the
deterministic mixture mean). The headline quantity is

```
D/B := mean D(det) / mean B(noise)        over 20 rollouts
```

`D/B = 1` marks the crossover where the action effect matches the dream's own noise, and
the crossover's position on a dose–response axis separates the **real** family (`shuf`,
`p25…p75`, `real`) from the synthetic one (`s3p10…s3p90`, `PC2`).

**Result** (frozen v5 table; `notes/Note11.md`): the gap is **τ-dependent** — ≈ **13×** at
τ = 0.1, ≈ **17×** at τ = 0.5, ≈ **19×** at τ = 1.0 — so "at least 15×" holds only for
τ ≥ 0.5. The crossover moves right as τ grows (keep fraction 0.50→0.75 at τ = 0.1,
0.75→0.90 at τ ≥ 0.5). On the seed axis the two families separate at **≈ 9σ**
(0.1241 ± 0.0113 vs 0.0227 ± 0.0011, 20 rollouts per checkpoint).

### Reproduce it

> **Note on checkpoints.** Model checkpoints are not shipped (`runs/**` is git-ignored;
> the full set is ~640 MB, mostly Adam state). Commands that probe a checkpoint need
> `runs/*/best.pt` to exist first — retrain with the Quickstart commands above. The
> frozen-credential commands run against `notes/artifacts/` alone.

**Runs from the frozen credentials** (no checkpoints needed):

```bash
# re-render the write-up (reads notes/artifacts/*.csv only)
python scripts/mdn/frozen_report.py

# re-run the merged audit (hydrates notes/artifacts/ then runs the four stages)
python scripts/mdn/n18_audit.py
```

**Needs checkpoints** (retrain, or drop in `runs/*/best.pt`):

```bash
# one rollout, all three temperatures
python scripts/mdn/dream_controllability.py \
    --mdn runs/mdn_pilot/best.pt --latents data/pilot_latents \
    --rollout 0 --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0

# the 15-row single-cycle table, and the seed-axis error bars
python scripts/mdn/frozen_table_v5.py
python scripts/mdn/seed_spread.py
```

The two structural rows (`real-struct`, `synth3-struct`) come from checkpoints whose own
action files are entirely zero; their probe directory is overridden inside the script
(to `data/pilot_latents` and `data/pilot_latents_synth3`), so the measurement does not
degenerate to zero.

---

## Verification

Three checks separated "the code runs" from "the number is real" (Note5 §10):

* `obs → z` reproduces the stored latents to **float16 rounding** (≈ 3e-4).
* Replaying a rollout's stored actions reproduces its reward exactly (**d = 0.000**).
* Evaluation is **bit-for-bit deterministic** for a fixed θ and seed set (identical
  values recur across evaluations, and `eval.json` matches the in-run evaluation).

---

## Roadmap

- [x] V accepted — R² 0.916 at the dataset's information ceiling
- [x] M accepted — corrected evidence (EV 0.906, ΔNLL −94.2)
- [x] C, both arms — 525.15 ± 165.50 (V-only) → 619.65 (V+`h_t`), paired +94.50 (t = 4.50)
- [x] Paper §3.4 — C in the dream
- [x] **Action-channel dose–response** — 15-row table, τ ∈ {0.1, 0.5, 1.0} (Note11)
- [ ] More generations / more search episodes per candidate (8 → 16–24)
- [ ] Scale data beyond 100 rollouts — the only lever on the action channel
- [ ] `pyproject.toml`; probe provenance (`argv`, hashes, normalisation source per JSON)

---

## Changelog

| date | change |
| --- | --- |
| 2026-10-01 | Action-channel study frozen (`notes/Note11.md`), audit added (`scripts/mdn/n18_audit.py`), repository history slimmed |
| 2026-09-21 | C complete (619.65, +94.50 paired); §3.4 dream; score distribution analysed (Notes 6–7) |
| 2026-09-20 | C module added; V-only arm (525.15); R²-over-persistence retracted (Note5) |
| 2026-09-19 | M trained and closed; `--norm-min-std` retracted (Note4) |
| 2026-09-17 | V accepted at R² 0.916 after three I/O and shape fixes (Note3) |
| 2026-09-16 | Data investigation corrected Note1's root cause (Note2) |

---

## Disclosure

* **Data:** 100 random-policy rollouts, `CarRacing-v3`, `continuous=True`,
  `lap_complete_percent=0.95`, 1000-step limit. The paper used 10,000.
* **Compute:** AMD Ryzen 9 5900HX laptop + RTX 3060 Laptop 6 GB. Rollout throughput is
  **CPU-bound at ~250 steps/s**, so the paper's CMA-ES configuration (~1.84 M rollouts)
  is out of reach. Every run reports its own generations and search episodes.
* **Score comparability:** v3 rewards and termination differ from v0; the paper's
  **906 ± 21 is not claimed as reproduced**.
* **The +94.50 paired gain** (95 % CI [+53, +136], t = 4.50, 70/100 rollouts) is on
  identical seeds, but the two arms also differ in parameter count, so the gain is not
  decomposed into information vs capacity (Note6 §4).
* **The "action sensitivity = 0.111" figure is withdrawn** (Note8 §4); the action channel
  is instead reported as a τ-resolved dose–response curve (Note11).
* **M's original acceptance metric (`0.4724`)** remains in Note4 with its original
  wording as the record of the error; Note5 §5–6 supersedes it.

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

