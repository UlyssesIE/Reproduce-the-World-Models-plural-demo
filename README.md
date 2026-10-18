# World Models (CarRacing) — a clean-room reproduction plus a dose–response study of the action channel

A from-paper-only reproduction of Ha & Schmidhuber (2018), with one additional measurement
the paper does not contain: **how much of the dream follows the action `a_t` versus the
dream's own dynamics**. Every deviation from the paper is numbered and traceable to a note;
every figure is reported against this environment's own baseline.

## Highlights

- **What we measure.** Dream controllability is quantifiable. On the dose axis at t = 1,
  the **synthetic** family crosses `D/B = 1` at all three temperatures — top dose (s3p90):
  2.55 at τ = 0.1, 1.41 at τ = 0.5, 1.06 at τ = 1.0; also s3p75 = 1.24 at τ = 0.1 — while
  the **real** family stays far below 1 (all six doses < 1; seed-axis real = 0.124 ± 0.011
  vs shuffled = 0.023 ± 0.001, ≈ 9σ).
- **What we reproduced.** C = **619.65** vs a random-policy baseline of **−32.78** (+652,
  ≈ 36 SE). This is the *design*, not the reported score: CarRacing-**v3**, 100 rollouts,
  ≈ 1/144 of the paper's search budget — the paper's **906 is not claimed**.
- A metric-validation protocol (5 self-checks + a positive control) and everyone's number
  is traceable to a note; retractions are numbered in `paper/callback_memo.md`.

## The finding: dose–response of the action channel

For a checkpoint and a probe directory, `dream_controllability.py` rolls the dream
forward and reports, per `(τ, t)`: `A` (stochastic response to the real action sequence),
`B` (response to a permuted one), `A/B`, and `D` (deterministic response of the mixture
mean, actions changed, sampling fixed). The headline quantity is

```
D/B := mean D(det) / mean B(noise)          over 20 rollouts
```

`D/B = 1` is the crossover where the action effect equals the dream's own noise. The
**synthetic** family crosses it on the dose axis at all three temperatures (above), while
the **real** family stays far below 1 at t = 1 (all six doses < 1). Frozen v5 table:
`notes/Note11.md`.

Two caveats, stated here so the numbers are not over-read:

- **The `noact` family reads *higher* than `real` at t = 1** (0.187 vs 0.124): the
  action slice was never trained, yet still responds deterministically to the action
  input. It is an artefact control — the informative floor is the *shuffled* family at
  0.023, not zero. The excess is localized to the numerator D (norm pinned; see
  `paper/` §4.4).
- **The synthetic family's dose axis is not pinned to a persisted parameter**: its
  `drive_frac` / `rho` / `seed` were not written into `index.json` / `ckpt.args`
  (Note11 §11). The *readings* are frozen in the v5 table; the *axis* (and the
  13×→17×→19× gap) is Tier 3 (declared non-reconstructable) in
  [`REPRODUCE.md`](REPRODUCE.md).

## Reproduction results

| | this reproduction | paper (CarRacing-v0) |
| --- | --- | --- |
| Random-policy baseline | **−32.78** (2000 rollouts) | — |
| V (ConvVAE) reconstruction | **R² = 0.916**, 87.6 % MSE gain | — |
| M (MDN-RNN) density vs persistence | **ΣΔNLL = −94.2**, 32/32 dims better | — |
| C, V-only (99 params) | **525.15 ± 165.50** (100 rollouts) | 632 ± 251 |
| C, full (867 params) | **619.65** (SE 17.96; 95 % CI [584, 655]) | 906 ± 21 |
| ↳ paired gain from `h_t` | **+94.50**, CI [+53, +136], t = 4.50 | — |

**Scope.** Reproduction of the *design*, not the score: **CarRacing-v3** (paper v0),
**100** rollouts (paper 10,000), laptop CPU. Compute: AMD Ryzen 9 5900HX + RTX 3060
Laptop 6 GB, rollout **CPU-bound ~250 steps/s** (the paper's ~1.84 M rollout search is
out of reach). The +94.50 paired gain is on identical seeds, but the two arms differ in
parameter count, so it is not decomposed into information vs capacity (Note6 §4).

## Pipeline

```
frame x_t ─►[ V: ConvVAE ]─ z_t ─►[ M: MDN-RNN ]─ h_t ─►[ C: linear 867 p ]─► a_t
                32-d latent        P(z_{t+1} | z_t, a_t, h_t)      CMA-ES, real env
```

V: 4-layer conv VAE, z ∈ ℝ³², BCE, β = 1.0 · M: LSTM 256 + 5-Gaussian MDN head ·
C: single linear layer, evolved by CMA-ES in the real environment.

## How this reproduction differs from the paper

| # | Difference | Notes |
| --- | --- | --- |
| 1 | Environment: **CarRacing-v3** (paper v0); episodes end at a 1000-step limit | 1, 2 §2–4 |
| 2 | Dataset: **100** rollouts (95/5 split) vs 10,000 | 2 §8, 4 §13 |
| 3 | Parameter counts where the paper underspecifies heads: V 3,506,435 vs 4,348,547; M 383,557 vs 422,368 | 3 §9, 4 §2 |
| 4 | M's first acceptance metric (**R² over persistence 0.4724**) is **retracted** — pooled statistic dominated by 26 near-constant dims; replaced by EV 0.906, ΣΔNLL −94.2 | 5 §3–6 |
| 5 | The proposed normalisation fix (`--norm-min-std`) and the early **"action-blind"** reading are **withdrawn**; the action channel is instead reported as the τ-resolved curve above | 4 §8, 8 §4, 11 |
| 6 | Paper §3.4 (C driving inside M's dream) is implemented; the dream's response to the action channel is the subject of this study | 7 §2–3, 11 |

## Quickstart

```bash
git clone https://github.com/UlyssesIE/Reproduce-the-World-Models-plural-demo.git /app
cd /app && export PYTHONPATH=/app
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Full pipeline (5 steps), smoke test, container setup and known pitfalls:
[`REPRODUCE.md`](REPRODUCE.md).

## Docs

- [`REPRODUCE.md`](REPRODUCE.md) — setup, full pipeline, smoke test, pitfalls, Tier 1/2/3
- `notes/Note1…11.md` — per-stage audit trail
- `CHANGELOG.md` — dated log incl. retractions (ledger: `paper/callback_memo.md`)
- `src/`, `scripts/` — code; `notes/artifacts/` — frozen credentials

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
