
```markdown
# World Models — from-scratch reproduction (CarRacing)

A clean-room reproduction of **Ha & Schmidhuber (2018), *Recurrent World Models
Facilitate Policy Evolution***, built **only from the paper** — no official or
third-party implementation was cloned or consulted. Where the paper is
underspecified, or where this reproduction knowingly differs, a numbered
**deviation log** records the measurement that decided it.

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

## Deviations from the paper

| # | Deviation | Note |
| --- | --- | --- |
| 1 | Data collected under **CarRacing-v3**, not v0; no off-track termination exists in either version, so episodes end at a self-imposed 1000-step limit. | 1, 2 §2–4 |
| 2 | Dataset sized by **frame budget**: **100** rollouts (95/5 train-val) vs 10,000. | 2 §8, 4 §13 |
| 3 | Under a random policy the car crawls at 3.13 units/s and steering cancels out, so **actions carry almost no information** → M is **action-blind** (ratio 0.111). A property of the data, not of M. | 2 §5–6, 4 §7 |
| 4 | Parameter counts differ where the paper underspecifies the heads: V **3,506,435** vs 4,348,547; M **383,557** vs 422,368. | 3 §9, 4 §2 |
| 5 | **The metric first used to accept M (`R² over persistence` = 0.4724) is retracted** — a pooled statistic dominated by 26 near-constant dims that flips sign with the evaluation subset. Replaced by EV 0.906 and ΣΔNLL −94.2 (32/32 dims). | 5 §3–6 |
| 6 | A proposed normalisation fix (`--norm-min-std`) was **wrong on mathematical grounds** and is withdrawn; the code path remains, defaulting to off. | 4 §8 |
| 7 | Paper §3.4 (C driving inside M's dream) is **implemented and executed**, but the dream is action-independent — the same picture regardless of who steers — because of deviation 3. | 7 §2–3 |

---

## Quickstart

```bash
pip install torch gymnasium[box2d] numpy pillow cma      # pyproject.toml still pending

python scripts/vae/train_vae.py       --data data/raw --out runs/vae_pilot2
python scripts/vae/encode_latents.py  --vae runs/vae_pilot2/best.pt --out data/pilot_latents
python scripts/mdn/train_mdnrnn.py    --latents data/pilot_latents --out runs/mdn_pilot

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
`notes/Note1…7.md`; `runs/` and `data/`.

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
- [x] M accepted — corrected evidence (EV 0.906, ΔNLL −94.2); action-blind by data
- [x] C, both arms — 525.15 ± 165.50 (V-only) → 619.65 (V+`h_t`), paired +94.50 (t = 4.50)
- [x] Paper §3.4 — C in the dream; action-independence reproduced over 300-step rollouts
- [ ] More generations / more search episodes per candidate (8 → 16–24)
- [ ] Scale data to the 2000 rollouts already on disk — the only lever on action-blindness
- [ ] `pyproject.toml`; probe provenance (`argv`, hashes, normalisation source per JSON)

---

## Changelog

| date | change |
| --- | --- |
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
  **CPU-bound at ~250 steps/s** (the environment's renderer is ~11.5 of 13 ms per step),
  so the paper's CMA-ES configuration (~1.84 M rollouts) is out of reach. Every run
  reports its own generations and search episodes.
* **Score comparability:** v3 rewards and termination differ from v0; the paper's
  **906 ± 21 is not claimed as reproduced**, and our result is quoted against this
  environment's own baseline.
* **The +94.50 paired gain** (95 % CI [+53, +136], t = 4.50, 70/100 rollouts) is on
  identical seeds, but the two arms also differ in parameter count, so the gain is not
  decomposed into information vs capacity (Note6 §4).
* **Action sensitivity = 0.111** is disclosed with every controller result: it bounds
  what a dream-based experiment can achieve on this dataset (Note7).
* **M's original acceptance metric (`0.4724`)** remains in Note4 with its original
  wording as the record of the error; Note5 §5–6 supersedes it.
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
```

