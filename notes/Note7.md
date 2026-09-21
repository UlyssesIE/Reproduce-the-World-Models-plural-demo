# Note7 — C in the dream (§3.4)

**Status:** §3.4 implemented and executed. Action influence is below the 300-step aggregate's resolution. Dream latent speed ≈ 0.85× real (n = 10), driven by the controller's throttle; at fixed actions the residual damping is **structural, not injected noise**.
**Date:** 2026-09-21 · **Extends:** Note6, Note2 §8, Note4 §7 · **Corrects:** a single-rollout "damped ≈ 2×" reading; Note4 §3's claim that the read-only-array warning was fixed in code · **Note6 unchanged.**

---

## 1. What was added, and what it showed

`dream_rollout.py --controller <npz>` seeds M with a real `z_0` and lets C drive the autoregressive rollout (`a_t = C(z_t^norm, h_t)`, `h` reset per rollout, τ = 1.15), decoding each sampled `z` through V. The arm is inferred from `theta.size` (867 → V+h, 99 → V-only) and asserted against M's config. `--controller` overrides `--action`, which stays for Note2 §8's V+M test. Two diagnostics were added: controller action stats, and the **latent** |Δz| ratio — the informative one, since decoded frames are dominated by flat regions (Note2 §7).

In one line: **the same dream regardless of who steers; dynamics ≈ real on average; where damped, it is structural.**

---

## 2. Action independence over 300 steps (rollout 0, τ = 1.15)

| action source | latent \|Δz\| ratio | image ratio |
| --- | --- | --- |
| **C (867), closed loop** | 0.520 | 0.726 |
| recorded (`real`) | **0.491** | **0.723** |
| uniform random | **0.491** | **0.723** |

Recorded and random agree to every printed digit; C's own actions differ by ≈ 6 %. §3.4 is implemented and differs from the paper for a documented reason: M was trained where steering carries almost no information (Note2 §5–§6); the only lever is a better dataset (deferred by design).

---

## 3. Dream dynamics — and a withdrawn claim

| rollout | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latent ratio | **0.520** | 1.115 | 0.961 | 0.870 | 0.680 | 1.214 | 0.950 | 0.693 | 0.650 | 0.873 |
| gas mean | 0.256 | 0.448 | 0.447 | 0.448 | 0.348 | 0.503 | 0.567 | 0.443 | 0.194 | 0.407 |

**n = 10: latent 0.853 ± 0.069 (SE)**, median 0.872, sd 0.218; pixels **1.087 ± 0.061**. Latent ≈ 15 % below real (t = −2.14, df 9, p ≈ 0.06), but the between-rollout spread is three times that deviation.

* Pixels move slightly *faster* than real while latents are slower — sampling noise spread by the decoder; another reason to quote the latent ratio.
* **Rollout 0 (0.520) is the minimum and was the sole basis of an earlier "damped ≈ 2×" claim. Withdrawn.**
* **Exploratory:** latent ratio vs mean throttle **r = 0.75 (n = 10, df 8, p ≈ 0.013)**; nothing comparable for steering. Correlational only, but it says part of the deficit is C's cautious style (gas means 0.19–0.57, never saturated), not M. The controller is a policy, not a switch: steer spans ±1.0, brake means 0.01–0.17.

---

## 4. τ tests

**Closed loop is confounded** — the sampled `z` feeds back into C, so each τ selects a different trajectory *and policy*:

| τ | latent ratio | gas mean |
| --- | --- | --- |
| 1.00 | 0.967 | 0.565 |
| 1.15 | 0.520 | 0.256 |
| 1.50 | 0.827 | 0.549 |

Non-monotonic in τ, which pure noise injection cannot produce; the ordering follows the throttle exactly. Interpretable **only** with actions fixed — and the confound is invisible in the output, since the script prints the same three lines regardless. Same mechanism as §3's correlation.

**Clean (open-loop) test — structural, not noise** (`--action real`, rollout 0, no controller):

| τ | σ inflation | latent ratio |
| --- | --- | --- |
| 1.00 | ×1.000 | 0.528 |
| 1.15 | ×1.072 | **0.491** |
| 1.50 | ×1.225 | **0.578** |

σ +22 % → +9.5 %, non-monotonically; never above 0.58. *Indicative* quadrature split (variance = `A + B·τ`, ends only): `A = 0.168`, `B = 0.111` → **≈ 60 % structural / 40 % injected**, predicting **≈ 0.41× at τ → 0**. Falsifiable, not a result: it does not fit the τ = 1.15 point (predicts 0.543, measured 0.491), and rollout 0 is the most damped of the ten — worst case, not average. **Leading untested hypothesis:** the sampler rarely visits the heavy-tailed dims — notably dim 13 (persistence-R² −5.79 while its NLL is best in the table, Note5 §4) — and mean |Δz| rides on exactly those rare jumps.

---

## 5. Instrument observations

1. **Bit-for-bit reproducible** (`torch.manual_seed`): the fixed-action τ = 1.15 run reproduces the earlier `--action real` run to every digit.
2. **The read-only-array warning is still in the code:** Note4 §3 states the fix as `np.array(...)`; the file uses `np.asarray` (read-only view on a memmap slice). Harmless, but a third instance of "fix documented in a note, absent from the code" (Note4 §4, Note5 §9).
3. **The float16 one-liner was wrong by 2×** (0.0096 in float32 vs 0.00475): confirms Note5 §7 item 6 — and means the Note5 §2.4 "checkpoint std disagrees 2–3×" scare was that same artifact.
4. **Cosmetic:** both closing branches print unconditionally; the header prints `action=real` even under `--controller`.

---

## 6. Retractions and ledger

| Version | Hypothesis | Status | Settled by |
| --- | --- | --- | --- |
| v32 | damped ≈ 2×, M under-predicts motion | **wrong** | single-rollout outlier; n = 10 mean 0.853 |
| v33 | τ sweep with C isolates injected noise | **wrong** | closed loop; non-monotonic, tracks throttle |
| v34 | residual damping is a noise floor | **wrong** | fixed-action sweep: ≤ 0.58 throughout |
| v35 (current) | latent speed ≈ real on average, partly set by C's throttle; where damped at fixed actions it is ≈ 60 % structural; action influence below the 300-step resolution | measured (n = 10; 3 τ points, one rollout) | — |

*Process correction: my effort estimates for these runs were far too high — one dream is seconds. Estimates are now given as cost-per-run × n.*

---

## 7. Open items and disclosure

**Open:** n = 10 → 30 rollouts; clean τ test on more rollouts; test the §4 prediction (τ → 0.1 should give ≈ 0.41); per-dimension dispersion dream-vs-data; `np.asarray` → `np.array`; log the action *source* and print only the applicable closing branch; dataset 100 → 2000 (deferred by design — affects only this note, not the Note6 score).

**Disclosure:** §3.4 was executed as the paper specifies for CarRacing — C put back into M's hallucinated environment; VizDoom (§4) and iterative training (§5) are out of scope. τ = 1.15 unless stated, and τ affects only sampling, never the trained mixture parameters. **M's action sensitivity is 0.111**, reproduced over 300-step rollouts — a claim that "C learns to drive inside the dream" would be unsupported. **Small samples, stated:** n = 10 rollouts; 3 τ values (rollout 0 only); n = 1 for the action comparison; the §4 split is indicative. Deterministic and reproducible from `runs/dream_c/`; Ryzen 9 5900HX laptop, VAE decode in batches of 256 on CPU.