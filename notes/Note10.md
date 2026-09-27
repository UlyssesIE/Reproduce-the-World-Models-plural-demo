# Note10 — Dream controllability: instrument, verification, verdict

Date: 2026-09-27 · Companion to Note9 (action channel). Note9 §1–§11 stand unchanged;
this note covers M's behaviour *inside the dream* (autoregressive rollout).
Supersedes the dream-controllability statements of the handover brief (§2.7, §3.2)
and every reading produced by the pre-guard version of the script.

## 0. One line

**Verdict**: inside a dream the action effect is strictly *smaller* than the dream's own
sampling noise over the entire measured horizon (A/B = 0.02–0.88 at τ=1), the ratio rises
monotonically toward ≈1, and a cold (h=0) dream is ≈16× less controllable than a warm one.
The brief's "the dream's noise is half the teacher-forcing prediction" story is **false**
(input mismatch, not physics).

| Claim | Verdict |
|---|---|
| "the dream's sampling noise is half what teacher forcing predicts" | **False** (§4) |
| "the mixture gate collapses inside the dream (Σπ²≈0.98, 1 effective component)" | **False** — measured Σπ²=0.5260, eff#comp **1.901** (§4) |
| "A/B → 1 because A is a noise channel" | **False** — A/B keeps rising; PC2 reaches 3.21 at t=200 (§3) |
| "A/B < 1 holds" | **True** at all 18 (t, regime) grid points measured |

## 1. Instrument

`scripts/mdn/dream_controllability.py`. One arm set, three arms, identical start state:

| Arm | Perturbation held | Perturbation varied |
|---|---|---|
| **A** (action effect, stochastic) | sampler seed | action sequence (global permutation) |
| **B** (dream's own sampling noise) | action sequence | sampler seed |
| **D** (deterministic response) | — | action permutation, `mix_mean` instead of sampling |

- Warm-up: `h0 = warm_hidden(model, z[:warmup], A[:warmup])`, `z0 = z[warmup]`. Arm step 1
  therefore consumes exactly the tensor teacher forcing consumes at index `warmup`
  (this is what makes §4 exact).
- `div(t) = ‖z_arm(t) − z_ref(t)‖₂ / E|dz|`, "typical steps" units. `E|dz| = 0.05378`,
  computed from `index.json` mean/std over the first 2000 frames, normalised space.
- Seeding: `PERM_BASE = 31337` + one `Generator` per rep for the permutation; the sampler is
  seeded per rollout (`torch.manual_seed(base)`, arm B uses `base + 7919`).
- `medRatio = median_over_reps(A / B)` — paired per rep, **not** `medA / medB`.

## 2. Instrument verification (5 checks, all pass)

| # | Check | Output | Status |
|---|---|---|---|
| 1 | same arm twice | `max|diff| = 0.000e+00` | pass |
| 2 | a-slice zeroed → A must vanish | `arm A max|diff| = 0.000e+00` | pass |
| 3 | permutation is non-vacuous | `moved 100.0%` of rows | pass |
| 4 | two identical invocations | bit-identical output | pass |
| 5 | **B(1) predictable from one forward pass** (new) | warm: pred median 24.93 vs measured 23.92 (**1.4σ**); cold: 386.31 vs 382.95 (**0.25σ**) | pass |

Check 5 is the load-bearing one: it proves B(t=1) *is* the mixture's two-sample noise, with no
extra mechanism, and it retires the brief's "half-noise" reading (§4).
It also retires a data-source worry: `max|ck.latent_mean − index.json mean| = 0.000e+00`
(and likewise for std) ⇒ `E|dz|` and the z-normalisation share one source.

## 3. Dream divergence curves

`--mdn runs/mdn_pilot/best.pt --rollout 0 --steps 200 --reps 50 --taus 1.0`.

`medRatio` (paired per rep, τ=1, `E|dz| = 0.05378`):

| t | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| warm (warmup 16) | 0.06 | 0.10 | 0.08 | 0.06 | 0.10 | 0.19 | 0.36 | 0.60 | **0.82** |
| cold (warmup 0) | 0.02 | 0.03 | 0.05 | 0.08 | 0.15 | 0.25 | 0.44 | 0.73 | **0.88** |

- **A/B < 1 at every grid point** (0.02–0.88), monotonically rising for t ≥ 16.
- Deterministic arm at t=200: **D = 27.08 (warm) vs 167.30 (cold)** — h=0 is an OOD state
  (flagged as observation, not explained).
- **The initial condition is forgotten by t=200**: `medB(200)` warm **176.77** vs cold
  **176.67** (0.06% apart). Both arms then measure the same steady-state two-dream distance.
- The warm column reproduces the brief's `real` column **to two decimals at all 6 shared t**
  (0.06 / 0.06 / 0.19 / 0.36 / 0.60 / 0.82) and B(1) 23.83 vs 23.92 (0.4%) ⇒ code and checkpoint
  identical; any brief/brief-session discrepancy is a *config* difference, not a code difference.

**Interpretation** (retires brief #16): A/B → 1 is *not* constructive. On the synthetic positive
control PC2 (`runs/mdn_pilot_pc2`, where a drives 68.7% of the step) the instrument reads
`medRatio(t=1) = 1.48` → implied variance share `r²/(1+r²) = 68.7%`, matching the construction
within 0.0 pt at t=1 (73.6% / 67.1% at two other points). The instrument's discriminative power
is therefore demonstrated, and PC2 **keeps growing to 3.21 at t=200** rather than converging to 1.

## 4. What the dream's noise actually is (teacher forcing, val, `--seed 0`)

Conventions: `nw ≡ Σ_k π_k ‖σ_k‖²` (π-weighted, per element), `sigma_min=1e-3`, `sigma_max=10`.

**Gating** (this is what retires brief #18's inference):

| quantity | measured | uniform ref |
|---|---|---|
| mean max-π | 0.6435 | — |
| mean entropy | 0.8127 | 1.6094 |
| Σπ² | **0.5260** | 0.20 |
| effective #components | **1.901** | 5.0 |

⇒ The mixture is *half concentrated* (≈1.9 effective components), neither collapsed nor uniform.
The brief's back-solved Σπ² ≈ 0.979 (`eff#comp ≈ 1.0`) is **void**; it was derived from a
`between mean sd 0.0281` that cannot coexist with `mean pairwise ‖μᵢ−μⱼ‖ = 1.5456`
(the identity `B = ½Σ_{k≠k'}π_kπ_{k'}‖μ_k−μ_{k'}‖²` demands B ≈ 0.566 in L2² units, vs
0.0281² = 0.00079 per element — off by 22× at the most favourable conversion). The components
are **separated**; what is not concentrated is the gating.

**σ distribution is heavy-tailed, and the tail is in the *sample* direction:**

| quantity | value |
|---|---|
| E[σ] | 0.11285 |
| E[σ²] | 0.17800 |
| rms = √E[σ²] | 0.4219 |
| max | **10.0000** (= σ_max ⇒ `clamp` fires in **forward**, not only in `sample`) |
| p50 / p90 / p99 / p99.9 | 0.02436 / 0.24654 / 1.34423 / 6.51202 |
| skew mean/median of `nw`, per (n,t) | t=0 **480×**, t=8 **76×**, t=16 5× |
| top-10% share of Σσ², by (n,t) | **0.9781** |
| top-6 share of Σσ², by dim | 0.357 (16 of 32 dims carry the other 0.64) |
| frac at σ_max (>9.9) | 6.06e-4 (≈34% of Σσ² lives in 0.06% of the elements) |
| frac at σ_min (≤1e-3) | 1.58e-4 |

⇒ σ² is **not** "a typical noise level"; it is a pure tail statistic. Any comparison against a
mean-derived prediction is invalid unless both sides are the same functional (mean↔mean, median↔median).

**σ is extremely time-localised:**

| | π-weighted `nw` |
|---|---|
| TF t=0 | 25.9221 |
| TF t=8 | 0.7303 |
| TF t=16 | 0.0424 |
| TF mean(t≥16) | 0.0597 |
| share of Σσ² in t<16 | **99.15%** |

⇒ The brief's σ² mean (0.109 → "B(1) ≈ 49.1") is dominated by the t=0–2 cold steps.
The dream instrument measures t=1 from a **warm** h (warmup 16), where `nw ≈ 0.06` (TF)
or 0.99 (ep0) — a population 16–23× larger than the dataset position-16 mean.

**Cross-component term is negligible:** `between share = 0.0253 / 3.4969 = 0.72%` (matches the
brief's independently computed 0.72%).
**Equal vs π weighting:** `5.6960 / 3.4969 = 1.63×` — π weighting *lowers* the noise norm²
(σ and π are positively correlated), so π weighting cannot explain the brief's 49.1 vs 24.6.

**Why the dream is not simply "teacher forcing with a warm h":** autoregressive rollout,
`nw` at t=0 is bit-identical to TF (25.9221) but then decays *below* TF:
`ratio self/TF = 0.37 at t=16, 0.21 at mean(t≥16)`. The dream's h has **smaller** per-step σ
than teacher forcing at the same index; its extra divergence comes from deterministic
amplification, not from extra injected noise.

**Exactness of B(1) (closes brief #22):** on the *same* forward tensor (ep0, τ=1),
- warm: `nw = 0.9880`, `bw = 0.0075` → predicted median **24.93**, measured **23.92** (1.4σ, SE≈0.74)
- cold: `nw = 228.4006`, `bw = 0.8836` → predicted median **386.31**, measured **382.95** (0.25σ)

⇒ `24.6` is simply this mixture's two-sample noise. There is no "half-noise". And `nw(ep0, pos 16)
= 0.9880` is 23× the dataset position-16 mean (0.0424) and 16.5× the warm-segment mean (0.0597) —
a single heavy-tail input, not a discrepancy.

## 5. Single-step action effect vs dream noise (independent estimator)

On 155 val windows, median of the per-step ratio (not the instrument's per-rep pairing):

| t | med ‖Δμ‖ (action permutation) | med two-sample noise | ratio |
|---|---|---|---|
| 0 (cold) | 0.0510 | 0.3590 | 1/7.0 |
| 8 | 0.0313 | 0.1371 | 1/4.4 |
| 16 (warm) | 0.0328 | 0.1350 | 1/4.1 |

Direction matches the instrument (warm is more controllable than cold) but the estimator differs
from `medRatio` (this one is a ratio of medians over windows with one permutation; `medRatio`
is a median of per-rep paired ratios on one episode). Quote the instrument's `medRatio` for
the headline; use this table only as an order-of-magnitude cross-check.

## 6. Cold-start, 5-rollout sweep (`--warmup 0 --reps 50 --steps 200`)

| rollout | B(1) mean | D(1) | medRatio |
|---|---|---|---|
| 0 | 376.76 | 2.40 | 0.02 |
| 1 | 394.72 | 3.00 | 0.03 |
| 2 | 402.55 | 2.42 | 0.02 |
| 3 | 380.73 | 2.44 | 0.02 |
| 4 | 369.77 | 3.11 | 0.03 |
| **mean ± sd** | **384.91 ± 13.42** (SE 6.00) | **2.674 ± 0.350** (SE 0.157) | **0.024 ± 0.0055** |

Cold/warm B(1) ratio = **15.3–17.4×** (brute: noise, not action amplification — D grows only 1.8×).

## 7. Retractions (continuing Note9's §7, which ended at #15)

| # | Retracted claim | Reason |
|---|---|---|
| 16 | "A/B → 1 is constructive" | PC2's A/B keeps growing to **3.21** at t=200; A/B → 1 only when the action does not drive the trajectory |
| 17 | "the dream's noise is dominated by mixture ambiguity" | `between share = 0.72%` (§4) |
| 18 | "the 5 component means collapse (pairwise distance 0.028)" | 0.028 was a π-weighted RMS deviation from the mean, not a pairwise distance; the real pairwise distance is **1.5456** ⇒ components are separated; the back-solved Σπ²≈0.979 is replaced by the direct **0.5260** |
| 19 | the handover's artefact inventory | `make_noaction_latents.py` and `--probe-only` do not exist; `--norm-min-std` is a dead flag |
| 20 | several order-of-magnitude predictions | noise-frac calibration, synth3 scale, gap ≤ 0.91, 2.6 h cost, 13×, `--probe-only`, MLP paired seed, ratio-1.5 criterion |
| 21 | "what collapsed inside the dream is the gate (Σπ²≈0.979, eff#comp≈1.0)" | direct measurement: Σπ² **0.5260**, eff#comp **1.901**, max-π 0.6435, H 0.8127 |
| 22 | "the dream's noise is half the teacher-forcing prediction (49.1 vs 24.6)" | **input mismatch**: 49.1 uses the all-timestep mean σ² (99.15% of which is t<16); 24.6 is one warm single input (ep0, `nw = 0.9880` = 23× that position's dataset mean). Check 5 (§2) reproduces 23.92 from the same tensor |
| 23 | "two-seed medB violates the Markov bound ⇒ the dream's σ is ≥2.6× teacher forcing" | the bound was applied to a **single-step** quantity using an **all-step** σ; the argument is void |
| 24 | "an 18× gap between measured med‖Δz‖(t=1)=0.3926 and √(2·mean nw)=7.2" | mean-vs-median mismatch: correct comparison is sim median **0.3590** vs measured **0.3926** (9%) |
| 25 | "the brief's cold-start triple (B(1)=395.90, D(1)=1.94, medRatio 1/204) is reproducible" | 5-rollout sweep gives B **384.91 ± 13.42** (395.90 = +1.83σ, borderline), D **2.674 ± 0.350** (1.94 = −4.68σ), medRatio **0.024 ± 0.0055** (reproducible). Also `1.94 / 395.90 = 1/204.0` exactly ⇒ the brief's cold "ratio" was `D(1)/B(1)`, not the paired median. Brief cold triple **superseded**; §3.2's `395.90` → `384.9 ± 13.4` and `1/204` → `1/50` (medRatio) / `1/42` (mean ratio) |

## 8. Protocol rules (additions to Note9 §8)

9. For heavy-tailed quantities report the **median** (`medRatio`), never the mean ratio; `medA/medB`
   and `medRatio` are different estimators (paired median vs median-of-marginals) and differ by
   more than the σ they are being tested against.
10. Keep `--reps` fixed: 20 and 50 differ by ~1.5× in the mid-horizon.
11. Do not compare an all-timestep aggregate against a single-input or warm-segment quantity.
    Always state the population (all steps / position t / episode k).
12. Report a mean-derived prediction as √(E[X²]) and compare it to the **RMS** measured value;
    compare medians to medians.

## 9. Verbatim source-line anchors

- `dream_controllability.py`: `PERM_BASE = 31337`
- `dream_controllability.py`: `    return a_seq[torch.randperm(a_seq.shape[0], generator=gen)]`
- `dream_controllability.py`: `def div_curve(x, ref, eabs):` / `    return (d / eabs).cpu().numpy()`
- `dream_controllability.py`: `    if a.warmup > 0:` / `        h0 = warm_hidden(model, z[:a.warmup], A[:a.warmup])` (guarded zero-h branch prints `[warn] warmup=0: h0 = zeros (metric_validation's t=0 condition)`)
- `dream_controllability.py`: `            r_med = np.median(av / np.maximum(bv, 1e-9))        # paired per rep` → the definition of `medRatio`
- `dream_controllability.py`: `def e_abs_dz(latents, k=2000):` / `    return float(np.abs(np.diff(z, axis=0)).mean())` → per-step convention, **not** the metric's `E|dz|`
- `mdn_rnn.py`: `    out, h = self.lstm(x, h)` (line 105) → `forward` requires a 3-D `(N,T,D)` input; single-step replay must keep the T axis
- `mdn_rnn.py`: `        log_sigma = log_sigma + 0.5 * math.log(temperature)` (inside `if mode in ("sqrt","both")`) → σ ∝ √τ, hence `D_a/D_s ∝ τ^(−1/2)` is an identity, not a finding
- `mdn_rnn.py`: `        log_sigma = log_sigma.clamp(math.log(c.sigma_min), math.log(c.sigma_max))` → **confirmed to fire in `forward`**: TF σ `max = 10.0000` exactly

## 10. Artefacts and cost

- New: `scripts/mdn/dream_controllability.py` (instrument).
- New this session: nothing else was needed; all readings come from `runs/mdn_pilot/best.pt` and
  `data/pilot_latents`.
- **Cost**: dream run (reps 50, steps 200) ≈ 1 min CPU per (rollout, warmup); cold 5-rollout
  sweep ≈ 5 min; the TF σ analysis ≈ 1–2 min per invocation.

## 11. Open

1. Brief §3.2's qualitative conclusions survive, but two numbers do not; keep only the corrected
   pair (`B(1) 384.9 ± 13.4`, `1/50`).
2. `D(200)`: 27.08 (warm) vs 167.30 (cold) — the cold deterministic response is 6× larger.
   Likely an OOD h=0 artefact; recorded as an observation, not explained.
3. An 18-pt grid was measured at τ=1 only in this session; the τ-scaling of A/B is an identity
   (anchor above), so no τ sweep is needed for A/B itself, but the τ-scaling of `D` was not measured.
4. Still un-run: `runs/mdn_pilot_shuf_s4`; and Note9's open items 1–3 are untouched by this note.
