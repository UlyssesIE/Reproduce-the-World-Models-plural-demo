# Note4 — MDN-RNN (M model): training, evaluation, and a retracted fix

**Status:** M accepted as-is (action-blind, a property of the data); controller (C) is next
**Date:** 2026-09-19
**Extends:** Note2 §6 (why the random-policy data is informationally poor), Note3 §8 (the open question about z)
**Retracts:** the `--norm-min-std` recommendation made while diagnosing M
**Follows:** Note3 §8 — V was accepted, the latent dataset was built, and M was trained on the pilot rollouts only.

---

## 1. Summary

M was implemented, trained and measured. Four things came out of it:

1. **M works in the sense that it predicts beyond `z_t`** — R² over the persistence
   baseline is **0.4724** — but it is **action-blind**: shuffling or randomising `a_t`
   changes the prediction by only **0.117** of the mean frame-to-frame change. The
   three action variants (`real` / `random` / `repeat`) produce **visually identical**
   dreams.
2. The cause is the dataset, not the implementation. Note2 §6 already quantified it:
   the car moves at 3.13 units/s (≈0.2 % of the frame per step) and the steering
   commands cancel out, so `a_t` carries almost no information about `Δz`. The paper
   anticipates this in §5 and offers iterative training as the remedy.
3. **A fix I proposed was wrong and is withdrawn.** Adding a floor to the latent
   normalisation divisor (`--norm-min-std`) does not do what the argument claimed:
   the per-dimension gradient is invariant to that divisor, and raising it reintroduces
   the scale spread normalisation exists to remove. Details in §8.
4. **A retrain intended to apply that fix never applied it**, because the argument
   `norm_min_std` was declared as a local variable rather than a parameter, so callers
   could not pass it and the floor silently evaluated to zero. The "retrained"
   checkpoint is bit-for-bit the original model.

Net effect on the project: **none of M's findings block the CarRacing result**, because
the paper trains C in the *real* environment and uses M only for a visualisation
(§3.4: "put our trained C back into this hallucinated environment"). M is closed here.

---

## 2. What the paper specifies, and what we chose

**Specified by the paper**

| Item | Value |
| --- | --- |
| Recursive model for M | **LSTM**, not GRU — "In the Car Racing task, the LSTM used 256 hidden units" |
| Hidden size (CarRacing) | **256** |
| Prediction target | `P(z_{t+1} \| a_t, z_t, h_t)` |
| Output distribution | a **mixture** of Gaussians, chosen deliberately so that discrete modes can be represented |
| `done` prediction | **not** part of the CarRacing variant (VizDoom only) |
| Temperature `tau` | used when *sampling* during controller training; `tau = 1` while training M |
| Reported parameter count | **422,368** for the CarRacing MDN-RNN |

**Not specified by the paper (our choices, to be reported as deviations)**

| Item | Our choice |
| --- | --- |
| Number of mixture components | **5** |
| σ parameterisation | `log σ` output, clamped to `[1e-3, 10]` |
| How `tau` enters the distribution | `sigma <- sigma * sqrt(tau)` (`tau_mode="sqrt"`) |
| Sequence length, optimiser | `T = 32` windows, Adam, lr 1e-3, grad-clip 100 |
| Latent normalisation | per-dimension standardisation using training-set statistics |

**Parameter-count gap.** Our M has **383,557** parameters against the paper's
4,22,368 — a delta of **−38,811**. The paper does not state the number of mixture
components or the head architecture, so the count cannot be reproduced exactly. As
with V in Note3 §9, this is recorded rather than concealed.

---

## 3. Implementation

| file | role |
| --- | --- |
| `src/models/mdn_rnn.py` | `MDNRNN`: LSTM + linear head emitting `(pi, mu, log_sigma)`; `nll()` via `logsumexp`; `sample()` with temperature |
| `src/data/sequences.py` | `LatentSequenceDataset`: windows that never cross a rollout boundary |
| `scripts/mdn/train_mdnrnn.py` | training loop with per-dimension NLL reporting |
| `scripts/mdn/mdn_probe.py` | post-hoc diagnostics on a checkpoint |
| `scripts/mdn/dream_rollout.py` | autoregressive dream through M, decoded by V |

Two details worth recording.

**Boundary safety.** Windows are drawn per rollout, so M is never asked to predict the
first latent of one episode from the last latent of another. The train/val split is by
*rollout*, not by window, so no rollout contributes to both sides. Validation uses
non-overlapping windows (stride = `seq_len`), so its windows are not mutually
correlated.

**RNG hygiene in `dream_rollout.py`.** `np.asarray` on a memmap slice returns a
read-only view and `torch.from_numpy` warns on those:

```
UserWarning: The given NumPy array is not writable ...
```

Fixed by copying: `np.array(A[s0:s0+T], dtype=np.float32)`.

---

## 4. Three silent failures in the wiring

The first two are the same genre as the `np.memmap` pickle bug in Note3 §3: nothing
raises, the code runs, the output looks plausible, and the configuration is not what
you think it is.

### 4.1 `norm_min_std` declared as a local variable (retrain was a no-op)

```python
def __init__(self, ..., normalize: bool = True):     # no such parameter
    ...
    norm_min_std: float = 0.0                        # a local, always 0.0
    self.norm_min_std = norm_min_std
    if getattr(self, "norm_min_std", 0.0) > 0:       # never True
        std = np.maximum(std, self.norm_min_std)
```

Because the name was absent from the signature, a caller passing
`norm_min_std=0.1` would have raised `TypeError` — so the training script simply did
not pass it, the local stayed at zero, and the floor never applied. A second
contributor was `getattr(self, "norm_min_std", 0.0)`: reading an unassigned attribute
through a default turns "not configured" into "configured as off", which is exactly the
failure mode that hides itself.

**Evidence that the retrain did nothing**

```
[norm]  stats_source=checkpoint  norm_min_std=not recorded
[data]  {... 'norm_min_std': 0.0, 'std_min': 0.01446 ...}
```

`not recorded` means the checkpoint lacks the key the new training script is supposed
to write, so `runs/mdn_pilot_norm/best.pt` was written by the *old* script. Every
diagnostic value matches the original run exactly — total NLL −71.21, MSE 0.043194,
action ratios 0.117 / 0.122 — which for a deterministically seeded run means the
weights are identical. The two runs are the same model.

**Resolution:** the parameter is now declared in the signature and assigned explicitly,
the `getattr` default is gone, and `stats()` reports `norm_min_std` plus the effective
`std` range so a mis-wiring is visible at a glance. (The *fix* it was meant to enable is
withdrawn anyway — see §8.)

### 4.2 The probe ignored the checkpoint's normalisation statistics

`train_mdnrnn.py` stores the statistics it used:

```python
"latent_mean": tr.mean.tolist(), "latent_std": tr.std.tolist(),
```

but the first version of `mdn_probe.py` constructed its dataset from `index.json`
instead:

```python
ds = LatentSequenceDataset(args.latents, args.seq_len, "val", seed=args.seed)
```

If training had used a transformed divisor while the probe used the raw one, the probe
would have evaluated the model out of distribution — and reported a number that looks
like any other number. The fix adds a `stats=` argument that overrides the file, and
the probe now prints its source:

```
[norm]  stats_source=checkpoint  norm_min_std=not recorded
```

### 4.3 Preemptively fixed: memmap pickled through the spawn pipe

`LatentSequenceDataset` now opens its arrays lazily (`_open()`) and drops them in
`__getstate__`, re-opening per worker in `__setstate__`. This is the Note3 §3 bug in a
different container. It was harmless for the pilot latents (6.4 MB) but would have
pushed **128 MB** (2 M frames × 32 dims × float16) through a Windows pipe for each
worker once the full dataset was encoded.

---

## 5. Training

```
python scripts/mdn/train_mdnrnn.py --latents data/pilot_latents --out runs/mdn_pilot \
    --epochs 30 --batch-size 64 --seq-len 32 --lr 1e-3
```

| item | value |
| --- | --- |
| data | 95 train rollouts / 5 val rollouts, `seq_len = 32` |
| model | LSTM 256 units, 1 layer, K = 5 mixtures, 383,557 params |
| wall clock | **583 s** for 30 epochs |
| final `[val]` NLL | **−81.49** |

The reported NLL is far below the unit-variance-Gaussian reference of
`0.5·ln(2πe) = 1.419` per dimension (45.4 for 32 dims), i.e. per-dimension −2.55.
Inverting `NLL = 0.919 + ln σ` gives **σ ≈ 0.02** in normalised units.

**This number is not by itself evidence of anything.** Consecutive latents differ by
only `mean |z_{t+1} − z_t| = 0.0616`, so an LSTM that tracks a smooth trajectory can
reach small σ by extrapolation alone. Distinguishing "learned the dynamics" from
"copies `z_t` confidently" is exactly what `mdn_probe.py` is for.

Note also that the probe samples a single batch of 64 windows from the 155 available,
whereas training averaged the whole validation set; hence the discrepancy between
−81.49 and the −71.21 reported in §6. See §10.

---

## 6. Diagnostics

```
python scripts/mdn/mdn_probe.py --mdn runs/mdn_pilot/best.pt --latents data/pilot_latents
```

Batch: B = 64, T = 32, drawn from 155 validation windows. Raw latent std spans
**0.0145 – 1.0240**; **6 of 32 dims** exceed 0.3 and are labelled ACTIVE.

### [1] Per-dimension NLL

```
total = -71.21   per-dim mean = -2.225
reference: unit-variance Gaussian = 1.419 per dim (45.41 for 32 dims)

  dim 24   nll -1.903   raw_std 0.0221   amp.
  dim  2   nll -1.910   raw_std 0.0155   amp.
  dim  7   nll -1.919   raw_std 0.0237   amp.
  dim 25   nll -1.919   raw_std 0.0179   amp.
  dim 29   nll -1.925   raw_std 0.0202   amp.
  ...
mean NLL  active = -2.965     amplified = -2.055
```

Two readings, in opposite directions, both worth recording:

* **Per dimension**, the six active dims are predicted *better* (−2.965) than the
  26 near-constant ones (−2.055). The amplified dims are harder precisely because
  dividing by their own small std magnifies whatever noise they contain.
* **By magnitude of the sum**, the 26 amplified dims contribute the larger share,
  because there are 26 of them. The exact share printed by the probe is invalid — see
  §9.

### [2] Fitted σ

```
mean = 0.1110   min = 1.00e-03   max = 10.0000   sigma_min = 1.0e-03
fraction at the floor: 0.1%
```

No overconfidence: only 0.1 % of σ values sit at `sigma_min`, so the low NLL is not an
artefact of σ collapsing to zero.

### [3] Prediction versus the trivial "copy `z_t`" baseline

```
mean |z_t+1 - z_t|         : 0.0616
MSE model                  : 0.043194
MSE persistence (copy z_t) : 0.081864
R^2 over persistence       : 0.4724
```

**This is the decisive positive result.** Had M degenerated to an identity map, R²
would be ≈ 0. At 0.4724 it predicts the next latent nearly twice as well as copying the
current one.

### [4] Action sensitivity

```
|mean(a_true) - mean(a_shuffled)| : 0.00724
|mean(a_true) - mean(a_random)  | : 0.00751
scale: mean |z_t+1 - z_t|           : 0.06164
ratio (shuffled / step)             : 0.117
ratio (random   / step)             : 0.122
```

Replacing the true action with a shuffled one, or with uniformly random values, moves
the predicted mixture mean by about **12 %** of the mean frame-to-frame change. M is
**action-blind**.

---

## 7. Finding: M is action-blind, and that is a property of the data

The visual check agrees with the numbers. `dream_rollout.py` was run with three action
regimes, all at `tau = 1.15`:

| run | actions fed to M |
| --- | --- |
| `--action real` | the true actions from the rollout |
| `--action random` | uniform samples from the action space |
| `--action repeat` | the first action repeated |

**All three produce visually indistinguishable dreams**: grey road and green grass
separated cleanly, the red car at the bottom of the frame, no visible motion. Because
`ratio ≈ 0.12`, M's predictions barely depend on which of the three it is given, so the
three images are, as expected, the same image.

The dream's apparent stillness is **not** evidence of a frozen M. Two separate facts:

* the real sequence is itself nearly static at this scale —
  `mean |frame_t+1 − frame_t| = 0.0095 ≈ 2.4/255`;
* the car is drawn at a **fixed screen position** because `ZOOM_FOLLOW` keeps the camera
  on the vehicle; car position therefore carries no motion information at all.

**Root cause**, already quantified in Note2 §6: the car moves at 3.13 units/s
(0.062 units per step, ≈0.2 % of the frame), and the steering joints (±0.4 rad, 3 rad/s)
cannot follow commands resampled at 50 Hz, so steering contributions cancel and net
steering ≈ 0. Under a random policy, `a_t` genuinely carries almost no information
about `Δz`. **M learning that actions barely matter is correct behaviour on this
dataset, not a defect in M.**

The paper anticipates exactly this in §5:

> "the tasks are relatively simple, so a reasonable world model can be trained using a
> dataset collected from a random policy. But what if our environments become more
> sophisticated? In any difficult environment, only parts of the world are made
> available to the agent only after it learns how to strategically navigate through its
> world."

and offers iterative training as the remedy. Retraining M on the same random-policy
data cannot change ratio 0.12.

---

## 8. Retraction: the `--norm-min-std` fix was wrong

While diagnosing the per-dimension breakdown I proposed flooring the normalisation
divisor, arguing that the 26 near-constant dims were amplified 23–70× and "absorbed 75 %
of the NLL". **That argument is invalid, on two counts.** I worked the gradient before
defending it further.

**(a) The per-dimension gradient is invariant to the divisor.** For a Gaussian, the
mean term contributes `(r/σ)²/2` with `r = μ − target`. With normalisation
`r_norm = r_raw/d` and a model σ that scales with the data, `σ_norm = σ_raw/d`:

```
∂/∂θ [ ½ (r_norm/σ_norm)² ] = (r_norm/σ_norm²)·(∂r_norm/∂θ)
                            = ((r_raw/d) / (σ_raw/d)²)·(1/d)·(∂r_raw/∂θ)
                            = (r_raw / σ_raw²)·(∂r_raw/∂θ)
```

The divisor cancels. Changing `d` therefore does **not** change which dimensions the
shared LSTM is pushed to fit; it mainly rescales the reported NLL and, to second order,
the mixture-weight preferences. "Dilution of the objective" was not a real effect.

**(b) Raising the divisor worsens conditioning.** Plain normalisation sets every
target dimension to unit variance, which is the point. With `d = raw_std`, an active dim
whose raw std is 1.0 keeps normalised variance 1; flooring `d` at 0.1 for a dim whose
raw std is 0.015 gives normalised variance `(0.015/0.1)² = 0.0225`, i.e. 44× smaller
than an active dim. That reintroduces exactly the scale spread normalisation exists to
remove, forcing the network to emit μ values whose scales differ by more than an order of
magnitude.

**Decision:** plain normalisation is correct and already in use. The `norm_min_std`
code path is retained but defaults to 0 (the paper's behaviour) and is not used. No
retraining is required, and the existing `runs/mdn_pilot/best.pt` stands.

### 8.1 A related instrument error

The probe's "NLL share" line is also invalid:

```
NLL share, 6 active dims    : -1778947939368960.0%
```

`dim_nll.sum()` is negative (−71.21) and the guard `max(sum, 1e-12)` clamps it to
`1e-12`, producing an astronomically large ratio. (Amusingly, the original version's
plain ratio of two negatives printed a numerically sane 25 % — the "fix" for the sign
introduced the blow-up.)

More importantly, **a share of total NLL is scale-dependent** and therefore not
comparable across normalisations even when computed correctly, so it was never a valid
basis for the argument in §8. It is replaced by the **scale-invariant per-dimension
R² over persistence**:

```python
resid = ((pred - zt) ** 2).mean(dim=(0, 1))
base  = ((z - zt) ** 2).mean(dim=(0, 1))
r2_dim = 1.0 - resid / base          # comparable across normalisations
```

---

## 9. Known limitation of the probe

`mdn_probe.py` evaluates a **single batch** (`next(iter(loader))`, B = 64 of 155
available validation windows), which is why §6 reports −71.21 while training reported
−81.49. The numbers are self-consistent for the subset they describe, but the script
should aggregate over the whole validation loader before its values are quoted
alongside training logs. The per-dimension R² and the action-sensitivity ratio are less
sensitive to this than the absolute NLL, since both are ratios computed on the same
batch.

---

## 10. Why this does not block the CarRacing result

The paper's two experiments differ in where the controller is trained:

| experiment | C trained in | role of M |
| --- | --- | --- |
| CarRacing (§3) | the **real environment**, by CMA-ES | §3.4 only: "put our trained C **back into** this hallucinated environment, generated by M" |
| VizDoom (§4) | inside the dream | decides success or failure |

The CarRacing score is therefore determined by the quality of V's representation and by
C's evolution in the real environment — **not by M**. An action-blind M degrades the
quality of the dream visualisation and nothing else on this track.

**Consequence for planning:** M is closed. Its two measured properties (R² over
persistence 0.4724, action sensitivity 0.117) are recorded, the withdrawn fix is
documented, and the next module is the controller.

If a "train inside the dream" experiment is ever attempted on CarRacing, ratio 0.117
means M must be improved first — and the only lever that addresses the cause is a
better *dataset* (a driving policy that actually steers), i.e. the paper's iterative
training, not further tuning of M.

---

## 11. Hypothesis ledger (extended)

Continuing from Note3 §10.

| Version | Hypothesis | Status | What overturned it |
| --- | --- | --- | --- |
| v12 | M is degenerate: low NLL comes from copying `z_t` | **wrong** | R² over persistence = 0.4724 |
| v13 | the 26 amplified dims absorb 75 % of the NLL, so the objective is diluted | **wrong** | the per-dim gradient is invariant to the normalisation divisor (§8a); the share metric is also invalid (§8.1) |
| v14 | flooring the normalisation divisor will concentrate the objective | **wrong** | raising the divisor worsens conditioning (§8b) |
| v15 | the `--norm-min-std` retrain differed from the original model | **wrong** | checkpoint lacks the key; every value identical to the first run |
| v16 | the probe can be run with a different normalisation from training | **wrong** | it read `index.json` instead of the checkpoint; fixed with `stats=` |
| v17 (current) | M predicts beyond `z_t` but is action-blind, because the random-policy data makes `a_t` uninformative; M does not affect the CarRacing score | source-verified + measured | — |

Retained deliberately: v13 is the clearest example in this log of a *quantitative*
argument being wrong for a mathematical reason rather than a measurement reason. Two
fixes, two retrains and two rounds of review were spent on it.

---

## 12. Artifacts added in this note

| file | purpose |
| --- | --- |
| `src/models/mdn_rnn.py` | `MDNRNN`: LSTM + Gaussian mixture head, `nll()`, temperature `sample()` |
| `src/data/sequences.py` | `LatentSequenceDataset` with the bounded-stats fix; pickle-safe memmaps; reportable `stats()` |
| `scripts/mdn/train_mdnrnn.py` | M training loop; per-dimension NLL printed every epoch |
| `scripts/mdn/mdn_probe.py` | post-hoc diagnostics: per-dim NLL and R², σ concentration, persistence baseline, action sensitivity |
| `scripts/mdn/dream_rollout.py` | autoregressive dream decoded through V, saved as a strip; `--action` and `--tau` sweeps |
| `runs/mdn_pilot/` | M checkpoint (383,557 params) and `log.jsonl` |

Also fixed in passing: the non-writable-array warning in `dream_rollout.py`, and the
`norm_min_std` wiring in `sequences.py`.

---

## 13. Disclosure

* **M trained on 100 random-policy rollouts only** (95 train / 5 val rollouts,
  `seq_len = 32`, non-overlapping validation windows). The paper reports **10,000**
  rollouts; this reproduction uses a frame-budgeted subset, and any comparison against
  the paper's numbers must state that.
* **Architecture choices not specified by the paper:** 5 mixture components,
  `log σ` clamped to `[1e-3, 10]`, `tau` applied as `σ·√tau`, Adam at lr 1e-3 with
  gradient clipping at 100.
* **Parameter count:** 383,557 against the paper's **422,368** (delta −38,811).
  The paper does not specify the mixture count or the head, so the count cannot be
  reproduced exactly. Printed on every training run.
* **M does not predict `done`**, matching the paper's CarRacing variant (`predict_done=False`).
  Only the VizDoom variant predicts `done`.
* **Action sensitivity = 0.117** must be disclosed in any claim about training a
  controller inside M on this dataset. It reflects the random-policy data
  (Note2 §6), not the architecture, and it is the quantity that iterative training is
  expected to move.
* **The retracted `--norm-min-std` fix** is recorded here rather than removed, together
  with the reason it was wrong. The code path remains available but defaults to the
  paper's plain normalisation.
* **Hardware:** all timings are from an RTX 3060 **Laptop** GPU with 6 GB, not a
  desktop 3060.
