

# Note5 — Controller (C), and the resolution of the R²-over-persistence dispute

**Status:** M re-closed with corrected evidence; C (V-only, 99 params) trained and measuring **525.15 ± 165.50**; 867-parameter C next
**Date:** 2026-09-20
**Extends:** Note4 (M training and the retracted `--norm-min-std` fix)
**Corrects:** Note4 §1, §6 (the 0.4724 claim), §9 (the single-batch remedy), §10 and ledger v12
**Follows:** Note4 closed M and named the controller as the next module

---

## 1. Summary

1. **The dispute was not a code change.** `R² over persistence = 0.4724` (Note4 §6) and `−0.1936` (the probe's whole-split output) came from **the same `mdn_probe.py`**, differing only in the evaluation subset: `--batch-size 64 --max-batches 1` (64 of 155 validation windows) versus all 155. Re-running the former reproduces **all eight** of Note4 §6's numbers bit-for-bit, including 0.4724, 0.117, 0.122 and σ = 0.1110.
2. **The pooled R² is the wrong instrument, and Note4 §6's "decisive positive result" is retracted.** In normalised space the pooled mean-square of `Δz` has RMS/MAE ≈ **4.6–9.0** where any ordinary distribution gives **1.25–1.7**, i.e. it is dominated by a handful of rare large excursions in the near-constant dimensions. The number also moves from **+0.47 to −0.19** purely by changing which windows are sampled.
3. **M's corrected verdict is positive but different in kind.** Per dimension, the conditional density beats a **persistence-Gaussian on 32/32 dimensions** (ΣΔNLL = **−94.2**); on the six information-carrying dims the pooled explained-variance against a constant mean is **0.906**; the point estimate (mixture mean) is not better than copying `z_t` on those dims, mainly because of one heavy-tailed dimension (dim 13). M is action-blind (ratio 0.111), a property of the data (Note4 §7, Note2 §6).
4. **C was implemented and the first (V-only, 99-parameter) arm trained.** 60 generations, 8.35 h: random-policy baseline **−32.78** → **525.15 ± 165.50** over 100 rollouts. The paper's V-only ablation on CarRacing-v0 reports 632 ± 251; same order, different environment version (see §17).
5. **Four integration failures of the same silent-failure genre as Note3 §3 and Note4 §4** are recorded in §9, plus one real bug I introduced (final evaluation scheduled after the worker pool was destroyed, so `eval.json` was never written).
6. **The machine is the binding constraint.** Rollout throughput is **≈ 250–260 steps/s** on this laptop under any worker count: the bottleneck is `env.step` (pygame rendering, ~11.5 ms of 13 ms per step), not the networks. Running VAE/M on the GPU does not help (§12).

---

## 2. Resolution of the dispute

### 2.1 The chain of evidence

| Query | Result | Consequence |
| --- | --- | --- |
| `git diff HEAD -- scripts/mdn/mdn_probe.py` | empty | **inconclusive** — `git status` shows the file as `??` (untracked), so the diff is empty by construction |
| `git ls-files scripts/mdn/` | lists only `dream_rollout.py`, `train_mdnrnn.py`, `z_dynamics_check.py` | `mdn_probe.py`, `runs/` and `notes/Notes4.md` are all untracked |
| `git diff -- scripts/mdn/train_mdnrnn.py` | 4 hunks, **none involving R²** (`import math`, `parents[1]→parents[2]`, `--norm-min-std`, a blank line) | `train_mdnrnn.py` **has never computed an R²** |
| `Select-String` over `*.py/*.json/*.jsonl/*.md` for `0.4724` | one hit: `notes/Notes4.md:253` | the only surviving record of the number is the note itself |
| `runs/mdn_pilot/log.jsonl` head | `{"epoch","step","split","nll"}` only | training never logged an R² |
| timestamps | `Notes4.md` 01:57:21 → `mdn_probe.py` **02:03:11** → `probe.json` 02:03:19 | the note was written first, the whole-split probe ran 8 s after the script was touched |

**Conclusion:** there was no hidden edit to hide. The script's modification time precedes its own output; the note's 0.4724 predates both.

### 2.2 Bit-for-bit reproduction of Note4 §6

```powershell
python scripts\mdn\mdn_probe.py --mdn runs\mdn_pilot\best.pt --split val --batch-size 64 --max-batches 1
```

| quantity | Note4 §6 | this run |
| --- | --- | --- |
| `mean |z_{t+1} − z_t|` | 0.0616 | **0.0616** |
| MSE model | 0.043194 | **4.319403e-02** |
| MSE persistence | 0.081864 | **8.186381e-02** |
| R² over persistence | 0.4724 | **0.4724** |
| ratio shuffled / random | 0.117 / 0.122 | **0.117 / 0.122** |
| σ mean / floor fraction | 0.1110 / 0.09 % | **0.1110 / 0.09 %** |

### 2.3 The same script, whole split

```powershell
python scripts\mdn\mdn_probe.py --mdn runs\mdn_pilot\best.pt --split val --max-batches 0
```

| quantity | 64 windows | 155 windows |
| --- | --- | --- |
| `mean |Δz|` | 0.0616 | 0.0561 |
| MSE model | 0.043194 | 0.304450 |
| MSE persistence | 0.081864 | 0.255087 |
| **R² over persistence** | **+0.4724** | **−0.1935** |

**Nothing about the model changed.** Note4 §9 correctly identified that the probe evaluated a single batch and said it "should aggregate over the whole validation loader". Doing exactly that produced a *negative* number — which is what exposed the metric rather than the sampling as the problem.

### 2.4 A third, unreproducible configuration (open)

`runs/mdn_pilot/probe.json` records a whole-split run (155 windows) whose values sit **×1.21** above the current default output on every scale-dependent quantity, while agreeing on the ratio-based R²:

| quantity | `probe.json` | current default (155 w) |
| --- | --- | --- |
| step | 0.0652734786 | 0.0561 |
| MSE model | 0.36860523 | 0.304450 |
| MSE persistence | 0.30882156 | 0.255087 |
| R² over persistence | **−0.19358647** | **−0.1935** |
| action ratio (shuffled) | 0.10256817 | 0.111 |

A single global rescale would leave `1 − MSE_m/MSE_p` unchanged, so the leading hypothesis is **a different normalisation divisor** (a different `stats` source) rather than a different window set. The file has no `stats_source` and no record of its arguments, so it cannot be resolved from the artifact alone; `probe.json` does carry a `latent_std` field, which would settle it by comparison against `index.json` (open item, §14).

**Defect:** the probe's JSON omits `argv`, the `--latents` path, and the checkpoint/data hashes, so *any* two of its outputs are ambiguous. This is why the dispute was expensive.

---

## 3. Why the pooled R² is the wrong instrument

### 3.1 RMS/MAE of the normalised increment

For any distribution of a comparable shape, `RMS/MAE = √(E[X²])/E[|X|]` lies near **1.253** (normal) and stays within **1.25–1.7** for light-to-moderate tails. Measured here:

| source | MSE | MAE | **RMS/MAE** |
| --- | --- | --- | --- |
| 64-window probe | 0.081864 | 0.0616 | **4.65** |
| 155-window probe | 0.255087 | 0.0561 | **9.0** |
| whole file, `latents.npy` | 0.4355 | 0.0883 | **7.5** |

Three independent configurations, all far outside the physical range. The pooled second moment is therefore **not a summary of the typical step**; it is a summary of the tail. The probe's own `[check]` line agrees: `max|z_{t+1} − z_t| = 8.837735` in **normalised** units, i.e. an **~8.8σ** event, enough to dominate an average taken over 155 × 32 values.

### 3.2 Variance concentration

From `z_dynamics_check.py` (ran in this session, whole file, 100 rollouts):

| group | dims | share of total latent variance |
| --- | --- | --- |
| dims 1, 4, 8, 9, 28 (+13) | 6 | **99.7 %** |
| all remaining | 26 | **0.34 %** |

Per-dimension std spans **0.0145 → 1.0240** (70×). Dividing each dim by its own std amplifies the 26 small ones by up to 70×, so their **rare** excursions — irreducible quantisation and sampling noise in a dimension that barely moves — land in the pooled mean with full weight. **One such dimension can outweigh the other 31.** The same numbers confirm Note2's and `sequence.py`'s earlier claim independently: the 26 near-constant dims carry **61.5 / 80.5 = 76 %** of the total NLL (the docstring says 75 %).

### 3.3 Subset instability

R² over persistence is a **ratio of two pooled means**, so it is not scale-invariant in the presence of heavy tails: whichever windows happen to contain the excursions determine the answer. Observed range across three windows sets of the *same* checkpoint: **+0.4724, −0.1935, −0.1936**. A metric that flips sign under sampling is not a basis for accepting or rejecting a model.

---

## 4. Per-dimension decomposition (whole split, 155 windows)

```
micro (pooled)      R² = −0.1935      macro (mean of per-dim)  −0.2651     median +0.0427
```

| group | dims | Σresid | share of Σresid | mean per-dim R² |
| --- | --- | --- | --- | --- |
| collapsed, raw std ≤ 0.04 | 26 | 9.290 | **95.4 %** | −0.100 |
| information-carrying, raw std > 0.3 | 6 | 0.452 | 4.6 % | −0.978 |

The 6 "active" dims are themselves dominated by one outlier:

| set | pooled R² |
| --- | --- |
| all 6 active dims | **−3.31** |
| dropping dim 13 | **+0.121** |
| dropping dims 13 and 1 | **+0.297** |

Two mechanisms, both visible per dimension:

* **dim 13** — `resid 0.413` against `persist 0.061`, giving R² = **−5.79**, while its NLL (−3.512) is the *best in the whole table*. That is the signature of a mixture density that is confident and usually right, with a small number of very large jumps: the log-likelihood likes it, the squared error hates it.
* **dim 1** — persistence already explains **99.4 %** of the within-rollout variance (inertia 0.994), so copying `z_t` is nearly exact and any model error scores as a large negative R². Using an almost-perfect baseline as denominator is guaranteed to produce this.

**Change of denominator fixes it.** Against the constant per-dimension mean (explained variance, EV):

| dim | 1 | 4 | 8 | 9 | 13 | 28 | pooled, 6 active | pooled, 4 best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EV | 0.987 | 0.978 | 0.997 | 0.997 | **−1.83** | 0.996 | **0.906** | **0.993** |

**Scale-free distributional test.** Comparing the model's density to that of the persistence-Gaussian, per dimension:

```
Σ ΔNLL vs persistence-Gaussian = −94.2      dims better = 32/32
   active 6 dims: −13.6 (6/6 better)        collapsed 26 dims: −80.6 (26/26 better)
```

ΔNLL is a ratio of likelihoods and therefore invariant to any per-dimension rescaling. This is the strongest statement available about M: **its conditional distribution beats copying-with-correct-uncertainty on every single dimension.**

---

## 5. Corrected verdict on M (replaces Note4 §6 and ledger v12)

> **M is accepted.** The instrument used to accept it in Note4 §6 is withdrawn; the conclusion survives on different evidence.
>
> * **R² over persistence is discarded as a headline metric.** 95.4 % of its numerator comes from 26 dims that hold 0.34 % of the variance, and it moves from +0.4724 to −0.1935 as a function of sampling. Both the Note4 §6 value and the whole-split value are recorded here, and neither is quoted as a result.
> * **What replaces it:** (i) pooled **EV against a constant mean = 0.906** on the six information-carrying dims (0.993 on the four cleanest); (ii) **ΣΔNLL vs persistence-Gaussian = −94.2 with 32/32 dimensions better**; (iii) the action-sensitivity ratio **0.111**, unchanged in meaning.
> * **The point estimate (mixture mean) is not better than copying `z_t` on the active dims when pooled** (−3.31, driven by dim 13) and *is* better on four of the six (+0.297). Both facts hold; the mixture mean is a point estimator of a multi-modal density and is known not to be the NLL-optimal summary. This is a property of the estimator, not evidence of a degenerate M.
> * **dim 13 is recorded as a named outlier** (R² −5.79, best NLL in the table) rather than averaged away.
> * **Action-blindness is unchanged** and remains a property of the random-policy data (Note4 §7, Note2 §6).

---

## 6. Corrections to Note4

| Note4 location | Claim | Correction |
| --- | --- | --- |
| §1.1, §6 | "R² over persistence is **0.4724** … the decisive positive result" | it is a **64/155-window subset** estimate of a metric dominated by 0.34 %-variance dims. Retracted; see §5 |
| §9 | "evaluates a single batch … the script should aggregate over the whole validation loader before its values are quoted" | done — the whole-split value is **−0.1935**. The remedy was correct and *insufficient*: the metric, not the sampling, is the problem |
| §8.1 | "replaced by the **scale-invariant** per-dimension R² over persistence" | true of the **per-dimension** form; **not** true of the pooled form that the note then quotes. The pooled form belongs to the same class of instrument error as the invalid NLL-share line it replaced |
| §10, ledger v12 | "M is degenerate → **wrong**, because R² over persistence = 0.4724" | the verdict (M is not degenerate) stands; **the cited evidence is replaced** by EV 0.906 and ΣΔNLL −94.2 |
| §3, §12 | cites `src/data/sequences.py` | the file is `src/data/sequence.py` (singular) |
| §12 | `sequences.py` holds "the bounded-stats fix" | `sequence.py`'s module docstring still advertises `norm_min_std = 0.1` as "recommended", contradicting the Note4 §8 retraction — **doc fix outstanding** (§14) |

---

## 7. Retractions of my own claims in this session

Recorded in the same spirit as Note4 §11. The first five were overturned by the user's measurements, in the order they occurred.

| # | claim | what overturned it |
| --- | --- | --- |
| 1 | the seven reported numbers "are not probe output" | they are exactly `probe.json`'s fields |
| 2 | mean-of-ratios vs pooled is "a pseudo-problem" | they can take opposite signs when `persist` varies across dimensions |
| 3 | 0.4724 = the `within / total` variance share from `z_dynamics_check.py` | measured share = **82.0 %** |
| 4 | the macro (mean per-dim) R² would be ≈ +0.47, proving 0.4724 was the macro value | measured **−0.2651**, median +0.0427 |
| 5 | `MSE` is computed in normalised space while `step` is in raw space | both are normalised (`step 0.0561` ≈ normalised MAE) |
| 6 | float16 reduction is "a mandatory fix in the repository" | `index.json` statistics are float64 and the datasets cast on read; only my own one-liner was affected |
| 7 | `runs/mdn_pilot_norm` differs from `runs/mdn_pilot` (log file twice the size) | `log.jsonl` is opened in append mode; the 2× is a re-run, and the 64-byte checkpoint difference is consistent with an extra `args` field. **Downgraded to undetermined** |
| 8 | `ctor=VAE` / `ctor=MDNRNN`, `make_env`, `resolve_frame_preprocessor`, `data/pilot/*.npz` | all four were guesses; the checkpoint holds a `state_dict` under `model` with a `cfg` dict, `make_env` lives in `src/data/collect.py`, the preprocessor is `preprocess_frame`, and the rollouts are in `data/raw/` |
| 9 | the final evaluation call site | **a real bug I wrote** — the pool was destroyed before the final evaluation ran (§9.5) |
| 10 | the A/B timing protocol | it did not suppress the final 100-rollout evaluation, so it timed the wrong thing (§11.3) |

---

## 8. The controller module

### 8.1 What the paper specifies (fetched in this session)

| Item | Specification |
| --- | --- |
| Form | `a_t = W_c [z_t; h_t] + b_c` — single linear layer, **867 parameters** |
| Input for CarRacing | `z_t` **and the LSTM output vector `h_t` only** — the cell vector `c_t` is added for VizDoom, not here |
| Output mapping | tanh, then bound to the action ranges: steer ∈ [−1, 1], gas ∈ [0, 1], brake ∈ [0, 1] |
| Optimiser | CMA-ES, **population 64**, each candidate evaluated on **16 rollouts** with different seeds, fitness = mean cumulative reward |
| Evaluation cadence | best individual evaluated on **1024 rollouts every 25 generations** |
| Reported outcome | after **1800 generations**, 900.46 over 1024 rollouts; **906 ± 21** over 100 |
| V-only ablation | `a_t = W_c z_t + b_c` (**99 parameters**): **632 ± 251** |
| V + 40-tanh hidden layer | **1443 parameters**: **788 ± 141** |
| Data volume | **10,000** random-policy rollouts |

### 8.2 Implementation

| file | role |
| --- | --- |
| `src/models/controller.py` | `Controller`: linear `[z; h] → 3`, tanh + affine bound, `get_flat`/`set_flat` for CMA-ES, `check_spec()` asserting 867 |
| `src/rollout.py` | `Agent` (V encode → normalise → M hidden step → C act) and `run_episode`; env factory taken from `src.data.collect.make_env` |
| `scripts/controller/train_controller.py` | CMA-ES driver: spawn pool, per-generation logging, periodic and final evaluation, `--no-hidden` / `--eval-only` |

The rollout loop, matching how M was actually trained:

```
z_t      = VAE.encode(x_t).mu                 deterministically, no reparameterisation
z_norm   = (z_t − mean) / std                 mean/std taken from M's checkpoint
a_t      = C(z_norm, h_t)                     tanh + affine, clipped to the action space
x_{t+1}  = env.step(a_t)
h_{t+1}  = M(z_norm, a_t, h_t)[−1]            h reset to zeros at the start of every rollout
```

### 8.3 Two design points worth recording

**`z` vs `z_norm` is not a choice.** C is affine and the normalisation is a per-dimension affine map, so the two parameterisations generate *the same function class* under the bijection `W → W·diag(s)`, `b → b + W·m`. Feeding C the normalised `z` is therefore not a deviation from the paper; it is the same model with a better-conditioned coordinate system for CMA-ES (raw requires weight entries up to 70× larger to compensate for the std spread).

**Only `h`, never `c`.** Conflating the two would double the input width to 544 and the parameter count to 1635, silently breaking the comparison against the paper's 867.

---

## 9. Integration failures (same genre as Note3 §3 / Note4 §4)

None of these raised an error at the point of the mistake.

**9.1 `torch.load` returns a dict, not a module.** Both checkpoints are `{"model": state_dict, "cfg": {...}, "args": {...}}` (M additionally stores `latent_mean` / `latent_std`). The first loader attempted `ckpt.eval()`. Correct form, taken from the repository's own `encode_latents.py`:

```python
ck = torch.load(path, map_location=dev, weights_only=False)
model = VAE(VAEConfig(**ck["cfg"])).to(dev).eval()
model.load_state_dict(ck["model"])
```

**9.2 Guessed symbols.** `ctor=VAE`, a nonexistent `make_env` export, a speculative `resolve_frame_preprocessor` import chain, and `data/pilot/*.npz`. All four were resolved by inspecting the modules rather than by guessing again.

**9.3 The rollouts are in `data/raw/`, not `data/pilot/`.** `index.json` lists bare filenames, so the pipeline verification initially failed with `IndexError`. Fixed by locating the file by name.

**9.4 `signal.SIG_IGN` was installed too late in the worker.** A `Ctrl+C` landing before the initializer runs kills the worker while the parent is blocked in `pool.map`, and the parent then waits forever. Mitigation in place: never `Ctrl+C` a spawn-pool run — launch with `Start-Process` and close the window. Permanent fix outstanding (§14).

**9.5 Final evaluation scheduled after `pool.terminate()` — a real bug I wrote.** `evaluate()` calls `pool.imap_unordered`, but the teardown block ran first, so the run ended with `ValueError: Pool not running` and **`eval.json` was never written** after 8.35 h of training. The result itself was not lost (the gen-59 periodic evaluation had already used the same θ and the same seed set), but the artifact was. Fixed by moving the final evaluation inside the `try` and leaving only teardown in `finally`; `--eval-only` was added so a saved `best_theta.npz` can be re-evaluated without retraining. Recovered:

```
FINAL  525.15 +/- 165.50 over 100 episodes (99 params)
```

---

## 10. Verification before any training was trusted

**10.1 `obs → z` reproduces the stored latents.**

```python
x = to_float(preprocess_frame(obs[k], 12, 64, False))          # (64,64,3) float32 in [0,1]
mu, _ = vae.encode(torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0))
```

| frame | `max |mu − latents.npy|` |
| --- | --- |
| 0 | 8.88e-4 |
| 1 | 9.19e-4 |
| 137 | 3.27e-4 |
| 999 | 3.14e-4 |

All at float16 rounding (the file is stored as float16). **The live pipeline and the dataset agree.**

**10.2 The environment is reproducible.** Replaying `data/raw/rollout_000000.npz`'s stored actions through `make_env` reproduces its recorded reward to **d = 0.000**.

**10.3 The random-policy baseline is −32.78.** Recomputed directly from the `total_reward` field of the 2000 `data/raw` files, confirming Note1's figure and the environment's identity.

**10.4 The evaluation path is fully deterministic.** Within the Phase-1 log the evaluation values repeat **bit-for-bit** wherever `best_theta` has not changed:

```
gen 19 eval = gen 29 eval = 437.59250454493156
gen 39 eval = gen 49 eval = 494.8499701892896
```

Same seeds, same θ, identical float64 output to all printed digits.

---

## 11. Phase 1 — the V-only arm

```
python scripts/controller/train_controller.py --vae runs/vae_pilot2/best.pt \
  --mdn runs/mdn_pilot/best.pt --no-hidden --workers 8 \
  --popsize 16 --search-episodes 8 --eval-episodes 50 --eval-every 10 \
  --generations 60 --out runs/controller_vonly
```

| item | value |
| --- | --- |
| parameters | **99** (`--no-hidden`, matching the paper's V-only ablation) |
| generations | 60 |
| search | 16 candidates × 8 rollouts = 128 rollouts/generation |
| wall clock | **8.35 h** (501 s/generation) |
| random-policy baseline | **−32.78** |
| generation-0 policy (θ = 0: straight, half throttle) | −33.1 |
| search mean, gen 0 → gen 59 | −33.1 → **406.0** |
| **final evaluation, 100 rollouts** | **525.15 ± 165.50** |
| ratio std/mean | **0.315** (paper's V-only: 0.397) |
| paper's V-only ablation (CarRacing-v0, 100 rollouts) | 632 ± 251 |

**The learning curve had already saturated by generation 9** (periodic evaluation 495.18) and gained only ~5 points over the following 50 generations. That is the argument for spending the next run on the 867-parameter arm rather than extending this one: if `h_t` carries anything beyond `z_t`, it should show up in the same number of generations.

**Control condition for the next arm:** identical seeds (`20000..20099`), identical environment, identical V and M — **525.15 ± 165.50** with 99 parameters is the number to beat.

### 11.3 A protocol error worth recording

The A/B timing run used `--eval-every 999` expecting to skip evaluation, but the *final* evaluation still ran, so the 517.9 s wall clock measured ~370 s of post-training evaluation rather than search throughput. Throughput must be measured with a dedicated harness that runs neither CMA-ES nor a final evaluation; `tmp_bench_throughput.py` was written for this purpose and reports **steps/s** only.

---

## 12. Machine characteristics and the CPU/GPU decision

**Hardware:** AMD Ryzen 9 5900HX (8 physical cores, 16 logical), 45 W laptop part; RTX 3060 Laptop, 6 GB.

**Where the time goes** (single worker, 1000-step episode ≈ 13 s):

| stage | per step | note |
| --- | --- | --- |
| `env.step` (physics + pygame surface rendering) | **≈ 11.5 ms** | CPU only, single-threaded |
| frame preprocessing (PIL crop + bilinear resize) | ≈ 0.3 ms | CPU |
| VAE encode | **1.23 ms** (measured) | 64×64 input |
| M single-step LSTM + head | ≈ 0.15 ms | 256 units, batch 1 |

**Why the GPU is not used.** V+M+preprocessing is 10–25 % of a step, and at **batch size 1** on CUDA the kernel-launch and host↔device transfer overheads consume most of that. The argument is therefore not "we did not optimise it": the fraction of the step that is GPU-addressable is too small to matter, and each worker process would hold a CUDA context of **400–700 MiB**, against 6 GB total — capping the worker count at 3–4 and losing more than is gained.

**Measured throughput (steps/s, the only comparable unit):**

| workers | steps/s |
| --- | --- |
| 4 | 161 |
| 6 | 202 |
| 8 | 180 |
| 8 (repeat) | 228 |
| **Phase 1, in-training** | **~255** |

The spread between the two 8-worker runs (180 vs 228) is larger than the difference between worker counts: **the machine is saturated at ~250 steps/s and 8 workers is the ceiling.** The cause is a full-core frequency wall (45 W part: ~4.6 GHz single-core, ~3.3 GHz all-core) plus memory-bandwidth contention, not scheduling. Two free remedies: **plug in the power supply** and set the Windows power plan to best performance.

---

## 13. Data inventory findings

| observation | implication |
| --- | --- |
| `data/raw/` holds **2000** `rollout_*.npz` | only **100** are registered in `data/pilot_latents/index.json` — 20× more data is already on disk |
| `data/pilot_frames.index.json` exists but `data/pilot_frames.npy` does **not** | `encode_latents.py` will fail until `build_frame_cache.py` is re-run. C is unaffected (it needs only `pilot_latents`) |
| the paper's V-only arm used **10,000** rollouts | the pilot's 100 is a 1 % subset; any comparison against the paper must state this |

**This is the standing lever on action-blindness.** Note4 §7 identified the remedy for ratio 0.111 as *more informative data*, not more tuning of M. 2000 rollouts already exist; re-running `build_frame_cache.py` → `encode_latents.py` → `train_mdnrnn.py` is the only route to moving it, and it is now a data-engineering task rather than a research question.

---

## 14. Open items

| # | item | cost |
| --- | --- | --- |
| 1 | compare `probe.json`'s `latent_std` field against `index.json`'s `std` to settle §2.4 | one line |
| 2 | add provenance to the probe's JSON: `argv`, `--latents` path, checkpoint and data hashes, `stats_source` and the statistics themselves | small |
| 3 | print `micro / macro(median) / active-only` R² plus `ΣΔNLL` and EV in the probe; add an `assert` that checkpoint statistics match the data's | small |
| 4 | `sequence.py` docstring still says `norm_min_std = 0.1` is "recommended" — contradicts Note4 §8 | one paragraph |
| 5 | install `SIG_IGN` at worker import time, and give the parent a timeout-and-report loop instead of a blocking `map` (§9.4) | small |
| 6 | add `--skip-final-eval` (§11.3) | trivial |
| 7 | `README.md` is still 0 KB; `pyproject.toml` is absent (would end the `sys.path` juggling that produced Note4 §4.1's silent failure) | medium |

---

## 15. Hypothesis ledger (extended)

Continuing Note4 §11.

| Version | Hypothesis | Status | What overturned it |
| --- | --- | --- | --- |
| v18 | `mdn_probe.py` was modified, which is why R² differs from the note | **wrong** | running it with the old arguments reproduces all eight numbers bit-for-bit; the file is untracked, so `git diff` could never have decided it |
| v19 | 0.4724 comes from `train_mdnrnn.py` | **wrong** | that script's diff contains no R² code, and its log contains no R² field |
| v20 | 0.4724 is the `within / total` variance share | **wrong** | measured 82.0 % |
| v21 | 0.4724 is the macro (mean per-dimension) R² | **wrong** | macro = −0.2651, median +0.0427 |
| v22 | the pooled and per-dimension R² are interchangeable | **wrong** | they differ in sign; only the per-dimension form is scale-invariant |
| v23 | the near-constant dimensions merely "absorb 75 % of the NLL" and R² over persistence summarises the model | **wrong** | 95.4 % of the pooled residual and 76 % of the NLL, from 0.34 % of the variance; RMS/MAE 4.65–9.0 |
| v24 | the checkpoint's `latent_std` disagrees with `latents.npy` by 2–3× | **wrong** | that was float16 reduction in a one-liner; `resolve_stats` reports `max|dmean| = max|dstd| = 0.000e+00` |
| v25 | `runs/mdn_pilot_norm` is a different model from `runs/mdn_pilot` | **undetermined** | append-mode logging and a 64-byte checkpoint delta; not resolved |
| v26 | the GPU will shorten controller training | **wrong** | `env.step` is 11.5 of 13 ms; batch-1 CUDA gains ≈ 0 and costs worker count (§12) |
| v27 | the rollout pipeline needed fixing before C could learn | **wrong** | `obs→z` matches the stored latents to float16 rounding, replay is exact, and the baseline matches — C reached 525 with no pipeline change |
| v28 (current) | M's conditional density is better than persistence on every dimension, but the pooled R² metric is dominated by 26 near-constant dims; C (V-only) reaches ~525 and is saturated | source-verified + measured | — |

Retained deliberately: **v23** is the second instance in this log (after Note4's v13) of a *quantitative* argument that was wrong for a mathematical reason rather than a measurement reason — here, treating a ratio of pooled means as if it were an estimate of a typical relative error.

---

## 16. Artifacts added in this note

| file | purpose |
| --- | --- |
| `src/models/controller.py` | linear `[z_t; h_t] → a_t` controller, 867 params, tanh + affine bound, CMA-ES flat vector |
| `src/rollout.py` | `Agent` + `run_episode`; the exact V→normalise→M→C loop used in training |
| `scripts/controller/train_controller.py` | CMA-ES driver with spawn workers, periodic evaluation, `--no-hidden`, `--eval-only` |
| `runs/controller_vonly/` | 60-generation log, `best_theta.npz`, `eval.json` (525.15 ± 165.50) |
| `runs/mdn_pilot/probe_full.json` | whole-split probe output, including `r2_per_dim`, `resid_per_dim`, `persist_per_dim`, `latent_std` |
| `tmp_bench_throughput.py` | rollout throughput harness reporting steps/s, with no CMA-ES and no final evaluation |

---

## 17. Disclosure

* **C was trained in the real environment**, as the paper specifies for CarRacing. M contributes only the LSTM output vector `h_t`; no dreaming was used.
* **Budget.** The paper's configuration (population 64 × 16 rollouts × 1800 generations ≈ 1.84 M rollouts) would take **~35–53 days** at this machine's measured ~250 steps/s. The V-only arm used 16 × 8 × 60 ≈ 7,680 rollouts (0.4 % of the paper's). Generations and search episodes are therefore reported for every run; **the paper's 906 ± 21 is not claimed as reproduced.**
* **Environment version.** The paper reports CarRacing-**v0**; this reproduction uses gymnasium **CarRacing-v3** (`continuous=True`, `lap_complete_percent=0.95`, `max_episode_steps=1000`). Reward composition and termination semantics differ; scores are reported against **this** environment's own random-policy baseline of **−32.78**, not against 906.
* **V-only (99 params) = 525.15 ± 165.50** against the paper's 632 ± 251. Same order, different environment version and 0.4 % of the compute budget.
* **Data.** V trained and M trained on **100** random-policy rollouts; the paper used 10,000.
* **Action-blindness (ratio 0.111)** is disclosed for every controller result, since it bounds what any dream-based experiment could achieve (Note4 §10).
* **Hardware.** All timings from an AMD Ryzen 9 5900HX laptop and an RTX 3060 **Laptop** GPU with 6 GB — not i9-class CPUs or desktop GPUs. Rollout throughput is CPU-bound at ~250 steps/s.
* **The `R² over persistence = 0.4724` figure** remains in Note4 §1, §6 and §10 with its original wording; this note supersedes it. The number is retained here only as the record of a retracted measurement.
