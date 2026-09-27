# Note9 — Action channel: verdict

Date: 2026-09-26 · Supersedes every statement about M and a in Note4 §7, Note7, content.md and README "Deviation 3".
Verdict: **M is not action-blind; the a-channel is worth ≈1.5 nats, of which ≈1 nat is recoverable by retraining; ≈0.4–0.5 nats is irreducible.**

## 0. One line

| Claim | Verdict |
|---|---|
| "M ignores a / a has no influence" | **False** (response strong, stable, per-sample) |
| "a is worth nothing" | **False** (removal costs 1.5 ± 0.4 nats) |
| "a contributes materially" | **Weak** (irreducible share ≈0.4–0.5 nats ≈ 0.5%) |
| "action-blind" | **False** |

## 1. Canonical table

**Rule**: cross-run gaps per §7.1 (per-epoch paired means); same-run ablations are λ=0 − λ=1 differences in nats (a within-partition, deterministic quantity).

| Quantity | Value | Nature |
|---|---|---|
| NEW (own partition, 5 training seeds) | 2.905 / 3.727 / 2.348 / 2.987 / 3.075 → **3.008, cv 15.8%** | includes partition variation |
| NEW (fixed partition, 5 training seeds) | cv **2.5%** | training-seed component only |
| 5-shuffle-seed resolution | 0.65%–1.44% | within run |
| NEW ∝ λ | ratios 1.99–2.03 in 5/5 models | metric is exactly linear in a-slice scale |
| **Ablation, clean (5 training seeds, own partitions)** | 0.910 / 1.431 / 1.290 / 1.883 / 1.848 → **1.472 ± 0.406 nats**, SE 0.182, **t 8.1** | headline |
| Ablation, leakage-contaminated (fixed partition) | 1.746 / 2.186 / 2.184 / 2.158 (mean 2.069) | inflated by **+0.46 ± 0.28** |
| Ablation on the shuffled-a model | **0.019** | channel presence is free → content, not presence |
| **Training axis**: trained-with vs trained-without a, both channels inert, same partition | **0.400** (s0, deterministic) | irreducible component |
| Training axis, per-epoch paired (3 partitions) | −1.299 / −0.069 / −0.063 → RE −0.477 ± 0.411 | best.pt-selection-biased; corroborative |
| Mutual differences of the three nulls (s0) | 0.101 / 0.102 / 0.001 | <0.15σ |
| Real − shuf, per epoch (3 partitions) | −1.196 / −0.437 / −0.090 → RE −0.574 ± 0.327 | — |

**Decomposition**: removing a costs **1.47 ± 0.41**; retraining without it costs only **0.40** ⇒ retraining recovers **≈1.07 nats** and **≈0.4–0.5 nats is irreducible**.

## 2. The ablation measures content, not presence

Same ablation, same architecture, same partition, same denominator:

| Model | λ=0 | λ=1 | cost of removing a |
|---|---|---|---|
| real (informative a) | −81.227 | −82.137 | **0.910** |
| shuf (live but uninformative channel) | −81.294 | −81.313 | **0.019** |

**≈48×**. The shuf model also has a live, gradient-trained a-channel; removing it costs nothing ⇒ 0.91 (and therefore the 1.47 mean) is **a's content**.
Data dependence: **‖Wa‖ grows ×3.02 on real data (1.732 → 5.242) but shrinks ×0.65 on shuffled data (1.732 → 1.118)**.

## 3. Leakage trap (why anchored-split CIs are wrong)

`metric_validation` fixes the partition via `--seed`. Evaluating checkpoints trained on *other* partitions with one fixed split leaks (~95% window overlap): `[3]` ablations read 1.746 / 2.186 / 2.184 / 2.158 vs clean 1.431 / 1.290 / 1.883 / 1.848, i.e. **inflated by 0.46 ± 0.28**, and the ordering vs the training-loop gaps inverts. **Every checkpoint must be evaluated with its own `--seed`.** (`--seed` drives `perm` in `LatentSequenceDataset` → `perm[n_train:]`; training used the same `seed=args.seed`, so the split matches exactly.)

## 4. The metric's discriminative power (PC2) and the status of g

Deconfounded positive control: `data/pilot_latents_synth3` (drive-frac 0.70 / noise-frac 0.005 / rho 0.02; amp 1.35 vs real 1.14), model `runs/mdn_pilot_pc2`.

| | Real | PC2 (a genuinely drives 69% of the step) |
|---|---|---|
| nats ladder λ=0→λ=1 | 0.91 (s0) | **135.30** (148×) |
| In-dataset "ignore a" ceiling | — | −29.9; PC2 = −106.0 (76.8 lower) |

→ Capacity/optimisation is excluded, and "the metric cannot decide whether a is used" is **void**.

**g = NEW/‖Wa‖ is not a usefulness measure.** g values: **noact 6.556** (a-slice never trained) > a-reinit 7.04 > PC2 5.114 > random-init 0.73 > real 0.554. The highest value belongs to the model whose a-slice received no gradient at all. Quote g only as a ratio against an architecture-matched control.
**The script's own verdict line is unreliable**: it prints "the metric does carry learned action usage" for the **noact** model (ratio 0.11) — whose a-slice was never trained.

## 5. Data-side probes

| Probe | Result |
|---|---|
| Linear, given z_t | **−0.0009 nats/step** |
| Linear, given 3 frames of history | **−0.0003 nats/step** |
| MLP (paired three-arm, underpowered) | paired diff −0.10 ± 0.21 → inconclusive |
| Is z_t Markov? | residual variance 2.095 → 0.291 with 3 frames → **no** |

→ a's conditional linear information ≈ 0; whatever M extracts (≈0.4–0.5 nats irreducible) is not linearly available.

## 6. Mechanism

The channel is **high-gain with a loss-irrelevant direction**: ‖Wa‖ 5.242 vs random columns 1.732 (3.0×), yet g falls from 0.73 to 0.55, λ=2 costs 4.3–5.0 nats, and λ=1 is better than λ=0.5 in 4/5 seeds (s0 is the exception, marginally favouring λ=0.5) ⇒ the a-columns are about the right size after training. Combined with §5, the response is a **norm-driven structural response carrying a small non-linear signal**.

## 7. Retractions

| # | Retracted claim | Reason |
|---|---|---|
| 1 | "the metric has no discriminative power" | rested on PC1, a broken model (val +15.7, diverging) |
| 2 | "the nats bound is anti-correlated with whether a carries information" | same cause |
| 3 | "a-reinit is a usable criterion" | PC1 3.46× / PC2 6.28× / real 6.14×, unrelated to action usage |
| 4 | "dead-column artefact = 0.55" | a≡0 ⇒ the a-slice outputs `W_a@0 = 0` identically |
| 5 | "cross-axis subtraction 1.27 − 0.72 = 0.55" | mixes protocols and axes |
| 6 | "a's value is non-compensable" | retraining recovers ≈1 of 1.5 nats |
| 7 | "OLD is insensitive to the a-columns (2.5%)" | 5 seeds cv **87.3%** |
| 8 | "a-reinit NEW = 12.193 / 4.20×" | not reproduced (10.11–11.05) |
| 9 | "the frozen M's best.pt sits at epoch 0" | it is **ep 28** |
| 10 | "the positive control is scale-confounded, all its readings are void" | only absolute NLL is incomparable |
| 11 | "0.91 may be channel presence rather than content" | refuted by the shuffled-a control (§2) |
| 12 | "the 5-checkpoint loop gives a CI for 0.91" | the 4 extra checkpoints were leakage-contaminated (§3) |
| 13 | "a is worth ≈0.4–0.9 nats" | the clean cross-training-seed mean is **1.47 ± 0.41**; 0.91 is the low end |
| 14 | "the metric path reads systematically 0.2–0.7 nats better" | signs differ (real −0.50, PC2 −0.70, noact **+0.37**) |
| 15 | the handover inventory | `make_noaction_latents.py` and `--probe-only` do not exist; `--norm-min-std` is a dead flag |

## 8. Protocol rules

1. Cross-run gaps use per-epoch paired means; `best.pt` subtraction is corroborative at most (min-selection bias).
2. Across seeds, compare only same-seed paired differences (`--seed` changes both init and the split; real best spread 11.5 nats).
3. The two evaluation paths differ by ~0.4–0.5 nats with **no fixed sign** ⇒ never compare across paths.
4. Thresholds are ≥1σ, using the measured σ of that run; n=3 ⇒ 95% CI half-width = 2.48·sd.
5. An effect needs the same sign, comparable magnitude and t ≥ 2.
6. Never read the metric on a checkpoint whose a-slice was not trained except at λ=0 (injecting a costs +43.9 nats there).
7. **Evaluate every checkpoint on its own held-out partition** (`--seed` must match its training seed); a fixed split leaks for all other checkpoints.
8. `metric_validation`'s built-in verdict line is heuristic and can invert; do not quote it.

## 9. Verbatim source-line anchors

- `train_mdnrnn.py`: `    tr = LatentSequenceDataset(args.latents, args.seq_len, "train", seed=args.seed)` → `norm_min_std` never forwarded ⇒ `--norm-min-std` is dead.
- `train_mdnrnn.py`: `        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,` with `        if v < best:` → `ck` built before `best` updates ⇒ the checkpoint `best` field is **off by one** (stores −81.7242; the true value is −81.9321).
- `sequences.py`-equivalent: `        mean = np.asarray(info["mean"], np.float32)` (in `src/data/sequence.py`) → stats come from `index.json`.
- `metric_validation.py`: `    ds = LatentSequenceDataset(a.latents, seq_len=a.seq_len, mode=a.split, seed=a.seed, normalize=True, stats=stats)` → `--seed` selects the partition, which is what makes the clean CI possible.
- `make_synthetic_latents.py`: `        eps = std * a.noise_frac * rng.standard_normal(D)` → noise scaled by per-dim std, not `real_step` (`--noise-frac 0.3` is 9.5× the real step); with rho>0 drift≡drive; never prints DRIVE SHARE.
- `mdn_rnn.py`: `        log_sigma = log_sigma.clamp(math.log(c.sigma_min), math.log(c.sigma_max))` → σ∈[1e-3,10] ⇒ −82 is not a ceiling.
- Notes filenames are inconsistently capitalised (`Notes4.md`, `note5.md`) → 404 on Linux.

## 10. Artefacts and cost

| Data | Models |
|---|---|
| `data/pilot_latents` / `_noact` / `_shuf` / `_colshuf` | `runs/mdn_pilot` · `_s1..s4` · `_noact` · `_noact_s1/s2` · `_shuf` · `_shuf_s1..s4` · `_colshuf` |
| `data/pilot_latents_synth2` / `_synth3` | `runs/mdn_pilot_pc` (void) · `runs/mdn_pilot_pc2` (valid) |

**Cost**: 14 training runs × 8.2 min ≈ **1.9 h**; metric runs ~1–2 min each (CPU).

## 11. Open

1. Trace the `0.06164` denominator (never observed in this session's logs; real partitions give 0.05024–0.06135).
2. Normalise Note filename capitalisation.
3. Ablation axis (1.47) vs training axis (0.40): the ~1 nat gap is attributed to retraining recovery; a direct test would be a noact model trained *with* the real a present but stopped-gradient, if such a variant is ever wanted.
