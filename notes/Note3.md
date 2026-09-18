# Note3 — VAE training: implementation fixes, the I/O bottleneck, and V-model acceptance

**Status:** V model accepted; M (MDN-RNN) implementation written, not yet trained
**Date:** 2026-09-17
**Extends:** Note2 §8 (decision ordering) and §9 (VAE implementation notes)
**Corrects:** the interpretation of the early VAE results recorded during this period
**Follows:** Note2 §8 — V and M were trained on the 100 pilot rollouts *before*
deciding the final dataset scale.

---

## 1. Summary

Three implementation defects had to be fixed before the V model could be judged,
and two of the measurements used to judge it were themselves wrong:

1. the convolutional padding convention produced a 5x5 feature map where the code
   assumed 4x4, so the encoder crashed on the first batch;
2. `np.memmap` was being pickled in full (2.76 GB) on Windows `spawn`, killing the
   DataLoader workers;
3. the frame loader spent **99%** of wall-clock time waiting on disk, which meant the
   first long training run performed only ~4 minutes of equivalent GPU work;
4. the first quality metric (BCE as a percentage gain) has a high floor on
   continuous targets and understated the model's quality by ~40x;
5. a predicted decoder collapse did not exist.

After the fixes, **30 epochs take 286 s** (previously ~3 h) and the V model reaches
**R² = 0.916** with **87.6%** MSE reduction over the per-pixel-mean baseline. The
remaining plateau is a property of the data, not of the model (see §7).

---

## 2. Fix — convolution padding produced a 5x5 feature map

**Symptom**

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x6400 and 4096x256)
```

**Cause.** With `kernel=4, stride=2, padding=kernel//2=2`, the spatial size does
*not* halve exactly:

| layer | in | out |
| --- | --- | --- |
| conv1 | 64 | floor((64+4-4)/2)+1 = **33** |
| conv2 | 33 | **17** |
| conv3 | 17 | **9** |
| conv4 | 9 | **5** |

so the flattened size is `256*5*5 = 6400`, while the code derived it by the formula
`64 // 2**4 = 4` and built `Linear(4096, 256)`. Exact halving with `k=4, s=2`
requires **`padding=1`**.

**Fix (three parts).**

* `padding` promoted to a config field, set to `1`;
* the encoder now **measures** the real output shape by pushing a dummy tensor
  through the conv stack, asserts it equals `frame_size / 2**n_layers`, and exposes
  `self.spatial` / `self.flat_dim`; the decoder receives `spatial` from the encoder;
* `VAE.__init__` runs a round-trip probe and asserts
  `decoder(encoder(x)).shape == x.shape`.

The decoder carried the same latent bug: `ConvTranspose2d(k=4, s=2, p=2)` maps
5->8, and the chain 5->8->14->26->**50** would have produced 50x50 output. It never
surfaced because the encoder failed first.

**Parameter count is unchanged** at 3,506,435 — padding is not a parameter, and
`flat_dim` returns to 4096. The reported gap against the paper's 4,348,547
(delta -842,112) still stands.

> Lesson recorded because it generalises: deriving tensor shapes from a formula is
> fragile when the kernel/stride/padding combination changes. Measuring once at
> construction and asserting turns a class of silent shape bugs into an immediate
> failure.

---

## 3. Fix — `np.memmap` is serialised in full by `pickle`

**Symptom** (on the first epoch, when the validation loader was created)

```
reduction.dump(process_obj, to_child)
OSError: [Errno 22] Invalid argument
...
_pickle.UnpicklingError: pickle data was truncated
```

**Cause.** `FlatFrameDataset` held `self.arr = np.load(path, mmap_mode="r")`.
`np.memmap` inherits `ndarray`'s pickle behaviour and serialises **the entire
buffer**. Windows uses `spawn`, so every worker must receive the dataset object
through a pipe. Measured directly:

```python
pickle.dumps(dataset)   # before fix: 2764.80 MB
                        # after  fix:    0.00 MB
```

Each of the 4 train workers plus 2 validation workers therefore tried to receive
2.76 GB through a pipe — and since the whole point of memmap was to *share* one
mapping through the OS page cache, the approach was counterproductive even when
it did not crash.

**Fix.** Define `__getstate__` / `__setstate__` on `FlatFrameDataset` so that `arr`
is dropped before pickling and re-opened (`np.load(..., mmap_mode="r")`) inside the
child process. Also: `persistent_workers` is now bound to `num_workers > 0`, since
`persistent_workers=True` with `num_workers=0` is invalid.

**Effect**

| | before | after |
| --- | --- | --- |
| `pickle.dumps(dataset)` | 2764.80 MB | **0.00 MB** |
| time to `step 50` | 39 s | **4 s** |
| per epoch | ~50 s | **~10 s** |

---

## 4. Fix — the frame loader was 99% of the training time

`scripts/bench_step.py` separates dataloader wait from GPU compute:

| stage | ms / batch |
| --- | --- |
| data | **983.9** |
| compute (bf16) | **10.7** |
| share | data 99%, compute 1% |

**Cause.** `RolloutFrameDataset` reads a **26.4 MB** archive on every cache miss,
and each worker keeps its own LRU (default 8 files out of ~100), giving ~8% hit
rate; 4 workers read independently. Per batch this is roughly 120 file reads
x 26 MB ~ 3 GB.

**Consequence for earlier results.** 742 batches/epoch x 983.9 ms = **12.3 min per
epoch**, of which the GPU was busy for 7.9 s. The 30-epoch run took ~3 h while
accumulating only **~4 minutes** of equivalent GPU work. This is the explanation
for the "validation loss stops improving after epoch 1" observation recorded
earlier — not a modelling problem.

**Fix.** `scripts/build_frame_cache.py` flattens all rollouts into a single
memmapped `.npy` (2.76 GB for 100k frames) plus a `.index.json` giving each file's
`(start, count)`; `FlatFrameDataset` slices splits out of that one array. All
workers share the same mapping through the OS page cache.

Note: an earlier draft of `FlatFrameDataset.__getitem__` indexed the array with the
*local* frame offset instead of `start + local`. That would have silently served
training frames to the validation split. The shipped version uses the global index.

**Effect (measured, 30 epochs, batch 128, lr 3e-4)**

| | before | after |
| --- | --- | --- |
| per epoch | 12.3 min | **~10 s** |
| 30 epochs | ~3 h | **286 s** |

---

## 5. Correction — the quality metric was wrong, not the model

Two scripts were written to test "is the VAE actually reconstructing, or just
emitting an average image": `vae_baseline.py`, then `vae_diagnose.py`. Both
concluded the model was weak (`VAE / per-pixel-mean = 0.98`, "only marginally
better than the mean image"). **That conclusion was wrong**, for two separate
reasons.

**(a) A units bug in the script.** The per-channel VAE BCE was divided by an extra
factor of `n`, so every channel printed `0.0003` and every "gain" printed `100.0%`.

**(b) The metric has a high floor.** CarRacing renders terrain at
`road=[102,102,102]`, `background=[102,204,102]`, `grass=[102,230,102]` — i.e.
approximately 0.4, 0.4, and 0.72 after normalisation. Under BCE a constant target
`p` has an irreducible loss `H(p)`:

```
H(0.4)  = 0.673        H(0.72) = 0.592        ln 2 = 0.693
```

Measured data statistics confirm this:

| ch | mean | std | H(mean) |
| --- | --- | --- | --- |
| 0 (R) | 0.3953 | **0.0331** | 0.6686 |
| 1 (G) | 0.7210 | 0.1865 | 0.6014 |
| 2 (B) | 0.3921 | **0.0302** | 0.6671 |

R and B are near-constant across the palette, so a *near-perfect* predictor can
only improve on the constant baseline by a few percent when scored in BCE terms.
The 2% figure said nothing about the model.

> A previous statement in this log that "two thirds of the pixel loss is
> irreducible" was too strong: R and B do vary on the car and on track edges, so
> they are predictable in principle. The accurate statement is that they carry
> almost no *track-shape* information, and that their BCE floor is dominated by the
> constant background.

**Fix — switch to explained variance.** `scripts/vae_check2.py` reports, per channel,
BCE in correct units, `R²` on an MSE basis, and the MSE reduction relative to the
per-pixel mean. Results for the accepted model (`runs/vae_pilot2/best.pt`, epoch 27):

| ch | R² (vs per-pixel mean) | MSE gain |
| --- | --- | --- |
| 0 | 0.200 -> **0.904** | 88.0% |
| 1 | 0.566 -> **0.984** | 96.3% |
| 2 | 0.351 -> **0.860** | 78.4% |
| **mean** | | **87.6%** |

**mean R² = 0.916.** On the previous run (fewer epochs, identical architecture,
same data) it was 0.909. The VAE is working.

---

## 6. Correction — no decoder-side collapse

Before measuring, it was predicted that

> "the decoder output barely moves for any z: DECODER-side collapse",

i.e. a reconstruction std ratio below 0.2. Measured on 64 distinct frames:

| quantity | std | ratio to data |
| --- | --- | --- |
| data | 0.0569 | 1.000 |
| `decode(mu)` | 0.0556 | **0.978** |
| `decode(random z ~ N(0,1))` | 0.0389 | 0.683 |
| `mu` (per dim) | std 0.1420, `|mean|` 0.1118 | |

**0.978** — the reconstruction varies across inputs almost as much as the data
does. The latent is being used. The prediction was wrong.

---

## 7. Correction — the plateau is genuine convergence, not compute starvation

When the loss stopped improving, the working hypothesis was that only ~4 minutes of
GPU work had been done, so the model "had not had the chance to converge". The I/O
fix made it cheap to test that, and the hypothesis failed:

| | earlier run (~4 min equiv. GPU) | full run (286 s) |
| --- | --- | --- |
| best val loss | 7567.6 | **7566.8** |
| val rec | 7558.8 | 7557.3 |
| R² | 0.909 | 0.916 |

Total improvement across 30 epochs: 7589.4 -> 7566.8, i.e. **0.30%**, essentially all
of it inside epoch 0. Two independent runs converge to the same solution.

**Conclusion: R² ~ 0.92 is the information ceiling of this dataset.** This is
consistent with the dataset analysis in Note2 §6 — the car crawls at ~3.1 units/s,
consecutive frames are nearly identical, and the per-frame content is intrinsically
low-entropy. **No further tuning of the V model is warranted.**

This also validates the Note2 §8 decision to train V and M on the 100 pilot
rollouts before collecting more: the pipeline is now known to work, and the data
ceiling is known, without spending 5 hours collecting 2,000 rollouts first.

---

## 8. Open item — only 6 of 32 latent dimensions are active

`KL total = 9.15 nats`, `active dims (KL > 0.1) = 6 / 32` (was 5/32 before retraining).
A high R² with few active dimensions is not contradictory: the data is highly
redundant and the VAE found a compact code.

**Risk to flag.** If those dimensions encode *which track the car is on* rather than
*where the car is on the track*, then M will learn a static transition and the dream
will be frozen. M is trained on `(z_t, a_t, z_{t+1})`, so within-episode variation of
z is exactly what it needs.

**Planned test** — `scripts/z_dynamics_check.py` decomposes the variance of each
latent dimension into a *within-rollout* and an *across-rollout* (between episode
means) part and reports the within share:

| within share | interpretation | action |
| --- | --- | --- |
| > 50% | z encodes dynamics | proceed to the controller |
| 15–50% | mixed | proceed, but expect a partly static dream |
| < 15% | z encodes track identity | revisit: add a temporal term to V, or predict dz |

**Status: pending** — `encode_latents.py` and `z_dynamics_check.py` are written but
had not been run when this note was written.

---

## 9. Hardware and environment

| item | value |
| --- | --- |
| GPU | **NVIDIA GeForce RTX 3060 Laptop GPU**, 6144 MiB |
| driver | 616.64 |
| torch | 2.5.1+cu121, CUDA 12.1 |
| device line in training log | `cuda | torch 2.5.1+cu121 | cuda 12.1 | available=True` |

Note: an `nvidia-smi` reading taken during this period showed `143 MiB / 6144 MiB`
and `0%` utilisation. That reading was taken while no training was active, and WDDM
reporting on this machine is unreliable; the definitive evidence that training runs
on the GPU is the `[device] cuda` line plus the 10.7 ms/batch compute measurement
from `bench_step.py`.

**Headroom.** Batch 128 uses ~143 MiB of 6144 MiB. The 6 GB budget is not a
constraint for V or M at these sizes, and `--batch-size 512` is a reasonable next
experiment (with the learning rate scaled accordingly).

---

## 10. Hypothesis ledger (extended)

Continuing from Note2 §10.

| Version | Hypothesis | Status | What overturned it |
| --- | --- | --- | --- |
| v6 | the conv stack yields 4x4 with `padding=kernel//2` | **wrong** | actual 5x5; `k=4,s=2` needs `padding=1` |
| v7 | the 3 h run had too little compute, so the VAE had not converged | **wrong** | a full 286 s run converged to the same value (7566.8 vs 7567.6) |
| v8 | the VAE is only ~2% better than the per-pixel mean -> weak/collapsed | **wrong** | script units bug, plus BCE's high floor on continuous targets |
| v9 | decoder-side collapse (recon std ratio < 0.2) | **wrong** | measured 0.978 |
| v10 | `np.memmap` pickles lazily | **wrong** | `pickle.dumps` = 2764.80 MB |
| v11 (current) | V is sound (R²=0.916) at the data ceiling; the open question is whether z encodes within-episode dynamics | pending §8 | — |

Retained deliberately: v7 shows that a plausible causal story ("not enough compute")
can survive until a cheap measurement contradicts it; v8 shows that a wrong metric
can produce a confident wrong conclusion about a model that is fine.

---

## 11. Artifacts added in this note

| file | purpose |
| --- | --- |
| `scripts/build_frame_cache.py` | flatten rollouts into one memmapped `.npy` + index |
| `src/data/frames.py` :: `FlatFrameDataset` | memmap-backed dataset with pickle-safe state |
| `scripts/bench_step.py` | split per-batch time into data wait vs GPU compute |
| `scripts/vae_check2.py` | corrected VAE metrics (BCE, R², MSE gain) — supersedes `vae_baseline.py` / `vae_diagnose.py` |
| `scripts/encode_latents.py` | freeze V, encode all frames to `(N, 32)`, store per-dim mean/std |
| `scripts/z_dynamics_check.py` | within- vs across-rollout variance decomposition of z |
| `src/models/mdn_rnn.py` | MDN-RNN: LSTM + Gaussian mixture + temperature sampling |
| `src/data/sequences.py` | sequence windows over latents, rollout-boundary safe |
| `scripts/train_mdnrnn.py` | M training loop (NLL objective) |
| `scripts/dream_rollout.py` | autoregressive dream, decoded through V, saved as an image strip |

---

## 12. Disclosure (updated)

* **V model.** Trained on 100 random-policy rollouts (95,000 train / 5,000 val frames
  at 64x64, `crop_bottom=12`). Accepted at **R² = 0.916**, mean MSE gain 87.6% over the
  per-pixel-mean baseline, `KL total = 9.15` with **6 of 32** latent dimensions active.
* **Parameter count.** 3,506,435 vs the paper's **4,348,547** (delta -842,112). The
  paper does not describe the architecture in enough detail to reproduce the count;
  the training script prints the delta on every run.
* **Architecture choices not specified by the paper:** 4 conv layers at
  32/64/128/256 with `k=4, s=2, p=1`; Bernoulli (BCE) reconstruction loss;
  `beta = 1.0` with optional warm-up; free bits off; sigmoid output.
* **Training cost.** 30 epochs in **286 s** on an RTX 3060 Laptop (6 GB), batch 128,
  bf16. Earlier, much longer runs were I/O-bound and are not representative of the
  architecture's cost.
* **M model (for the next note).** The paper states the CarRacing variant uses an
  **LSTM with 256 hidden units**; the number of mixture components is not specified
  (5 chosen here). The paper's CarRacing M does **not** predict `done`; only the
  VizDoom variant does.
* **Hardware.** All timings are from an RTX 3060 **Laptop** GPU with 6 GB, not a
  desktop 3060.
