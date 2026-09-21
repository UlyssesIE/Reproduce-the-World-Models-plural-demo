
# Note6 — Controller (C): the full arm, and a paired comparison against V-only

**Status:** C complete — both arms trained; C(867) beats C(99) by **+94.5** on identical seeds
**Date:** 2026-09-21
**Extends:** Note5 §8 (controller specification) and §11 (the V-only arm, 525.15 ± 165.50)
**Follows:** Note5 §11 identified V-only as the control condition for the 867-parameter arm

---

## 1. Summary

The 867-parameter controller — `a_t = W_c [z_t; h_t] + b_c`, evolved by CMA-ES in the
**real** environment, with `h_t` the LSTM output vector of M — was trained for 100
generations and evaluated on **the same 100 seeds** used for the V-only arm.

| arm | parameters | eval (100 rollouts) |
| --- | --- | --- |
| V-only (`--no-hidden`), Note5 §11 | 99 | 525.15 ± 165.50 |
| **full (V + `h_t`)** | **867** | **619.65 ± 179.61** |

**Same seeds, so the comparison is paired: +94.50 per-rollout on average**, ≈ 3.9
independent standard errors. Within this environment and budget, **the LSTM hidden
state carries information beyond `z_t`** — the same direction the paper reports
(V-only 632 ± 251 → full 906 ± 21), at 0.4 % of its compute budget.

Two caveats are recorded in §4 and must accompany the claim.

---

## 2. Setup

```
python scripts/controller/train_controller.py \
  --vae runs/vae_pilot2/best.pt --mdn runs/mdn_pilot/best.pt \
  --workers 8 --popsize 16 --search-episodes 8 \
  --eval-episodes 100 --eval-every 25 --generations 100 \
  --out runs/controller_867
```

| item | value |
| --- | --- |
| parameters | **867** (`in_dim = 288`, `use_hidden = True`); asserted at startup |
| input | `[z_t (32); h_t (256)]`, `z_t` normalised by M's checkpoint statistics |
| search | 16 candidates × 8 rollouts = **128 rollouts/generation** (paper: 64 × 16) |
| generations | **100**, no early stopping |
| periodic evaluation | 100 rollouts at generations 24 / 49 / 74 / 99 |
| wall clock | **49 032 s ≈ 13.6 h** for the training loop (+6.5 min final evaluation) |
| per generation | **490 s** (of which a periodic evaluation costs **390 s**) |

**The search was saturated well before the budget ran out.** Over the last six
generations `search_mean_return` moved 520.4 → 540.3 → 543.9 → 511.8 → 506.2 → 534.0,
i.e. **flat within its own ±50 spread**; 100 generations is therefore not a
binding constraint on this result.

---

## 3. Result

```
gen 99  periodic eval :  619.65 +/- 179.61   (100 rollouts)
FINAL   eval.json     :  619.6490956828967 +/- 179.6130306747357
```

`eval.json` and the generation-99 evaluation agree **bit-for-bit** (same θ, same seed
set) — a third independent confirmation that the evaluation path is deterministic
(Note5 §10.4).

Because both arms were evaluated on **seeds 20000–20099**, the difference is a paired
quantity:

| | mean | std | SE (σ/√100) |
| --- | --- | --- | --- |
| V-only (99 p) | 525.15 | 165.50 | 16.6 |
| full (867 p) | **619.65** | 179.61 | 18.0 |
| **paired difference** | **+94.50** | — | 24.4 (independent) → **z ≈ 3.9** |

**Limitation.** `eval.json` stores only the mean and standard deviation, not the
per-rollout returns, so a *paired* standard error cannot be computed from the stored
artifacts. Given identical seeds, the paired error is the correct one and is likely to
be smaller than the conservative 24.4 above. Fixing this is one small change (§6.1).

---

## 4. What this does and does not show

**It shows.** On CarRacing-v3, with V and M frozen and identical evaluation seeds, a
linear controller that sees `[z_t, h_t]` scores **+94.5** over one that sees `z_t`
alone, at the same search budget.

**It does not show** that the whole gain is due to *information* in `h_t`. The 867
arm has 8.75× the parameters of the 99 arm, so part of the improvement may be
**capacity**, not information. Separating the two requires a control that keeps the
867-parameter architecture but removes the information:

| control | isolates | cost |
| --- | --- | --- |
| 867-parameter `W_c`, but feed `h_t ≡ 0` | information vs capacity | ~14 h |
| the paper's V + 40-tanh hidden layer (1443 p, 788 ± 141) | capacity scaling | ~14 h |

Neither was run. Until one is, the honest phrasing is **"adding `h_t` to the input
improves the controller by +94.5 on paired seeds"**, not "`h_t` contributes 94.5
points of information".

**It is not a reproduction of the paper's score.** The paper reports 906 ± 21 on
**CarRacing-v0** with **10 000** rollouts and 1 800 generations of population 64 × 16.
This run used v3, 100 rollouts, 100 generations, 16 × 8. Scores are reported against
this environment's own random-policy baseline (**−32.78**).

---

## 5. Two things worth recording about the code

**5.1 `imap_unordered` returns results in completion order, not submission order.**
Harmless for the mean and standard deviation, but any per-seed dump must carry the
index explicitly, or the seeds will be mislabelled. The planned `--dump-returns` change
passes `(i, theta, seed)` and writes into a preallocated list (§6.1).

**5.2 The pool now tears down cleanly.** `FIN`… the run ended with `FINAL 619.65 …`
rather than the `ValueError: Pool not running` that ended the Phase-1 run (Note5 §9.5);
`eval.json` was written and zero `python.exe` processes remained afterwards.

---

## 6. Open items

| # | item | cost |
| --- | --- | --- |
| 6.1 | `--dump-returns`: write the 100 per-rollout returns into `eval.json`, then re-evaluate both arms with `--eval-only` (~15 min total) to obtain a **paired** confidence interval on +94.5 | small |
| 6.2 | control ablation from §4 (`h ≡ 0`) | ~14 h |
| 6.3 | scale the dataset from 100 to the 2 000 rollouts already on disk — the only lever on action-blindness (ratio 0.111) | hours |
| 6.4 | accumulated technical debt: `pyproject.toml`, probe provenance, `sequence.py` docstring | small |

---

## 7. Hypothesis ledger (extended)

Continuing Note5 §15.

| Version | Hypothesis | Status | What settled it |
| --- | --- | --- | --- |
| v29 | adding `h_t` to the controller input improves the score beyond `z_t` alone | **measured** (paired, same seeds) | 619.65 vs 525.15, +94.5 |
| v30 | that +94.5 is entirely attributable to the *information* in `h_t` | **not established** | the arms differ in parameter count as well as input; §4 |
| v31 (current) | C converges well before 100 generations, and the evaluation path is bit-for-bit deterministic | measured | search flat over the last 6 generations; `eval.json` ≡ gen-99 evaluation |

---

## 8. Artifacts added in this note

| file | purpose |
| --- | --- |
| `runs/controller_867/` | 100-generation log, `best_theta.npz`, `stdout.log`, `eval.json` (619.65 ± 179.61) |
| `runs/controller_vonly/` | the V-only control arm (525.15 ± 165.50), retained |

---

## 9. Disclosure

* **C trained in the real environment**, per the paper's CarRacing experiment; M
  contributed only the LSTM output vector `h_t`, and no dreaming was used.
* **Budget.** 12 800 rollouts of search + 500 rollouts of evaluation, against the
  paper's ~1.84 M; **the paper's 906 ± 21 is not claimed as reproduced.**
* **Environment.** gymnasium **CarRacing-v3** (`continuous=True`,
  `lap_complete_percent=0.95`, `max_episode_steps=1000`), baseline **−32.78**.
* **Action-blindness of M (ratio 0.111)** continues to be disclosed: `h_t` improves the
  controller here, but M's one-step prediction barely depends on `a_t`.
* **The +94.5 improvement is paired on seeds 20000–20099**, but only an independent
  standard error (24.4) is computable from the stored artifacts; §6.1 removes that
  limitation.
* **Hardware.** AMD Ryzen 9 5900HX (8 physical cores); rollout throughput is CPU-bound
  at ~250 steps/s.


