# Note8 — Is M action-blind? Validating the instrument first

**Status:** opening section. The instrument is validated; the metric it validates **cannot** adjudicate action-blindness. Two independent measurements now suggest the action channel is near-inert, but the decisive experiment has not been run.
**Date:** 2026-09-24
**Extends:** Note4 §7 (action-blindness), Note5 §3 (the cost of using an unvalidated metric), Note7 §2 (the dream is action-independent)
**Corrects:** the evidentiary status of the ratio **0.111**, quoted in Note4 §7 / Note7 / README / content.md

---

## 1. Why this note exists

Note5's dispute (`R² over persistence`: +0.4724 vs −0.1935) cost a day because a metric was used as evidence before anyone checked what it measured. The same risk applies to the number behind every "action-blind" claim in this project — M's action sensitivity **0.111**, the numerator being `|mean pred(a_true) − mean pred(a_shuf)|` and the denominator `mean|z_{t+1} − z_t|`.

So the order was reversed: **validate the instrument before using it as evidence.** `scripts/mdn/metric_validation.py` runs four self-checks plus two architecture-matched controls on the same data (155 val windows, `seq_len = 32`, checkpoint normalisation; denominator `E|Δz| = 0.05609`, identical in every condition).

---

## 2. Instrument self-checks: all pass

| check | result | meaning |
| --- | --- | --- |
| identity — `pred(A)` vs `pred(A)` | **0.00000** (exact) | the path is deterministic; no dropout/sampling leakage |
| weight-null — a-slice of `lstm.weight_ih_l0` set to 0 | **0.00000** (exact) | `a` enters through exactly one slice, and the comparison is clean |
| resolution — 5 shuffle seeds at λ = 1 | 2.9358 ± **0.0254** (0.87 %) | signal ≫ its own noise |
| concentration — per-dim `mean|D|` | max/median **1.27** | no pathological dimensions (contrast Note5 §3) |

**The exact zeros are the load-bearing result** — they eliminate the "the metric cannot see anything" failure mode. The instrument is measurable.

---

## 3. The metric is a linear read-out of the a-columns' weight norm

λ ladder (a-slice multiplied by λ):

| λ | 0 | 0.25 | 0.5 | **1** | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `‖Wa‖` | 0 | 1.311 | 2.621 | **5.242** | 10.485 | 20.970 | 41.940 |
| `NEW = mean\|D\|/step` | 0 | 0.709 | 1.433 | **2.905** | 5.742 | 11.625 | 26.989 |
| ratio per doubling | — | — | 2.02 | **2.03** | 1.98 | 2.02 | 1.98 |

**`NEW` is proportional to `‖Wa‖` to within 2 % (r ≈ 1).** It measures the *amplitude of the response to perturbing the action slot*, not whether the action is being used. **Consequence: a cross-model `NEW` is meaningless unless divided by `‖Wa‖`.**

---

## 4. The decisive control: a randomly re-drawn a-slice responds *more* than the trained one

`a-reinit` = the trained model with **only the a-columns replaced by a fresh draw from the training init distribution** (`orthogonal_`, gain 1) — architecture, z-columns, LSTM and head all identical.

| condition | `‖Wa‖` | `NEW` | `NEW/‖Wa‖` | `OLD = \|mean D\|/step` | val NLL |
| --- | --- | --- | --- | --- | --- |
| trained | 5.242 | 2.905 | 0.554 | 0.009370 | −82.137 |
| **a-reinit** (random a-columns) | 1.732 | **12.193** | **7.039** | **0.009140** | −46.283 |
| random-init (never trained) | 1.732 | 1.250 | 0.722 | 0.000850 | +50.833 |

Two results, in opposite directions from what the ratio was supposed to show:

* **`NEW(a-reinit)/NEW(trained) = 4.20`** — per unit weight, **12.7×**. Randomly initialised action columns perturb the prediction far more than the trained ones. **Training did not increase the action channel's influence; it reduced it** (while growing its norm 3.0× — i.e. the channel was pushed to a larger but less consequential direction).
* **`OLD` does not respond to the a-columns at all**: 9.370e-3 (trained) vs 9.140e-3 (a-reinit), a **2.5 %** difference. Since `OLD` is the numerator of the quoted 0.111, that number is driven by the trained z-pathway/LSTM/head response geometry, **not** by the action columns.

**Therefore the "action sensitivity = 0.111" figure loses its evidentiary status.** It remains a correct arithmetic statement about `mdn_probe.py`; it is not evidence about whether M uses actions.

---

## 5. An unexpected but more useful quantity: the likelihood cost of removing the channel

The λ ladder also reports val NLL at each scale:

| λ (a-slice scale) | 0 (channel removed) | 0.5 | **1 (as trained)** | 2 |
| --- | --- | --- | --- | --- |
| val NLL | −81.227 | **−82.185** | −82.137 | −77.815 |

* **Removing the entire action channel costs 0.91 nats** — **1.1 %** of the total.
* λ = 0.5 scores *better* than λ = 1 (−82.185 vs −82.137), i.e. the trained a-columns are marginally too large for the likelihood.

This is an **upper bound** on the channel's contribution (an M retrained without actions might recover all of it), and it is consistent with the data-side argument of Note2 §5–§6: under a uniform-random policy `a_t` carries almost no information about `Δz`.

---

## 6. Corrections

| # | claim | what overturned it |
| --- | --- | --- |
| 1 | a large `NEW` implies M uses the action | `NEW` is linear in `‖Wa‖`, and a randomly re-drawn a-slice scores **higher** than the trained one (4.20×) |
| 2 | `NEW(random-init)/NEW(trained) = 0.43` supports "the metric carries learned action usage" (the script's own printed verdict) | that verdict used the **un-normalised** ratio; per unit weight it is 0.722/0.554 = **1.30**, i.e. the wrong way. `random-init` is also a poor control: its NLL is +50.8/dim, so its output scale has degenerated |
| 3 | the synthetic "positive control" (Note8 work in progress) | this run passed **no `--latents`**, so it was evaluated on the **real** latents (identical denominator 0.05609 — the proof); its `best.pt` is also epoch 0 (val 6.9960, later overfitting to 9.15). Not interpretable; to be redone |
| 4 | `OLD` (probe's formula) as an action-sensitivity measure | insensitive to the a-columns (2.5 %); its numerator is dominated by other pathways |

---

## 7. Verdict so far, and what is still missing

**Action-blindness is *measurable* but not *adjudicable* by this metric.** Both formulae describe the response geometry of the action slot; neither can establish that the semantics of the action were learned.

The direction is nevertheless supported by **two independent measurements**: the channel contributes **≤ 0.91 nats (1.1 %)** of the likelihood, and its learned influence is **lower** than a random one per unit weight. Pending, in order of cost:

1. re-run the `a-reinit` comparison over **5 init seeds** (~1 min) — confirm that `|mean D|` is structurally insensitive to the a-columns;
2. **functional ablation**: train M on `data/pilot_latents_noact` (actions zeroed) and compare val NLL (~10 min, 583 s per run) — the only experiment that yields a *lower* bound and closes the bracket with §5;
3. only then settle the wording of Note4 §7 (`action-blind` → `high-gain, information-free action channel`), and propagate to Note7, README and `content.md`.