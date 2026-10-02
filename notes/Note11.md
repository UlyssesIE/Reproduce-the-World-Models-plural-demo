# Note 11 - Dream controllability as a dose-response: single-cycle audit (v5)

Repository: `Reproduce-the-World-Models-plural-demo`. Source: arXiv:1803.10122 only;
no official or third-party code was read. Every number below is emitted by the generator
that wrote this file; none is typed by hand.

## 1. Instrument, and one thing that is not a check

`scripts/mdn/dream_controllability.py` slices the pilot dataset for one rollout `r`:

- `index.json` gives `(start, count) = (r*1000, 1000)` for `rollout_*.npz`;
- `n = min(steps + warmup, count) = min(216, 1000) = 216` frames are used;
- `Z = latents.npy[s0:s0+n]` and `A = actions.npy[s0:s0+n]`;
- actions are read from the *same* directory that is passed to `--latents`.

For each `(tau, t)` the script prints `A(stoch)`, `B(noise)`, `A/B` and `D(det)`; the default
`--taus` is `0.1,0.5,1.0` (`dream_controllability.py:117`). There are five self-checks:
`permutation moved X%`, `identity = 0`, a zeroed a-slice yields `A = 0`, two-pass bit-identical
output, and the printed medians. Note that **"`D` is tau-invariant" is not a self-check but an
identity of the code**: `mode="det"` calls `mix_mean(pi, mu)` and never assigns `temperature`.
Table 2 confirms this empirically (spread `0.0000` across tau).

## 2. Frozen protocol

```text
scripts/mdn/dream_controllability.py --mdn <ckpt> --latents <PROBE_DIR(row)> \
    --rollout r --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0      (r = 0..19)
```

- `D/B := D(det)_mean / B(noise)_mean` (the script does not print `D/B`; it is column 8 divided by column 5).
- `A/B` := the printed `A/B` column.
- `x := NLL(lambda=0) - NLL(lambda=1)`, tau-independent, from the row's seed-0 checkpoint.

`PROBE_DIR(row) = ckpt["args"]["latents"]`, with **two declared overrides**:

- `real-struct`   -> `data\pilot_latents`
- `synth3-struct` -> `data\pilot_latents_synth3`

Reason: those two rows' own `actions.npy` are entirely zero (`permutation moved 0.0%`), which
degenerates `A = D = 0`. The override is declared in the script together with this rationale.

## 3. Table 1 - the 15-row single-cycle table (t = 1, tau = 1.0)

| row | family | x | D/B | SE | A/B | D(det) |
|---|---|---|---|---|---|---|
| shuf | real | 0.019 | 0.023117 | 0.002491 | 0.0240 | 0.327 |
| p25 | real | -0.045 | 0.015406 | 0.001272 | 0.0160 | 0.326 |
| p50 | real | -0.131 | 0.024856 | 0.001602 | 0.0290 | 0.576 |
| p75 | real | 0.163 | 0.081848 | 0.008541 | 0.0860 | 1.014 |
| real | real | 0.910 | 0.089631 | 0.008373 | 0.0930 | 1.573 |
| real-struct | real | - | 0.156493 | 0.014435 | 0.1610 | 3.115 |
| s3p10 | synth3 | 0.685 | 0.142133 | 0.007201 | 0.1940 | 1.624 |
| s3p15 | synth3 | 2.161 | 0.145003 | 0.010504 | 0.1880 | 1.692 |
| s3p20 | synth3 | 2.435 | 0.184936 | 0.009912 | 0.2450 | 2.285 |
| s3p25 | synth3 | 6.813 | 0.270098 | 0.018495 | 0.3200 | 3.135 |
| s3p50 | synth3 | 13.993 | 0.464836 | 0.028735 | 0.4885 | 4.994 |
| s3p75 | synth3 | 22.053 | 0.667618 | 0.044041 | 0.6705 | 6.835 |
| s3p90 | synth3 | 26.837 | 1.059601 | 0.067951 | 1.0815 | 8.479 |
| PC2 | synth3 | - | 1.281279 | 0.067175 | 1.2830 | 8.967 |
| synth3-struct | synth3 | - | 0.731066 | 0.030239 | 0.7330 | 9.633 |

`D/B` here reproduces the frozen v2 table exactly (15/15), i.e. this repository's v5 single-cycle
run is bit-compatible with the earlier v2 table for that one column and for `SE`.

## 4. Table 2 - dose-response on the action-keep axis (x-free)

The axis is the fraction of ACTION ROWS LEFT IDENTICAL to the real episode (the `keep` column
below). The latents are held byte-identical across all seven doses (md5 `f5433cf8e0de`, Section 11),
so the dose acts only on the action channel; `mean|dz|` is `0.00275` at every dose. The historical
label `drive_frac` is retained in the credentials but is NOT the generator's `--drive-frac`: it is
this keep fraction (Section 11).

| row | keep_frac | D/B tau=0.1 | D/B tau=0.5 | D/B tau=1.0 | A/B tau=0.1 | A/B tau=0.5 | A/B tau=1.0 |
|---|---|---|---|---|---|---|---|
| s3p10 | 0.10 | 0.2727 | 0.1809 | 0.1421 | 0.3685 | 0.2460 | 0.1940 |
| s3p15 | 0.15 | 0.2899 | 0.1868 | 0.1450 | 0.3710 | 0.2405 | 0.1880 |
| s3p20 | 0.20 | 0.3868 | 0.2408 | 0.1849 | 0.5015 | 0.3145 | 0.2450 |
| s3p25 | 0.25 | 0.4697 | 0.3351 | 0.2701 | 0.5550 | 0.3955 | 0.3200 |
| s3p50 | 0.50 | 0.7679 | 0.5688 | 0.4648 | 0.8065 | 0.5975 | 0.4885 |
| s3p75 | 0.75 | 1.2435 | 0.8438 | 0.6676 | 1.2480 | 0.8475 | 0.6705 |
| s3p90 | 0.90 | 2.5461 | 1.4149 | 1.0596 | 2.5980 | 1.4445 | 1.0815 |

`D/B` crosses 1.0 at:

- tau = 0.1: between keep_frac 0.50 and 0.75
- tau = 0.5: between keep_frac 0.75 and 0.90
- tau = 1.0: between keep_frac 0.75 and 0.90

The real family never approaches 1.0 at any tau; the synth3 family crosses it, and the crossing
moves right as tau grows. `D_mean` is tau-invariant to the last printed digit at every `t`.

The coordinates here are the *nominal* keep fraction; Table 3 does not use them. Its crossover is
interpolated on the measured per-row `x = NLL(lambda=0) - NLL(lambda=1)` from `tmp_v5_table.csv`,
so the tau-resolved headline is independent of this axis label.

## 5. Table 3 - tau-resolved headline

`gap_i := x_cross(tau) / x_i`, where `x_i` are the five real-family per-training-seed
lambda-ablation values (`notes/Note9.md:25`).

| tau | D/B=1 bracket | x_cross (D/B) | x_cross (A/B) | gap min | gap p50 | gap mean | gap max | seeds >= 15x |
|---|---|---|---|---|---|---|---|---|
| 0.1 | s3p50 -> s3p75 | 17.93 | 17.53 | 9.52 | 12.53 | 13.07 | 19.70 | 1/5 |
| 0.5 | s3p75 -> s3p90 | 23.36 | 23.28 | 12.41 | 16.33 | 17.03 | 25.67 | 3/5 |
| 1.0 | s3p75 -> s3p90 | 26.11 | 25.89 | 13.87 | 18.25 | 19.03 | 28.69 | 3/5 |

The headline is not a single number: the family gap grows with tau (13x -> 17x -> 19x), and
"at least 15x" holds only for tau >= 0.5. The same interpolation on `A/B` gives a systematically
~2% smaller crossover, consistent with Section 8.

## 6. Table 4 - family separation on the tau-slope of A/B(t = 200)

| row | family | tau=0.1 | tau=0.5 | tau=1.0 | slope |
|---|---|---|---|---|---|
| shuf | real | 0.3305 | 0.3780 | 0.4665 | rising |
| p25 | real | 0.3350 | 0.4405 | 0.5445 | rising |
| p50 | real | 0.5980 | 0.6540 | 0.7355 | rising |
| p75 | real | 0.6930 | 0.7530 | 0.7820 | rising |
| real | real | 0.7815 | 0.8130 | 0.8130 | rising |
| real-struct | real | 0.8985 | 0.8870 | 0.9060 | non-monotone |
| s3p10 | synth3 | 0.3995 | 0.3080 | 0.2645 | falling |
| s3p15 | synth3 | 0.5040 | 0.4130 | 0.3440 | falling |
| s3p20 | synth3 | 0.5180 | 0.4180 | 0.3395 | falling |
| s3p25 | synth3 | 0.6680 | 0.5480 | 0.4900 | falling |
| s3p50 | synth3 | 0.9365 | 0.8280 | 0.6910 | falling |
| s3p75 | synth3 | 1.3435 | 1.1600 | 1.0440 | falling |
| s3p90 | synth3 | 2.1255 | 1.6670 | 1.4210 | falling |
| PC2 | synth3 | 9.1900 | 4.4660 | 3.0855 | falling |
| synth3-struct | synth3 | 0.8305 | 0.7240 | 0.6520 | falling |

Counts: rising = 5, falling = 9, non-monotone = 1. All six real-family rows are rising **or flat**
(`real-struct` is `0.898 -> 0.887 -> 0.906`, a 1.2% dip in the middle: endpoint-labelled, not strict);
all nine synth3 rows are falling. The two families separate on the sign of this slope.

## 7. Table 5 - seed-axis error bars (tau = 1, t = 1, 20 rollouts per checkpoint)

| family | n_seed | per-seed D/B | mean | sd | SE | cv |
|---|---|---|---|---|---|---|
| real | 5 | 0.0896 / 0.1544 / 0.1167 / 0.1174 / 0.1424 | 0.1241 | 0.0252 | 0.0113 | 20.3% |
| shuf | 4 | 0.0231 / 0.0208 / 0.0257 / 0.0213 | 0.0227 | 0.0022 | 0.0011 | 9.7% |
| noact | 3 | 0.1565 / 0.1781 / 0.2250 | 0.1866 | 0.0350 | 0.0202 | 18.8% |

- family separation, seed axis: `0.1241 +- 0.0113` vs `0.0227 +- 0.0011`, difference `0.1014 +- 0.0113` = **9.0 sigma**
- real vs noact: `0.1241 +- 0.0113` vs `0.1866 +- 0.0202` = **-2.7 sigma** (ratio 1.50x)

The seed-0 checkpoints used in Table 1 are the *lowest* member of each family (real: 1/5, noact: 1/3;
shuf is not), so both the seed-0 and seed-mean ratios must be quoted. Table 1's `SE` is rollout-level;
the seed-level SE for the `real` row is the one that matters for cross-seed claims.

## 8. Correction: D/B is not A/B

At `(tau = 1, t = 1)` the relative difference `|D/B - A/B| / A/B` ranges from `0.1%` to `26.7%`,
with 15 of 45 `(row, tau)` cells above 6%, and the paired per-rollout difference is significant at up to
`+10 sigma` (s3p20). `D/B` is therefore **systematically smaller** than `A/B`, as expected from
Jensen's inequality (`E||draw|| >= ||mean||`). An earlier claim that `D/B` is a proxy for `A/B`
was based on the four cells with the smallest difference and is withdrawn (ledger #105).

## 9. Errata to earlier notes (these notes are not modified)

- `notes/Note10.md:199-201` states the `medA/medB` and `medRatio` definitions with the bracket order
  reversed. The code citation at `notes/Note10.md:214` is correct.
  `medRatio = np.median(av / np.maximum(bv, 1e-9))` is a paired-per-rep median, not median-of-marginals.
- `notes/Note10.md:12` and `notes/Note10.md:151` use "16x" / "16.5x" for the *cold-vs-warm*, within-episode
  ratio. That is a different quantity from this note's tau-resolved *cross-family* gap (13x -> 19x).
  The two must not be conflated.
- `notes/Note8.md:45` (Section 4: a randomly re-drawn a-slice responds *more* than the trained one) is
  consistent with Table 5 here: destroying the action channel raises the t = 1 reading.
  `notes/Note8.md` lines 58-60 already note that the 0.111 figure lost its evidential status.

## 10. Withdrawn and non-reproducible quantities

- `brief-x` (an externally supplied x column) has no provenance in this repository and cannot be
  recomputed; it is abandoned. The quantity used here is `x := NLL(lambda=0) - NLL(lambda=1)`.
- Two columns circulated in the interim v5 summary are not reproducible as `(tau = 1, t = 1, n = 20)`
  aggregates and are withdrawn: the `A/B` column (6/15 rows disagree with the table) and the `D(det)`
  column (13/13 rows are single-cell raw values, not aggregates). Note 11 uses `tmp_v5_table.csv` only.
- `content.md` does not exist; the reference to it in `notes/Note8.md:98` is dangling.

## 11. Settings that were never persisted

`drive_frac`, `rho` and `seed` were used to build the synthetic latents but are written neither into
`index.json` nor into any checkpoint `args`. For the *unsuffixed* synthetic families (`_synth`,
`_synth2`, `_synth3`) the action files are byte-for-byte copies of the source directory
(`actions.npy` md5 `18401300a2dd`), which explains the cross-family md5 collisions.

The `synth3pXX` dose series differs and needs three corrections, all recovered by linear
decomposition of the frozen arrays (no metadata is trusted):

| dose | md5(latents) | md5(actions) | keep frac | rho |
|---|---|---|---|---|
| s3p10 | `f5433cf8e0de` | `a2b53a7c551e` | 0.1011 | 0.021 |
| s3p15 | `f5433cf8e0de` | `01bc52cd8437` | 0.1497 | 0.021 |
| s3p20 | `f5433cf8e0de` | `ce345e90b68d` | 0.2003 | 0.021 |
| s3p25 | `f5433cf8e0de` | `3a603392d0ed` | 0.2471 | 0.021 |
| s3p50 | `f5433cf8e0de` | `ae63e712ccd1` | 0.5017 | 0.021 |
| s3p75 | `f5433cf8e0de` | `17b84fb9cb7f` | 0.7501 | 0.021 |
| s3p90 | `f5433cf8e0de` | `821753ba675f` | 0.9000 | 0.021 |

- all seven doses share ONE byte-identical `latents.npy` (`f5433cf8e0de`); the dose lives entirely
  in `actions.npy`, and `mean|dz|` is `0.00275` for every dose.
- the labelled `drive_frac` is the **keep fraction** (the share of action rows left identical to
  the real episode; the rest are real rows permuted elsewhere). The measured keeps
  (`0.1011 .. 0.9000`) match the labels (`0.10 .. 0.90`) to 4 decimals.
- the mean reversion is `rho ~ 0.021`, consistent with `synth3` (`rho = 0.020`) and NOT with
  `_synth` (`rho = 0.90`); the `--noise-frac` / `drift == drive` caveats of `notes/Note9.md:107`
  therefore do not apply to this series.
- the generator that produced the `synth3pXX` series is **not in this repository**;
  `make_synthetic_latents.py` is excluded because it re-derives the latents as a function of
  `--drive-frac`, whereas these doses share one fixed latent array.

## 12. Credentials

| file | bytes | md5 (first 12) |
|---|---|---|
| `notes/artifacts/tmp_v5_protocol.csv` | 2566 | `5AC6BA494E7E` |
| `notes/artifacts/tmp_v5_cells.csv` | 458594 | `6ABF94F0FBA4` |
| `notes/artifacts/tmp_v5_table.csv` | 37743 | `77DDAD7821F0` |
| `notes/artifacts/tmp_v5_headline.csv` | 629 | `9B111421BA39` |
| `notes/artifacts/tmp_v5_notnote11.csv` | 1174 | `6F5444DF9612` |
| `notes/artifacts/tmp_v5_log.txt` | 72276 | `B119B3CA59F6` |
| `notes/artifacts/tmp_seed_summary.csv` | 635 | `805EBE1F414A` |
| `notes/artifacts/tmp_seed_dbread.csv` | 8140 | `2D6FFFC3EFA7` |
| `notes/artifacts/tmp_final_table_v2.csv` | 892 | `8537EEE7285A` |
| `notes/artifacts/tmp_final_provenance_v2.csv` | 19941 | `A1FCD28E76E6` |
| `notes/artifacts/manifest_ckpt.txt` | 16624 | `CF2B71FC7A54` |
| `notes/artifacts/tmp_n18_audit.txt` | 9600 | `DBF073AD22BF` |
| `notes/artifacts/tmp_n18_audit_p2.txt` | 5515 | `CDD8314A72C4` |
| `notes/artifacts/tmp_n18_audit_p3.txt` | 4393 | `9ED24045FF27` |
| `notes/artifacts/tmp_n18_credentials.csv` | 725 | `5BC8BB4A62FB` |
| `notes/artifacts/tmp_n18_deliverable.csv` | 6148 | `EF908588CDFA` |

## 13. Retraction ledger (this session: #47 - #109)

Notable entries: #105 (`D/B` is not `A/B`, Section 8); #108 (two interim v5 summary columns withdrawn,
Section 10); #97 / #98 (the `x = multi-seed lambda-ablation mean` and the normalised variant are both
falsified); #99 (`ckpt.args.latents` is not always the correct probe directory - the two structural rows
need an override, Section 2). The full ledger is kept in the session transcript.

## 14. Reproducibility certification

This note is the audit artifact for the dose-response claim. It is generated from frozen CSVs by
`scripts/mdn/gen_note11.py`, is independent of the training code, and reproduces the tau-resolved
headline from raw credentials alone.

