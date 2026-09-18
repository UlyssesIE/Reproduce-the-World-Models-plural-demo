# Reproduce-the-World-Models-plural-demo

## Deviation Log

Findings where this reproduction knowingly differs from Ha & Schmidhuber (2018),
or where the original paper is underspecified. Each entry records what was
observed, the root cause, and the decision taken.

### Note1 — CarRacing termination semantics (v3 vs. v0)

**Status:** open, workaround decided
**Date:** 2026-09-14
**Affects:** data collection, episode length, dataset size

#### What the paper does

The paper trains on `CarRacing-v0` (OpenAI Gym). In v0 an episode ends as soon
as the car leaves the track, so a uniform-random policy terminates after a few
hundred steps and produces a highly concentrated, on-track dataset.

#### What we observed

Pilot run of 100 random-policy rollouts (`scripts/collect_data.py`):

| Metric | Value |
| --- | --- |
| `mean_steps` | **1000.0** (exactly the `max_episode_steps` cap) |
| `mean_return` | -32.78 |
| Return across batches 25/50/75/100 | -33.7, -34.1, -33.4, -32.8 (spread < 1 point) |
| Disk, 100 rollouts | 2.8 GB |
| Wall clock | 8.9 s / rollout |

#### Root cause

`gymnasium`'s `CarRacing-v3` uses a **different** termination rule than v0. Per the
official documentation:

> "The episode finishes when all the tiles are visited. The car can also go
> outside the playfield - that is, far off the track, in which case it will
> receive -100 reward and die."

The key qualifier is **"far off the track"**: the car must leave the *playfield
boundary*, not merely the track. Under random actions the car typically drifts
onto the grass but never leaves the playfield, so **no episode ever terminates
early** — all 100 rollouts ran to the 1000-step truncation limit.

#### Why this matters

1. **Episode length is inflated ~3–7x** relative to v0 semantics. Dataset size and
   collection time scale with it: naive extrapolation to 10,000 rollouts gives
   **~276 GB and ~25 h** — roughly 7x the initial estimate of 41 GB (that
   estimate assumed v0-like episode lengths and was wrong).
2. **Low information density.** Decomposing the return: time penalty is
   -0.1 x 1000 = -100, so tile rewards contribute only ~+67 against a full-lap
   total of ~1000. Only ~7% of each episode is on-track driving; the remaining
   ~930 frames are the car idling on grass.
3. **Unnaturally low variance.** The near-identical returns across batches suggest
   every episode follows a very similar (degenerate) trajectory, which weakens
   the diversity argument for the random-policy dataset.
4. **No `done` signal.** `terminated` is never `True`. The paper's CarRacing M is
   not required to predict `done` (only the VizDoom variant is), so this does not
   block the V/M/C pipeline — but it would block the iterative training procedure
   of §5.

#### Decision

- Size the dataset by **frame budget**, not rollout count. Target ~2M frames
  (= **2,000 rollouts** under v3, ~55 GB raw, ~5 h) instead of 10,000 rollouts
  (~2.8M frames would need only ~2,800 rollouts; the original 10,000 is both
  unnecessary and unaffordable here).
- Keep **raw 96x96x3 uint8** on disk so preprocessing can be revisited without
  re-simulating; `np.savez_compressed` compression ratio to be measured.
- Optionally add an off-track early-termination wrapper to restore v0-like
  episode semantics (also restores a meaningful `done` label). Postponed
  pending the trajectory inspection below.

#### Verification

```bash
python scripts/collect_data.py --out data/pilot --num-rollouts 100
python scripts/verify_dataset.py --data data/pilot
# expected: terminated 0 / 100, truncated 100 / 100
```

To reproduce this entry, inspect a rollout strip:
`artifacts/pilot_strip.png` (frames sampled from one random-policy episode).

#### Disclosure

Any downstream comparison against the paper's numbers must state that data was
collected under **v3 termination semantics**, with episodes several times longer
than the paper's, and that this difference was not corrected unless the
early-termination wrapper is enabled.
