# Note2 — CarRacing data collection: source-level investigation and correction of Note1

**Status:** closed (decision revised)
**Date:** 2026-09-15 → 2026-09-16
**Affects:** data collection, episode semantics, dataset sizing, VAE implementation
**Supersedes:** the root-cause and impact analysis of Note1

---

## 1. Summary of what changed since Note1

Note1 explained the observed data characteristics with a **version difference**:
it asserted that `CarRacing-v0` ends an episode as soon as the car leaves the
track, whereas `v3` only ends it when the car goes *far off the track*.

**That claim is wrong.** Reading the actual environment source shows no
off-track termination in either version. The observed behaviour has a different
and now fully quantified cause:

> Under a uniform-random policy the car creeps along the track at ≈3.1 units/s.
> Over the 1000-step horizon it can travel at most ≈62.6 units, while reaching
> the game-over boundary requires a radial displacement of ≥108.3 units.
> **Early termination is therefore geometrically impossible — independent of
> how random the actions are.**

Three further corrections to Note1 are documented below (§3, §5, §6).

---

## 2. Erratum: Note1's central claim is unsupported

Note1 stated:

> "In v0 an episode ends as soon as the car leaves the track..."

Direct inspection of `CarRacing.step()` (gymnasium v3) shows the **only** two
termination conditions:

```python
if self.tile_visited_count == len(self.track) or self.new_lap:
    terminated = True                     # lap completed
x, y = self.car.hull.position
if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
    terminated = True
    step_reward = -100                    # left the playfield
```

There is **no track-proximity check anywhere**. The game-over boundary is the
circle of radius `PLAYFIELD` centred on the origin — not the edge of the track.

The environment's own version history documents only changes to *lap-completion
logic* and *domain randomization* (`v1`), and lap truncation→termination (`v2`).
**No release ever changed an off-track rule**, because none exists. Note1's
premise therefore has no basis.

Two related facts also require correction:

* `truncated` is **always `False`** inside the environment. The 1000-step
  truncation observed in our data comes from the `TimeLimit` wrapper we added
  ourselves in `make_env(max_episode_steps=1000)` — it is not environmental.
* The paper specifies a horizon of 1000 steps. **Our 1000-step truncation is
  therefore consistent with the paper**, not a deviation introduced by us.

---

## 3. Erratum: the car is not idle, and it is on the track

Note1 claimed that "the remaining ~930 frames are the car idling on grass"
(≈93% low-value frames). Both halves of this are wrong.

Measured over 5 seeds of a uniform-random policy:

| seed | steps | outcome | tiles | max_dist | final_dist | mean_spd | max_spd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1000 | truncated | 19 | 63.7 | 63.7 | 3.19 | 5.88 |
| 1 | 1000 | truncated | 18 | 59.7 | 59.7 | 2.99 | 5.87 |
| 2 | 1000 | truncated | 19 | 60.9 | 60.9 | 3.05 | 6.09 |
| 3 | 1000 | truncated | 18 | 57.9 | 57.9 | 2.90 | 5.71 |
| 4 | 1000 | truncated | 18 | 60.7 | 60.7 | 3.04 | 6.31 |

At `FPS = 50`, path length = `mean_spd × steps / 50 ≈ 3.13 × 20 = 62.6`, while
net displacement is 62.5 → **path efficiency 94%**, i.e. an almost perfectly
straight, monotonic trajectory. This is corroborated by `max_dist == final_dist`
holding exactly for 5/5 seeds: the maximum distance from the start occurs at the
final step, which is only possible if the displacement never decreases.

Visual inspection (`artifacts/pilot_strip.png`) confirms the car stays **on the
track** throughout. It is not idling on grass.

> The correct description is: **the car drives monotonically along the track at
> roughly 1/20 of the speed required to complete a lap.** Note1's "idling on
> grass" framing is withdrawn.

---

## 4. Why no episode terminates: a geometric proof

Environment constants, read from source:

| Constant | Value |
| --- | --- |
| `FPS` | 50 |
| `SCALE` | 6.0 |
| `TRACK_RAD` | 150.0 |
| `PLAYFIELD` | 333.333 |
| `TRACK_DETAIL_STEP` | 3.5 |
| `ZOOM` | 2.7 |
| `len(track)` | 319 |
| → approximate track length | 319 × 3.5 ≈ **1116 units** |

The start position is `(1.5 × TRACK_RAD, 0)` = `(225, 0)`, i.e. at radius 225
from the origin. Reaching the boundary needs a radial displacement of

```
333.33 − 225 = 108.3 units
```

The maximum displacement achievable within the horizon is

```
3.13 units/s × (1000 / 50) s = 62.6 units
```

**62.6 < 108.3.** Even a car driving perfectly radially outward for the entire
episode cannot reach the boundary. Hence `terminated` is never `True` for this
policy, and all episodes end in truncation.

### Why random steering does not produce erratic motion

The steering joints are mechanically constrained:

```python
revoluteJointDef(..., lowerAngle=-0.4, upperAngle=+0.4, motorSpeed=..., ...)
# Car.step():
w.joint.motorSpeed = dir * min(50.0 * val, 3.0)     # ≤ 3 rad/s
```

Steering is limited to ±0.4 rad (≈23°) and the joint motor is capped at 3 rad/s,
while actions are resampled every step (0.06 s at 50 Hz). The joint cannot follow
randomly flipping commands, so consecutive steering commands **cancel out** and
net steering is ≈0 — exactly the straight-line trajectory implied by
`net/path = 1.00`.

### Counter-evidence that termination works

Faster policies do terminate, confirming the mechanism is intact:

| Policy | mean steps | note |
| --- | --- | --- |
| uniform | 1000.0 | never reaches boundary |
| `signed(−0.5, 1)` | 941 | terminates in some seeds |
| `signed(−0.2, 1)` | 691 | terminates in most seeds |

---

## 5. Erratum: the mechanism is not an `engineForce` overwrite

An intermediate hypothesis held that `Car.brake()` overwrites the value set by
`Car.gas()` (an "engineForce overwrite"). **This is wrong** — neither function
touches `engineForce`. The correct mechanism is that **both act on the same
per-wheel angular velocity `w.omega`**, in opposite directions:

```python
# Car.step(): per wheel, per physics step
w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (abs(w.omega) + 5.0)

if w.brake >= 0.9:
    w.omega = 0
elif w.brake > 0:
    w.omega += -np.sign(w.omega) * min(15 * w.brake, abs(w.omega))
```

With `SIZE = 0.02`: `ENGINE_POWER = 4.0e4`, `WHEEL_MOMENT_OF_INERTIA = 1.6`,
`dt = 0.02`. Two asymmetries compound:

1. **Coverage asymmetry.** `gas()` applies only to the rear two wheels
   (`self.wheels[2:4]`), whereas `brake()` applies to **all four**. Under uniform
   sampling `brake ~ U(0,1)` is essentially always positive, and in 10% of steps
   (`brake ≥ 0.9`) it zeroes `w.omega` outright.
2. **Rate asymmetry in `gas()`.** Rise is rate-limited, fall is not:
   ```python
   diff = gas - w.gas
   if diff > 0.1:
       diff = 0.1          # "gradually increase, but stop immediately"
   w.gas += diff
   ```
   Since the target is resampled from `U(0,1)` every step, `w.gas` can drop
   instantly but can only climb by 0.1 per step, so its steady-state mean is
   pulled well below 0.5.

Together these explain the measured ≈3.1 units/s equilibrium.

---

## 6. Erratum: speeding up does not improve data quality

Four action distributions were compared (5 seeds each):

| strategy | steps | tiles | mean_spd | max_spd | net_dist | net/path |
| --- | --- | --- | --- | --- | --- | --- |
| **uniform** | 1000 | **18.8** | 3.13 | 6.24 | 62.5 | 1.00 |
| full_gas | 1000 | **19.6** | 22.82 | 97.95 | 252.7 | 0.56 |
| `signed(−0.5, 1)` | 941 | 15.4 | 10.22 | 23.24 | 173.8 | 0.91 |
| `signed(−0.2, 1)` | 691 | 10.4 | 41.39 | 64.66 | 162.8 | 0.38 |

Interpretation:

* `full_gas` (22.82 units/s vs 3.13) confirms the gas/brake mechanism of §5.
* **Tile coverage is essentially independent of speed** (18.8 → 19.6 → 15.4 →
  10.4). Faster policies cover *fewer* tiles, because random steering cannot
  follow curves: the faster the car, the sooner it leaves the track.
* `net/path` falling from 1.00 to 0.38 marks the transition from "driving along
  the track" to "wandering on grass after leaving it".

The car traverses `62.5 / 1116 ≈ 5.6%` of the track (≈19 of 319 tiles).

**Conclusion: within the space of random policies there is no tunable knob.**
Uniform sampling is already the best available choice on the metric that matters
(track coverage), and it is also the faithful reading of the paper, which states
only that data was collected using "an agent acting randomly to explore the
environment". Several independent reimplementations likewise use uniform
sampling.

---

## 7. Note on the compression ratio

A single rollout is 27.7 MB raw and 1.3 MB after `np.savez_compressed` — a
**21.67×** ratio. This is not prima facie evidence of a broken dataset: the
frames contain large flat colour regions (grass, asphalt), and with 1000 highly
similar frames per episode the ratio is expected to be high. It remains a useful
**cheap proxy for data diversity**: after any change to the collection policy,
re-measuring this number gives an immediate signal.

---

## 8. Revised decision

Note1 decided to proceed directly to collecting 2,000 rollouts (≈5 h). **That
ordering is reversed**, on the grounds that it front-loads the expensive step
before the cheap validation step:

1. **Train V and M on the existing 100 pilot rollouts first** (≈100k frames —
   ample for a small VAE). No new collection.
2. **Generate dream rollouts** — run M autoregressively, decoding each sampled
   `z` through V — and inspect: does the sequence show a coherent track and a
   moving car, or does it collapse?
3. **Only then decide the dataset scale.** If the dream has structure at 100k
   frames, 2,000 rollouts is not obviously needed.

Validation criteria:

| Dream reconstruction | Reconstruction grid | Decision |
| --- | --- | --- |
| structured, varies with action | sharp | proceed; extend only if M plateaus |
| collapses to a blur after a few steps | sharp | re-train M with more data, or adopt iterative collection (§5 of the paper) |
| — | already blurry | fix the VAE (β, epochs); unrelated to data volume |

An off-track early-termination wrapper is **not** adopted: the environment has
no such mechanism, and the paper already anticipates this limitation and
addresses it through *iterative training* (§5), stating that a single iteration
sufficed for simple tasks.

---

## 9. Implementation notes: VAE (V model)

Code: `src/models/vae.py`, `src/data/frames.py`, `scripts/train_vae.py`,
`scripts/recon_grid.py`.

**Specified by the paper:** input 64×64×3; `z ∈ R³²`; reconstruction is lossy;
V and M are trained separately from C.

**Not specified by the paper (our choices, to be reported as deviations):**

| Item | Our choice |
| --- | --- |
| conv depth / channels / kernel | 4 layers, 32/64/128/256, k=4, s=2 |
| reconstruction loss | Bernoulli (BCE with logits) |
| β (KL weight) | 1.0, with optional linear warm-up |
| free bits | off by default, configurable |
| output activation | sigmoid (paired with BCE) |

**Acknowledged parameter-count gap:** the paper reports **4,348,547** parameters
for the CarRacing V, while the default configuration here yields ≈**3.5 M**. The
training script prints this delta explicitly. The paper does not describe the
architecture in sufficient detail to reproduce the count, so this gap is
recorded rather than concealed.

---

## 10. Hypothesis ledger (how the diagnosis actually evolved)

| Version | Hypothesis | Status | What overturned it |
| --- | --- | --- | --- |
| v1 (Note1) | v0 terminates off-track, v3 only far off-track | **wrong** | source: no off-track rule in either version |
| v2 | the car is idling on grass (~93% of frames) | **wrong** | visual check: car is on track; `path efficiency 94%` |
| v3 | the car barely moves at all | **wrong** | it moves 62.5 units monotonically |
| v4 | `brake()` overwrites the value set by `gas()` | **wrong** | source: neither touches `engineForce` |
| v5 (this note) | gas and brake fight over `w.omega`; two rate/coverage asymmetries; termination geometrically impossible | **source-verified** | — |

This ledger is retained deliberately: the errors are part of the record, and each
was resolved by reading source or measuring, never by assertion.

---

## 11. Environment constants (for reference)

```python
FPS = 50
SCALE = 6.0
TRACK_RAD = 900 / SCALE            # 150.0
PLAYFIELD = 2000 / SCALE           # 333.333
TRACK_DETAIL_STEP = 21 / SCALE     # 3.5
TRACK_WIDTH = 40 / SCALE           # 6.667
SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE**2 # 4.0e4
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE**2   # 1.6

# action_space (continuous): Box([-1, 0, 0], [+1, +1, +1])  # steer, gas, brake
# steer: -1 = full left, +1 = full right; gas and brake each in [0, 1]
```

Verification commands used:

```bash
python -c "import gymnasium.envs.box2d.car_racing as m; print({k: getattr(m,k) for k in ['FPS','SCALE','TRACK_RAD','PLAYFIELD','ZOOM']})"
python -c "import gymnasium as gym; e=gym.make('CarRacing-v3', continuous=True); e.reset(seed=0); print(len(e.unwrapped.track))"
python -c "import inspect, gymnasium as gym; e=gym.make('CarRacing-v3', continuous=True); print(inspect.getsource(e.unwrapped.step))"
python -c "import inspect, gymnasium.envs.box2d.car_racing as m; print(inspect.getsource(m.Car.gas), inspect.getsource(m.Car.brake))"
```

---

## 12. Disclosure

Data is collected with a **uniform-random policy over `Box([-1,0,0],[1,1,1])`**,
using **`CarRacing-v3`** with a **self-imposed 1000-step `TimeLimit`** (matching
the paper's stated horizon). The paper reports **10,000** rollouts; this
reproduction uses **fewer, sized by frame budget**, and any comparison against
the paper's numbers must state this. Episodes end in truncation rather than
termination; this is consistent with the paper's modelling scope, since the
CarRacing variant of M is not required to predict `done` (only the VizDoom
variant is).
