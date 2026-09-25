# Room 2 training stability

Everything in this report is training side: runs are measured from the canonical Room 2 start, and nothing here
reads a held-out set. The fresh Room 2 held-out set (`config/heldout_starts-room2-v2.json`) has still never been
evaluated.

## Summary

- An attempt to test whether more demonstration routes help in Room 2 failed before any held-out test, because
  fine-tuning became unstable: many runs ended stuck before the room's hardest crossing.
- Most of these failures were runs that **never learned the crossing**, not runs that learned it and lost it.
- A training-only "stall" ending (an episode ends as an ordinary failure once it stops making progress) passed a
  small screen: with it, 1 of 8 runs from the same clone failed, against 5 of 8 without it.
- A second attempt at the more-routes question, with the stall ending in both arms and eight clones per arm,
  **failed its predeclared stability gate**: 2 of 8 runs failed from 7-route clones and 0 of 8 from 21-route clones,
  where at most 1 per arm was allowed. So it was not evaluated on held-out states, and the question stays open.

## Why this came up

The [Room 2 matched comparison](room2-matched-finetuning.md) found that both arms fail in one short stretch of the
room: a pit, a spike-topped block and a spiked gap between world x 360 and 459, which every demonstration crosses
with the help of a spring the policy cannot see. The next experiment asked whether a clone fitted on more
demonstration routes would fine-tune better.

Its plan (`3cabed1`) compared fine-tuning from the existing clone (arm A: 7 routes, 5 fitted) with fine-tuning from
a new clone (arm B: 21 routes, 16 fitted), eight matched seeds (30 to 37), canonical starts, `rew-v2`, shaping scale
2.0 and 500,000 transitions. The held-out evaluation was to be planned after a review. Training went wrong first:
many runs ended stuck, repeatedly timing out before the crossing.

## How a run is judged here

Each run's final checkpoint plays 50 stochastic episodes from the canonical start at seed 20260924, without any
training-only option. A run **collapsed** if it cleared fewer than 25 of the 50. Training clear rates below come
from each run's own training episodes, grouped into 100,000-step windows.

## The more-routes attempt collapsed

| Runs (no stall ending) | Clone | Final play, clears of 50 | Collapsed |
|---|---|---|---:|
| Original arm A, seeds 20 to 25 (the matched comparison) | 7-route, seed 0 | 46, 50, 41, 46, 42, 41 | 0 of 6 |
| More-routes arm A, seeds 30 to 37 | 7-route, seed 0 | 19, 0, 0, 1, 46, 47, 50, 16 | 5 of 8 |
| More-routes arm B, seeds 30 to 33 | 21-route, seed 0 | 0, 0, 0, 0 | 4 of 4 |

Arm B's seeds 34 to 37 were never played this way. On the training side none of them cleared more than 8% of its
episodes in any 100,000-step window.

Arm A of the more-routes attempt used the same clone and settings as the original arm A. Only the seeds and the
checkpoint interval differ. An exact rerun of original seed 20 at the later code, with the later checkpoint interval,
reproduced its first 143,360 steps episode for episode, so neither the code changes nor the checkpoint interval
explain the difference in that stretch. Every collapsed no-option run in the new batch had its first timeout inside
that stretch (between 8,192 and 32,768 steps). If collapse were equally likely in both batches, all five collapses
landing among the eight new seeds would happen about 3% of the time by this measure (7% by the training-episode
measure). This was one of several comparisons made after the collapses were seen, so chance is the most likely
explanation. No other cause was found.

## Clone replication

Each arm had used a single clone, so the clone fit was repeated with seeds 1 to 7 for both route sets. The seed
decides which routes are held back and how the network starts. Canonical play before fine-tuning, clears and
timeouts of 50:

| Clone seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Total of 400 |
|---|---|---|---|---|---|---|---|---|---|
| 7 routes | 10 / 0 | 11 / 2 | 12 / 9 | 4 / 3 | 14 / 5 | 20 / 4 | 14 / 1 | 1 / 4 | 86 clears, 28 timeouts |
| 21 routes | 5 / 8 | 11 / 11 | 8 / 3 | 22 / 2 | 16 / 8 | 15 / 11 | 18 / 9 | 4 / 5 | 99 clears, 57 timeouts |

Clone-to-clone variation (1 to 22 clears) is as large as any difference between the route sets, and arm B's
seed-0 clone was one of the weakest 21-route clones. One clone per arm cannot separate the route set from the
particular fit.

## The stall ending

The stall ending is a training-only option. When it is on, an episode ends as `stalled` once its best progress
toward the exit has not improved for 296 frames. That is about twice the longest such stretch in any of the 21
demonstrations. The value was set after the no-option runs' episodes had been seen (the plan says so; a first
rule based on those episodes gave 815). A stalled episode is an ordinary failure with the same return as a death or a timeout. The
evaluators never turn it on, and every result in this report is played without it.

The option does not reward attempts over stalling. It makes a stuck episode end sooner, so a stuck run gets more
attempts per 100,000 steps. For example, frames per episode in the last 100,000 steps were 368 with the option
against 1,760 without it for seed 31, and 430 against 1,478 for seed 32.

## Stability pilot (a screen)

The pilot plan (`0681f79`) trained the more-routes attempt's arm A seeds 30 to 37 again with the stall ending
(same clone, same seeds). The screen passed if at most 1 of the 8 collapsed. Four arm B seeds (30 to 33, 21-route
seed-0 clone) were added as an exploratory group outside the screen.

| Seed | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A with the stall ending | 44 | 45 | 48 | **0** | 33 | 46 | 50 | 33 |
| A without it (above) | **19** | **0** | **0** | **1** | 46 | 47 | 50 | **16** |
| B with the stall ending | 34 | 41 | 41 | 49 | | | | |
| B without it (above) | **0** | **0** | **0** | **0** | | | | |

**The screen passed:** 1 collapse in 8 with the option against 5 in 8 without (one-sided Fisher p = 0.059). In
the exploratory group it was 0 in 4 against 4 in 4 (p = 0.014). This is a screen, not a confirmatory result, and
it has these limits:

- The runs without the option were chosen by their outcome: the pilot exists because they collapsed. Pooled with
  the original arm A, runs without the option collapsed 5 times in 14. At that rate a screen of 8 passes about 16%
  of the time even with no effect.
- Runs with and without the option share a seed but go their own way within 1 to 28 episodes, so a seed-by-seed
  "rescue" cannot be claimed. Only the group counts can be compared.
- Seed 34 cleared fewer with the option (33 against 46 of 50; two-sided Fisher p = 0.0026). Clears with the
  option also took longer in 4 of the 5 seeds where both versions cleared at least 16 of 50 (median clear length
  for seed 36: 453 frames against 363). The pilot's harm check compares only the healthy runs' medians, so it
  cannot flag harm confined to a few healthy seeds.
- The 296-frame threshold was chosen after seeing the no-option runs (above).
- Game throughput dropped by about 30% partway through the pilot. Two of the no-option checkpoints were
  replayed afterwards, and all 51 episodes of each matched the earlier plays exactly, so the drop did not change
  results.

## H2: more routes with replicated clones

The second more-routes attempt (plan `37b9431`, the four runs cut short by the night's time limit finished from
`4b3b09a`, play plan `6c4a5b4`) paired eight clone seeds k from 0 to 7:

- Arm A fine-tuned from 7-route clone seed k, which fits 5 of the 7 routes.
- Arm B fine-tuned from 21-route clone seed k, which fits 16 of the 21.
- Both arms used the stall ending, fine-tuning seed 40 + k, canonical starts, `rew-v2`, shaping 2.0 and 500,000
  transitions.
- A held-out evaluation was allowed only if at most 1 of 8 runs per arm collapsed (the gate, applied by a pinned
  script).

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (7-route clone k) | 47 | 43 | 37 | **0** | 48 | 49 | 50 | **0** |
| B (21-route clone k) | 48 | 39 | 48 | 48 | 47 | 47 | 49 | 42 |

**The gate failed:** arm A collapsed 2 times in 8 and arm B 0 times in 8. As declared, there was no held-out
evaluation. The one-sided Fisher p for 2 of 8 against 0 of 8 is 0.23: no evidence that the route sets differ in
how often they collapse. The healthy runs cleared 74% to 100% (A) and 78% to 98% (B). Both collapsed runs time out
on nearly every episode (48 and 47 of 50), stuck near x 348, just before the crossing.

**B-k7 nearly failed as well.** Its training clear rate stayed at 6% or below for its first 400,000 steps and rose
to 46% only in the last 100,000. The final checkpoint caught that late recovery (42 of 50).

Disclosure on the clones: the seed decides which routes each clone holds back. A 7-route clone's fitted routes are
a subset of its paired 21-route clone's fitted routes only for k = 0 and k = 2. Arm A has only 6 distinct route
sets, because seeds 1 and 7 hold back the same two routes, and so do seeds 2 and 3. Pairing by k shares the clone
seed and the fine-tuning seed, not the data, so it is nominal. Held-back routes (route folder names):

| k | 7-route clone holds back | 21-route clone holds back |
|---|---|---|
| 0 | 164006, 164552 | 164006, 164552, 013324, 013921, 014025 |
| 1 | 021356, 164712 | 163841, 013144, 013423, 013826, 014213 |
| 2 | 164712, 164815 | 013423, 013605, 013727, 013921, 014051 |
| 3 | 164712, 164815 | 164111, 013226, 013249, 013921, 013952 |
| 4 | 163841, 164006 | 021356, 013144, 013249, 013826, 014051 |
| 5 | 163841, 164552 | 164111, 013144, 013605, 013826, 014213 |
| 6 | 164006, 164111 | 163841, 164006, 164552, 014051, 014213 |
| 7 | 021356, 164712 | 164111, 013226, 013324, 013637, 013826 |

Route 021356 is `20260920-021356`, the 1638xx to 1648xx routes are from 2026-09-22 and the 013xxx and 014xxx
routes from 2026-09-24. The full names are in `docs/results/room2-training-stability.json`.

## How runs fail: slow starts, not lost skill

Training clear rate per 100,000-step window shows what the final snapshot hides. Almost every run that ended
collapsed never learned the crossing in the first place:

| Group | Stall ending | Reached a 50% training clear rate within 500,000 steps | First reached at | Collapsed in final play |
|---|---|---:|---|---:|
| Original arm A | no | 6 of 6 | 150k to 300k | 0 of 6 |
| More-routes arm A | no | 4 of 8 | 150k to 450k | 5 of 8 |
| More-routes arm B | no | 0 of 8 | | 4 of 4 played |
| Pilot A | yes | 7 of 8 | 150k to 400k | 1 of 8 |
| Pilot B | yes | 4 of 4 | 250k to 500k | 0 of 4 |
| H2 arm A | yes | 6 of 8 | 150k to 300k | 2 of 8 |
| H2 arm B | yes | 8 of 8 | 150k to 500k | 0 of 8 |

"First reached at" is the end of the first 50,000-step window whose clear rate was at least 50%.

- Every collapsed run except one never passed 26% in any 100,000-step window; one of them (seed 30 without the
  option) was only starting to rise at the end (33% in its last 50,000 steps). The exception is seed 37 without
  the option: it cleared 71% to 96% of its training episodes from 100k to 400k, 74% in the last window (55% in the
  last 50,000 steps), and then 16 of 50 in final play.
- Most collapsed runs froze just before the crossing, near x 348. With the option, pilot seed 33 failed
  differently: most of its late stalls never got past the room entry (113 of its last 170 stalls stayed at x 263
  or less).
- Runs that did get going sometimes did so late: 5 of the 35 runs that reached 50% first did so at 400,000
  steps or later. So a single checkpoint at 500,000 steps partly measures whether a run got going before the deadline.
- An early stall share does not reliably predict failure. Among runs with the option, the ones that never got
  going, or only in their last window, stalled in 36% to 56% of their first 100,000 steps' episodes, but H2's
  B-k5 stalled in 50% and then learned.
- The weak clones do not freeze before the crossing on their own. Before fine-tuning, the three weakest clones
  (4, 1 and 4 clears of 50) mostly failed at the crossing: the furthest point reached (`max_x`) was between x 360
  and 459 in 41 of 46, 46 of 49 and 40 of 46 failures, nearly all deaths (41, 42 and 40). The freeze just before
  it was learned during fine-tuning.
- Among runs with the option, the three clones with 4 or fewer clears of 50 started slowly (H2's A-k3 and A-k7
  never reached 50%, B-k7 only in its last window). Among the 25 runs with the option from stronger clones, two did
  (pilot A33 never, pilot B30 only in its last window). This pattern was found after the fact, in 28 runs. It is a
  hypothesis for a new predeclared test, not a finding.
- Policy entropy measures how random the policy's choices are. In H2, every 7-route run ended with higher
  entropy (0.51 to 1.02 over the last 100,000 steps) than every 21-route run (0.34 to 0.48); in the first 100,000
  steps the arms overlap. In the pilot the groups overlap slightly (A 0.51 to 1.17, B 0.45 to 0.58). The collapsed
  runs are not the extremes, so entropy alone does not mark a collapse.

## What this does and does not show

- It shows that fine-tuning in Room 2 often fails to get going within 500,000 steps, and that the stall ending
  passed a small screen for reducing collapse in final play.
- It does not show that the stall ending is harmless: one healthy seed fell from 46 to 33 of 50 with it, and
  clears took longer in most seeds.
- It does not show that the stall ending improves held-out success. No run with the option has been evaluated on
  held-out states.
- It does not show that more routes prevent collapse. With the option, 21-route clones collapsed 0 times in 12
  and 7-route clones 3 times in 16 (one-sided Fisher p = 0.17).
- "0 of 12" is less safe than it sounds: 6 of those 12 runs first reached a 50% training clear rate only at 350,000
  steps or later (pilot B30, B31, B32; H2 B-k4, B-k5, B-k7). With a shorter budget or a less lucky final snapshot,
  several would have counted as collapses. Any plan that carries this combination forward needs a larger training
  budget or a predeclared rule for runs that have not started by a fixed step.
- It does not show that weak clones cause collapse.
- The more-routes held-out question is still open. Answering it needs both arms to train stably first.

## Scope and audit

All 50 runs in this report fine-tuned from the canonical Room 2 start for 501,760 accepted transitions, each in one
session with no faults or discarded rollouts. Every played final checkpoint was found by its SHA-256 among the recorded
plays. Where a checkpoint was played more than once in the same way, every play agreed episode for episode. The
stall pilot's screen and the H2 gate were computed by their pinned scripts (analysis blob `f2124c8`, gate blob
`257b718`). The H2 play plan records which campaign and commit trained each checkpoint. A separate check confirmed
the play commit, every result and episode file hash and every checkpoint hash.

The machine-readable summary, including every run's play, training windows, entropy, the clones' held-back routes
and the SHA-256 of every input, is `docs/results/room2-training-stability.json`, written by
`scripts/summarize_room2_stability.py`. Run records remain local under `runs/`.
