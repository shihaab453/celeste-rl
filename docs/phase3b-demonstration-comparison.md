# Phase 3B: room 1 with demonstrations

**Result: 82.5%, 98.0% and 98.5% success from the canonical start** across three seeds, 200 evaluation
episodes each. All three deterministic policies clear the room, and all three clear it faster than any
demonstration they learned from. The unshaped no-demonstration baseline reported in
[phase3-unshaped-baseline.md](phase3-unshaped-baseline.md) scored 0% over six million transitions.

This is the demonstration-assisted comparison the roadmap calls for when the no-demonstration budget is spent
without a clear. It does not replace that result, and it is not the same claim: this policy was shown solutions
and that one was not.

Run on 2026-09-18. Every number comes from a run whose commit and runtime were recorded and verified against
pinned hashes.

## 1. The comparison

| arm | canonical success rate | budget |
|---|---|---|
| RL only | **0%** | 3 seeds x 2,000,896 transitions |
| Cloning only | **6%** (3 of 50) | 7 demonstrations, 2,688 frames, 300 epochs on CPU |
| Cloning then RL | **82.5% to 98.5%** | the same clone plus 3 x 501,760 transitions |

Cloning alone clears the room rarely. RL alone never clears it. Cloning followed by RL clears it nearly always
on two seeds of three, and ends up faster than every demonstration it learned from.

Per seed, 200 evaluation episodes each from the canonical start, all from the same cloned policy with only the
reinforcement learning seed differing:

| seed | success | 95% interval | median clear | fastest clear | deterministic |
|---|---|---|---|---|---|
| 0 | 165/200 = **82.5%** | 76.6% to 87.1% | 336 frames | 210 | success |
| 1 | 196/200 = **98.0%** | 95.0% to 99.2% | 172 frames | 157 | success |
| 2 | 197/200 = **98.5%** | 95.7% to 99.5% | 206 frames | 193 | success |

The demonstrations run 285 to 519 frames. Every seed's median clear is faster than the shortest of them, and
seed 1's fastest clear of 157 frames is a little over half the shortest demonstration. The policies are not
replaying what they were shown.

## 2. Where the demonstrations came from, and what they cost

Seven complete clears of room 1, produced by `scripts/find_room_exit.py`, a Go-Explore style search that
replays savestates and consults the room's tile map. **No human played the game and no TAS was authored.** The
search is not a learning method and never runs during training; it is the same tool that produced the project's
original test fixture.

| | |
|---|---|
| Demonstrations | 7 |
| Frames | 2,688 |
| Lengths | 285 to 519 frames |
| Distinct openings | 6 of 7 |
| Inputs held per frame | 2.05 to 2.31 |
| Search cost | about 10 minutes of wall time per route, 7 searches, all successful |
| Cloning cost | about 2 minutes on CPU |

The routes differ in length by 50% and in 6 of 7 openings, so they are genuinely different solutions rather
than one rediscovered. That mattered: an earlier candidate dataset of 14 clears extracted from the training
archives turned out to be two routes with a median pairwise frame agreement of 0.89, and to hold 8.36 inputs
per frame rather than about 2.2, because they were the near-uniform policy's own solutions. Those were set
aside, and this report uses only the search routes.

## 3. Cloning

Behavioural cloning is supervised learning and never sees a reward. Five demonstrations were trained on and
**two whole demonstrations held out**, never individual frames: consecutive frames of one route are nearly
identical, so a frame-wise split measures memorisation.

| | input accuracy | always-zero baseline | frame accuracy | always-zero baseline |
|---|---|---|---|---|
| trained on | 0.9988 | 0.8947 | 0.9877 | 0.0406 |
| held out | 0.9423 | 0.8968 | 0.3973 | 0.0324 |

The baselines are what a policy scores by pressing nothing at all, and they are reported because the inputs are
sparse: held-out input accuracy of 0.9423 is only modestly better than 0.8968. Held-out frame accuracy of
0.3973 is the more informative number, and it says the cloned policy would press something different from the
demonstration on 60% of held-out frames.

Playing it: **3 clears in 50 episodes**, and the deterministic policy timed out. This is what distribution
shift looks like. A cloned policy has only ever seen states on the demonstrated path, so its first mistake puts
it somewhere no demonstration went, and errors compound.

## 4. Fine-tuning

PPO initialised from the cloned policy's weights (`rew-v2`, shaping scale 2.0, canonical starts, 501,760
transitions). The step counter, optimizer state and records all start fresh; only the weights are inherited.

Success rate during training, 50 episodes at each point:

| accepted steps | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 100,352 | **0%** | 44% | **0%** |
| 200,704 | **0%** | 92% | 12% |
| 301,056 | 16% | 94% | 100% |
| 401,408 | 72% | 100% | 96% |
| 501,760 | 84% | 98% | 100% |

Seeds 1 and 2 reach their final level by 300,000 transitions. Seed 0 is slower throughout and finishes lowest.

### The early collapse, which happens on some seeds and not others

Seeds 0 and 2 scored 0% at 100,000 transitions, and on seed 0 the median furthest-right position was 19.0,
which is the start position: the policy stood still and cleared nothing for 200,000 transitions, having been
handed a working policy. Seed 1 never collapsed at all, reaching 44% by the same point.

The cause is that **cloning never trains the value head.** The loss is computed from the action logits only, so
the critic stays at its initialisation while the shared feature extractor moves underneath it. PPO therefore
begins with a critic worse than predicting the mean (`explained_variance` -0.72 on seed 0's first update,
`approx_kl` 0.0399 against a steady state of about 0.96 and 0.015) and spends its early updates moving the
policy on arbitrary advantages.

That it happens on two seeds of three and not the third is worse than it happening every time: the damage is
unpredictable rather than a fixed cost. All three recovered here, so it cost time rather than results, but a
critic warm-up before allowing policy updates would remove both the delay and the ambiguity if a future run
ever ended below where it started.

**This was very nearly reported as a general property of fine-tuning.** It was measured on seed 0 alone, and
seed 1 contradicts it.

## 5. What this does and does not establish

**It establishes** that the task is learnable by this network, this observation and this action space, and that
the barrier in the no-demonstration branch was neither capacity nor the environment. Given episodes whose
returns differ, PPO improves on what it was shown by a wide margin, on every seed tried, and arrives at
solutions faster than the ones it was given.

**It does not establish:**

- **Generalisation.** Every episode here begins at the canonical start. The roadmap's Phase 3 criterion is 99%
  on 200 held-out reachable entry states, from a generator frozen before training. That generator does not
  exist yet, and this report makes no claim about it.
- **Reproducibility of the clone.** All three seeds start from the same cloned policy and differ only in the
  reinforcement learning seed, so this shows the fine-tuning is reproducible, not the pipeline end to end. A
  different demonstration set or a different cloning seed is untested.
- **That the seed spread is understood.** 82.5% against 98.5% is a wide gap for a difference of one seed, and
  nothing here explains it beyond the early collapse costing seed 0 roughly 200,000 transitions.
- **Anything about the no-demonstration question.** That result stands as published. This policy was shown
  seven solutions.
- **Speedrun-quality play.** Medians of 172 to 336 frames are clears, not good times. Faster than the
  demonstrations is a low bar, since those came from a random search.

## 6. A note on the seeds

The first attempt at seeds 1 and 2 produced byte-identical episode logs. Stable-Baselines3's `load()` runs
`_setup_model()`, which calls `set_random_seed` with the donor checkpoint's saved seed and re-seeds torch,
numpy and python globally, after the run's own model has been seeded. Every run initialised from a checkpoint
therefore sampled from the donor's stream whatever seed it was given. The runs above are from after that was
fixed. Seed 0's original run is unaffected, because it was configured with seed 0 and the donor was saved with
seed 0, so it sampled from the stream it claimed to.

## 7. Reproducing this

```
.venv-rl/Scripts/python.exe scripts/find_room_exit.py --seed <n>            # a demonstration, needs the game
.venv-rl/Scripts/python.exe scripts/clone_room1.py --routes-only            # clone and play it
.venv-rl/Scripts/python.exe scripts/train_room1.py --seed 0 --reward-version rew-v2 --shaping-scale 2.0 \
    --init-from runs/clone/<run>/cloned.zip --total-timesteps 500000
.venv-rl/Scripts/python.exe scripts/evaluate_checkpoint.py --checkpoint <run>/checkpoints/latest.zip --episodes 200
```

Every live script refuses to start from a dirty working tree or a runtime that differs from
`config/pinned_runtime.json`, so a result can always name the code and the game build that produced it.
Success rates are reported with Wilson score intervals, because a rate from a finite sample is not a point.
