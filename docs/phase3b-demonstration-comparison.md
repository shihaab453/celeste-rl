# Phase 3B: room 1 with demonstrations

**Result: 82.5% success from the canonical start**, over 200 evaluation episodes, with a 95% interval of 76.6%
to 87.1%. The deterministic policy clears the room too. The unshaped no-demonstration baseline reported in
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
| Cloning then RL | **82.5%** (165 of 200) | the same clone plus 501,760 transitions |

Cloning alone clears the room rarely. RL alone never clears it. Cloning followed by RL clears it most of the
time, and ends up faster than the demonstrations it learned from.

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

| accepted steps | success (50 episodes) | median max x | deterministic |
|---|---|---|---|
| 100,352 | **0%** | **19.0** | timeout |
| 200,704 | **0%** | **19.0** | timeout |
| 301,056 | 16% | 149.0 | death |
| 401,408 | 72% | 276.0 | **success** |
| 501,760 | 84% | 261.5 | **success** |

Final checkpoint over 200 episodes: **165 successes, 82.5%, 95% interval 76.6% to 87.1%**, 32 deaths, 3
timeouts, median max x 267. Clears take a median of 336 frames, the fastest 210, against demonstrations of 285
to 519 frames.

### The collapse in the first 200,000 steps was self-inflicted

A median max x of 19.0 is the start position: for the first 200,000 transitions the policy stood still and
cleared nothing, having been handed a working policy. The cause is that **cloning never trains the value
head.** The loss is computed from the action logits only, so the critic stays at its initialisation while the
shared feature extractor moves underneath it. PPO therefore began with a critic worse than predicting the mean
(`explained_variance` -0.72 on the first update, `approx_kl` 0.0399 against a steady state of about 0.96 and
0.015) and spent its early updates moving the policy on arbitrary advantages.

It recovered, so the cost was time rather than the result. A critic warm-up before allowing policy updates
would likely reach the same place sooner, and would remove the ambiguity if a future run ended below where it
started.

## 5. What this does and does not establish

**It establishes** that the task is learnable by this network, this observation and this action space, and that
the barrier in the no-demonstration branch was neither capacity nor the environment. Given episodes whose
returns differ, PPO improves on what it was shown by a wide margin.

**It does not establish:**

- **Generalisation.** Every episode here begins at the canonical start. The roadmap's Phase 3 criterion is 99%
  on 200 held-out reachable entry states, from a generator frozen before training. That generator does not
  exist yet, and this report makes no claim about it.
- **Reproducibility across seeds.** The headline is one fine-tuning run from one clone.
- **Anything about the no-demonstration question.** That result stands as published. This policy was shown
  seven solutions.
- **Speedrun-quality play.** A median of 336 frames is a clear, not a good time.

## 6. Reproducing this

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
