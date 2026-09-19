# Phase 3B: room 1 with demonstrations

**Result: about 98% success from the canonical start when the method works, and it fails outright roughly one
run in four.** Four PPO fine-tuning runs from the same cloned policy produced 98.5%, 98.0%, 84% and **0%**. The
unshaped no-demonstration baseline reported in [phase3-unshaped-baseline.md](phase3-unshaped-baseline.md)
scored 0 clears in 420 evaluation episodes, which bounds its rate below about 0.9%, so even the failing
fine-tuning run is not worse than the baseline; it is simply no better.

The spread is the most important thing in this report and it is not understood. An earlier version of this
document reported "at least 82.5% on every seed" from three runs. A fourth run, made to put all seeds on one
commit, collapsed to standing still and never recovered.

This is the demonstration-assisted comparison the roadmap calls for when the no-demonstration budget is spent
without a clear. It does not replace that result, and it is not the same claim: this policy was shown solutions
and that one was not.

Run on 2026-09-18. Every number comes from a run whose commit and runtime were recorded and verified against
pinned hashes.

## 1. The comparison

| arm | canonical success rate | 95% interval | budget |
|---|---|---|---|
| RL only, `rew-v1` | **0%** (0 of 420) | 0% to 0.9% | 3 seeds x 2,000,896 transitions |
| RL only, `rew-v2` matched to the fine-tune | **0%** (0 of 250) | 0% to 1.5% | 1 seed x 501,760 transitions |
| Cloning only | **6%** (3 of 50) | 2.1% to 16.2% | 5 demonstrations trained on, 2 held out |
| Cloning then RL | **0% to 98.5%** | see per-run table | the same clone plus 4 x 501,760 transitions |

Cloning alone clears the room rarely. RL alone never clears it, under either reward. Cloning followed by RL
clears it nearly always on two seeds of three.

The second row matters, because otherwise the comparison changes two things at once. `runs/train/shaping2-diagnostic`
is exactly the fine-tuning configuration minus the cloned weights (seed 0, `rew-v2`, shaping scale 2.0, canonical
starts, 501,760 transitions) and it scored 0 of 50 at every one of its five evaluations, with a median furthest
x of 76. The reward change alone does nothing; the demonstrations are what changed the outcome.

Every run below starts from the same cloned policy and differs only in the PPO seed and, for the first row,
the commit:

| run | success | 95% interval | median clear | deterministic |
|---|---|---|---|---|
| seed 0, at `e3df073` | 165/200 = **82.5%** | 76.6% to 87.1% | 336 frames | success |
| seed 1, at `8e910eb` | 196/200 = **98.0%** | 95.0% to 99.2% | 172 frames | success |
| seed 2, at `8e910eb` | 197/200 = **98.5%** | 95.7% to 99.5% | 206 frames | success |
| **seed 0, at `35faa14`** | **0/250** | 0% to 1.5% | none | timeout |

The last row is the same seed and configuration as the first. They differ only in where the global random
stream sat when training began, because the seed fix in `8e910eb` re-seeds after the donor checkpoint is
loaded rather than inheriting the donor's position in the stream. Both are legitimately "seed 0"; they are
different points in one stream, and one of them produces a working policy and the other does not.

Three of four runs therefore learned to clear the room and one did not, from identical inputs. **A success rate
quoted from a single fine-tuning run is close to meaningless at this variance**, and the three-run version of
this report was reporting the lucky end of a distribution it had not measured.

The demonstrations run 285 to 519 frames. Seeds 1 and 2 clear faster than the shortest demonstration on a
typical episode (medians of 172 and 206 frames); **seed 0 does not**, at a median of 336 and a deterministic
clear of 324. Only seed 0's fastest episode beats 285.

These lengths are reported as description, not as evidence. The demonstrations came from a random search, so
being quicker than them is a low bar and says little about the quality of play.

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
| Search cost | 20 to 77 seconds of wall time per route, 7 searches, all successful |
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

Two baselines, because one of them is easy to beat and the other is not. **Always zero** is pressing nothing,
which scores well only because the inputs are sparse. **Persistence** is repeating the previous frame's action,
which the policy could read straight out of its own observation, since obs-v1 carries the last four applied
actions.

| | input accuracy | frame accuracy |
|---|---|---|
| cloned policy, trained on | 0.9988 | 0.9877 |
| cloned policy, **held out** | **0.9423** | **0.3973** |
| always zero, held out | 0.8968 | 0.0324 |
| **persistence, held out** | **0.9786** | **0.8554** |

**On routes it had not seen, the cloned policy is worse than repeating its own last action, on both measures.**
It did not generalise across routes at all. That is consistent with it clearing the room only 3 times in 50,
and it means the value of this stage was not a policy that understands the room: it was a policy whose action
distribution is shaped enough that PPO sometimes sees a success.

Playing it: **3 clears in 50 episodes** (95% interval 2.1% to 16.2%), and the deterministic policy timed out
standing near the start. This is what distribution shift looks like. A cloned policy has only ever seen states
on the demonstrated path, so its first mistake puts it somewhere no demonstration went, and errors compound.

## 4. Fine-tuning

PPO initialised from the cloned policy's weights (`rew-v2`, shaping scale 2.0, canonical starts, 501,760
transitions). The step counter, optimizer state and records all start fresh; only the weights are inherited.

Success rate during training, 50 episodes at each point:

| accepted steps | seed 0 (`e3df073`) | seed 1 | seed 2 | seed 0 (`35faa14`) |
|---|---|---|---|---|
| 100,352 | **0%** | 44% | **0%** | **0%** |
| 200,704 | **0%** | 92% | 12% | **0%** |
| 301,056 | 16% | 94% | 100% | **0%** |
| 401,408 | 72% | 100% | 96% | **0%** |
| 501,760 | 84% | 98% | 100% | **0%** |

Three runs pass through an early trough and climb out of it. The fourth never climbs out: it produced 6
successes in its first quarter and none in the remaining three, ending with 250 timeouts in 369 episodes and a
median furthest x of 19, which is the start position. It stood still for 400,000 transitions.

### The early collapse, which is sometimes permanent

Three of four runs scored 0% at 100,000 transitions, having been handed a policy that already cleared the room
6% of the time. Two of those recovered. **One never did**, and finished 500,000 transitions with a median
furthest x of 19, the start position. Seed 1 barely dipped at all, reaching 44% by the same point.

So the collapse is not a fixed cost that is paid and repaid. It is a fork, and one branch ends the run.

**The value-head explanation is now dead.** It was that cloning never trains the value head, so PPO begins
with a critic worse than predicting the mean and moves the policy on arbitrary advantages. First-update
`explained_variance` across the four runs:

| run | first-update explained variance | outcome |
|---|---|---|
| seed 0, `e3df073` | -0.72 | 84% |
| seed 1 | -0.43 | 98.0% |
| seed 2 | -0.42 | 98.5% |
| seed 0, `35faa14` | **+0.36** | **0%** |

**The run with by far the best starting critic is the one that failed completely**, and the run with the worst
recovered. The critic's state does not predict the outcome, and a critic warm-up is no longer expected to fix
this on its own, though it is still worth running as an ablation.

**The surviving explanation** is about the clone rather than the critic: the cloned policy's most likely action
is to press nothing, its deterministic episode timed out standing near the start, and every collapsed run
shows the same signature of timeouts with a median furthest x of 19. The run that never recovered ended with
250 timeouts in 369 episodes. That is consistent with PPO sharpening the policy toward its own argmax before
any useful signal arrives, and sometimes never escaping it. It is an explanation, not a finding: nothing here
tests it.

A critic warm-up, training the value head on frozen features before allowing policy updates, is still the
cheapest ablation and has not been run. But on this evidence the thing to explain is not a delay: it is why
the same configuration sometimes escapes the trough and sometimes does not, which four runs cannot answer.

**This section has now been wrong twice.** It first stated the value-head explanation as the cause from one
run, then softened it to "most likely" when a second run contradicted it, and a fourth run has now inverted the
correlation entirely.

## 5. What this does and does not establish

**It establishes** that the task is learnable by this network, this observation and this action space, and that
the barrier in the no-demonstration branch was neither capacity nor the environment. Given episodes whose
returns differ, PPO improves on what it was shown by a wide margin, on every seed tried, and arrives at
solutions faster than the ones it was given.

**It does not establish:**

- **Generalisation.** Every episode here begins at the same canonical start, and the game is deterministic, so
  a stochastic success rate measures how robust the policy's own sampling is around one route rather than
  whether it can play the room. **There is no test set anywhere in this experiment.** The roadmap's Phase 3
  criterion is 99% on 200 held-out reachable entry states from a generator frozen before training; that
  generator does not exist yet and this report makes no claim about it.
  (A note for when it is built: 200 of 200 successes has a Wilson lower bound of about 98.1%, so 200 episodes
  can never demonstrate a rate above 99% with confidence. The criterion needs restating as an interval before
  any run is judged against it.)
- **Reproducibility.** Fine-tuning is *not* reproducible in outcome: four runs from one cloned policy gave 0%
  to 98.5%. Each individual run is exactly reproducible given its commit and seed, which
  `scripts/reproduce_check.py` verifies, but the method's outcome is not.
- **That the spread is understood.** Nothing here explains why the same configuration sometimes escapes the
  early trough and sometimes does not. Four runs cannot characterise it; this needs eight or more, and the
  headline should be a distribution rather than a number.
- **Anything about the clone's own variance.** All four runs share one cloned policy. A different demonstration
  set or cloning seed is untested.
- **Anything about the no-demonstration question.** That result stands as published. This policy was shown
  seven solutions.
- **Speedrun-quality play.** Medians of 172 to 336 frames are clears, not good times.
- **That the demonstrations taught the room.** The clone is below a persistence baseline on held-out routes
  (section 3), so what PPO started from was closer to a usefully-shaped action distribution than to a policy
  that knows the route. How much of the seven is needed is untested: one route might be enough.

## 6. A note on the seeds and the evaluation stream

The first attempt at seeds 1 and 2 produced byte-identical episode logs. Stable-Baselines3's `load()` runs
`_setup_model()`, which calls `set_random_seed` with the donor checkpoint's saved seed and re-seeds torch,
numpy and python globally, after the run's own model has been seeded. Every run initialised from a checkpoint
therefore sampled from the donor's stream whatever seed it was given. The runs above are from after that was
fixed. Seed 0's original run is unaffected, because it was configured with seed 0 and the donor was saved with
seed 0, so it sampled from the stream it claimed to. That is an argument rather than a measurement; the game is
deterministic, so it is checkable by replaying one rollout and comparing episodes.

The same mechanism applied to `scripts/evaluate_checkpoint.py`, which had no seed of its own: the three
200-episode evaluations above each drew the stream belonging to their checkpoint's saved seed, and re-running
one would have returned the same 200 episodes rather than a fresh sample. The script now takes `--seed`. The
numbers stand, but they are one sample each, not a repeatable draw.

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
