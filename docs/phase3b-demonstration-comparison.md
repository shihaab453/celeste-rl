# Phase 3B: room 1 with demonstrations

**Result: varied-start PPO fine-tuning generalises better across room 1 than canonical-start fine-tuning.** On
a replacement frozen set of 200 unique states from 11 independent search routes, the predeclared route-macro
success rate was 74.8% for varied-start training and 52.2% for canonical-start training. The matched difference
was **+22.6 percentage points**, with a 95% crossed seed-route bootstrap interval from +6.7 to +43.5 points and
an exact two-sided paired sign-flip p-value of 0.03125. The varied-start arm was better in all six matched
training seeds and on all 11 route-level descriptive comparisons.

This result replaces the withdrawn 79.9% against 65.1% preliminary comparison, which used a post-selected,
unmatched checkpoint subset and an invalid frozen artifact. Section 5b gives the full corrected analysis and
keeps the old numbers only as an audit trail. The unshaped no-demonstration baseline still never cleared the
room at all.

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
| Cloning then RL, canonical starts | **0% to 98.5%** | see per-run table | the same clone plus 501,760 transitions per run |
| Cloning then RL, varied starts | **74.8% route-macro** | cluster-aware interval in section 5b | 6 seeds x 200 frozen states |

The first four rows are canonical-start numbers and do not measure generalisation over the room. The final row
is the corrected held-out estimate. It accounts for matched training seeds and route clustering rather than
treating adjacent prefixes as independent episodes.

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

This section covers the first four runs, which is all there was when it was written. Twelve further runs are in
section 5b, and they revise two of its conclusions: the collapse is rarer than four runs suggested, and the
canonical success rates below are not the number to judge a policy by.

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
the barrier in the no-demonstration branch was neither capacity nor the environment. PPO can improve on what
it was shown by a wide margin and can arrive at solutions faster than the ones it was given. It does not do so
on every run.

**It does not establish:**

- **Uniform reliability.** The matched seed differences all favour varied starts, but range from +2.2 to
  +71.4 percentage points. Training-seed variation remains large.
- **Policy-sampling precision within each state.** Each checkpoint received one stochastic episode per frozen
  state. The design estimates performance across state and route clusters, not repeated-action uncertainty at
  one state.
- **That eleven routes cover every reachable state.** They are eleven unique full-route hashes from the same
  search procedure. The cluster-aware interval avoids pretending that 200 adjacent prefixes are independent,
  but eleven route clusters still limit the population claim.
- **Anything about the clone's own variance.** All four runs share one cloned policy. A different demonstration
  set or cloning seed is untested.
- **Anything about the no-demonstration question.** That result stands as published. This policy was shown
  seven solutions.
- **Speedrun-quality play.** Medians of 172 to 336 frames are clears, not good times.
- **That the demonstrations taught the room.** The clone is below a persistence baseline on held-out routes
  (section 3), so what PPO started from was closer to a usefully-shaped action distribution than to a policy
  that knows the route. How much of the seven is needed is untested: one route might be enough.

## 5b. The matched A/B campaign: corrected held-out result and the low-signal regime

Twelve fine-tuning runs from the same cloned policy were declared in `config/campaign-finetune-variance.json`
before training. Arm A fine-tuned from the canonical start. Arm B added varied starting states drawn with
success-weighted sampling from the reached-state archive. Seeds 3 through 8 were matched across arms, and every
final checkpoint had 501,760 accepted transitions.

The corrected evaluation and analysis were fixed before any checkpoint touched the replacement set in
`config/campaign-heldout-ab-200.json` (plan SHA-256
`5595361bdbf3fd2f8a4a7876773ffa7550b65633d139c38f3d5d5ed5bcd169b5`). The frozen manifest has 200 unique
states from 11 unique full-route hashes. Each checkpoint played one stochastic episode from every state with
evaluation seed 20260920, for 2,400 validated episode rows.

### Predeclared cluster-aware analysis

The unit of training replication is the matched seed, and the held-out cluster is the complete search route.
For each seed and route, the analysis calculates the success proportion over that route's states. It gives all
11 routes equal weight inside a seed, calculates paired B minus A, then gives all six seeds equal weight.

The confirmatory test is the exact two-sided paired sign-flip test over the six seed-level differences, all 64
sign assignments. The interval is a 100,000-draw percentile crossed-cluster bootstrap that independently
resamples matched seed pairs and complete routes while retaining the A/B pairing and every state inside a
selected route cell. Evidence favouring B required a positive estimate, an interval excluding zero, and
p <= 0.05. Any missing run, row, route, or hash would have refused the analysis instead of changing the sample.

<!-- BEGIN GENERATED PRIMARY TABLE -->
| primary result | A, canonical starts | B, varied starts | B minus A |
|---|---:|---:|---:|
| Route-macro success | 52.2% | **74.8%** | **+22.6 points** |
| Crossed seed-route 95% interval | | | **+6.7 points to +43.5 points** |
| Exact paired sign-flip test | | | **p = 0.03125** |
<!-- END GENERATED PRIMARY TABLE -->

The result meets every part of the predeclared decision rule. The state-weighted totals are secondary and
descriptive: A cleared 619 of 1,200 states (51.6%) and B cleared 894 of 1,200 (74.5%).

### Per-seed results

Route-macro rates give each of the eleven routes equal weight. The counts show the secondary state-weighted
result over 200 states per checkpoint.

<!-- BEGIN GENERATED SEED TABLE -->
| seed | A route-macro | B route-macro | paired difference | A successes | B successes |
|---:|---:|---:|---:|---:|---:|
| 3 | 59.4% | 76.4% | +17.1 points | 120 | 152 |
| 4 | 5.5% | 76.9% | +71.4 points | 10 | 153 |
| 5 | 72.5% | 77.4% | +4.8 points | 142 | 155 |
| 6 | 63.8% | 65.9% | +2.2 points | 124 | 132 |
| 7 | 64.3% | 78.8% | +14.5 points | 128 | 157 |
| 8 | 47.6% | 73.3% | +25.7 points | 95 | 145 |
<!-- END GENERATED SEED TABLE -->

Every matched difference is positive. Seed 4 contributes the largest gain, but the conclusion is not created
by that one pair: the exact sign-flip test reaches its minimum two-sided p-value for six pairs because all six
directions agree.

### Per-route results

These are descriptive rates pooled over the six checkpoints in each arm. Each row remains one cluster in the
primary analysis regardless of how many prefixes it contains.

<!-- BEGIN GENERATED ROUTE TABLE -->
| route | states per checkpoint | A | B | B minus A |
|---|---:|---:|---:|---:|
| `20260920-001246` | 18 | 57.4% | 76.9% | +19.4 points |
| `20260920-001338` | 18 | 60.2% | 77.8% | +17.6 points |
| `20260920-001447` | 17 | 36.3% | 59.8% | +23.5 points |
| `20260920-001710` | 18 | 44.4% | 77.8% | +33.3 points |
| `20260920-001734` | 16 | 61.5% | 71.9% | +10.4 points |
| `20260920-001754` | 18 | 40.7% | 63.0% | +22.2 points |
| `20260920-001819` | 21 | 64.3% | 85.7% | +21.4 points |
| `20260920-001842` | 18 | 53.7% | 78.7% | +25.0 points |
| `20260920-001917` | 21 | 50.0% | 72.2% | +22.2 points |
| `20260920-001950` | 26 | 44.2% | 73.7% | +29.5 points |
| `20260920-002044` | 9 | 61.1% | 85.2% | +24.1 points |
<!-- END GENERATED ROUTE TABLE -->

All eleven route-level differences are positive. The full machine-readable result, including checkpoint and
episode-file hashes, is in `docs/results/phase3b-heldout-ab-200.json`.

### What the canonical start was hiding

| checkpoint | canonical start | preliminary 149-entry artifact |
|---|---|---|
| seed 1 | 98.0% | 59.7% |
| seed 2 | 98.5% | 84.6% |
| seed 0 (`e3df073`) | 82.5% | 68.5% |

Every policy scores lower on this diagnostic artifact than on the start it was tuned against, by 14 to 38
points, and **not by a consistent amount**. Seeds 1 and 2 look identical on the canonical start and are 25
points apart on the artifact. Seed 1 clears 100% from recorded entries near the exit and 25 to 31% from the
early room. One fixed start could not have shown that difference, even though the artifact cannot support a
population estimate.

### Withdrawn preliminary selected-checkpoint comparison

| selected checkpoints | diagnostic score |
|---|---|
| A seeds 3, 5 and 7, canonical-start training | 291/447 = **65.1%** |
| B seeds 4, 5 and 7, varied starts with success sampling | 357/447 = **79.9%** |

Each checkpoint was evaluated on 149 entries, but 24 entries duplicate another route's prefixes exactly, so
there are 125 unique states from 5 unique routes. Prefixes along a route are also correlated. The ordinary
Wilson intervals previously printed here treated all 447 rows as independent and are withdrawn.

The selected B checkpoints scored 78.5%, 82.6% and 78.5%; the selected A checkpoints scored 62.4%, 62.4% and
70.5%. But the subset was chosen after the campaign, is not seed-matched, and omits A seeds 4, 6 and 8 and B
seeds 3, 6 and 8. This was promising diagnostic evidence, not an unbiased comparison of the two arms. The
corrected matched analysis above is now the result used for the generalisation claim.

Across the canonical-start evaluations of all six training runs, B includes final rates of 62% and 68%, while
four A runs reach 92% to 94%. That remains consistent with canonical training over-specialising to one start,
but it does not quantify what is gained over the room.

### The low-raw-advantage signature

The trap signature was declared in advance as a rollout with `advantage_std` below 0.02 and more than 80%
timeouts.

| arm | runs showing the signature | lowest `advantage_std` |
|---|---|---|
| A | **5 of 6** | 0.0056 to 0.0144 |
| B | **0 of 6** | 0.031 to 0.073 |

The 5 of 6 against 0 of 6 count is a real descriptive difference. The arms use the same six seeds, so the
natural exact paired calculation is two-sided p = 0.0625, or one-sided p = 0.03125 if that direction had been
predeclared. The previously reported unpaired Fisher value of about 0.015 ignores the matching and is not used
as the significance claim here.

`rew-v2` makes every completed failure have the same episode return, which can leave very little raw advantage
variation in timeout-heavy rollouts. It does **not** follow that PPO has no policy gradient. The recorded
`advantage_std` is measured before Stable-Baselines3 normalises advantages to unit variance within minibatches,
and the signature rows have nonzero policy losses and KL movement. Equal episode totals also do not make every
per-state GAE advantage equal. The evidence therefore supports a low-raw-signal regime, not an entropy-only
random walk or literally zero gradient.

Varied starts are a plausible reason the signature disappears, because episodes from different states create
more varied outcomes. This campaign associates the varied-start intervention with the missing signature; it
does not isolate that mechanism from success-weighted sampling or directly measure the quality of the
normalised policy gradient.

### What did not replicate

**The collapse is rarer than this report previously claimed.** No run in either arm collapsed under the
declared definition. The descriptive pooled count across the six new A runs and four earlier runs is **1
collapse in 10**, not the "one run in four" this document said after four runs. One of the earlier runs used
different reseeding behaviour, so a binomial interval over all ten is not presented. One A run came close,
finishing at 2% after 137,000 steps in the low-signal regime, and is classed partial.

The signature was absent from the six B runs, but the campaign could not show that this prevents collapses,
because collapses turned out to be too rare for twelve runs to compare. The plan said in advance that six
against six could not reach significance on that comparison, and it did not.

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
.venv-rl/Scripts/python.exe scripts/analyze_heldout_ab.py --plan config/campaign-heldout-ab-200.json \
    --campaign runs/campaign/20260920-011905-heldout-ab-200-v1/summary.json
```

Every live script refuses to start from a dirty working tree or a runtime that differs from
`config/pinned_runtime.json`, so a result can always name the code and the game build that produced it.
Canonical-start rates are reported with Wilson score intervals. The matched held-out comparison uses the
predeclared route-macro estimand, exact paired test and crossed-cluster bootstrap described in section 5b.
