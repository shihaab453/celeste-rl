# Phase 3 baseline: PPO on room 1 without demonstrations

**Result: no clears.** Three seeds, 2,000,896 accepted transitions each, 50,369 training episodes and 420
evaluation episodes, and not one of them reached room 2.

This is the documented no-demonstration attempt the project committed to before allowing demonstrations, and it
is reported here in full, including the parts that did not work. The headline number is not the interesting
part. The interesting part is what the diagnostics say about *why*, and what they rule out.

Run on 2026-09-17. Every number below comes from a run whose commit and runtime were recorded and verified
against pinned hashes.

## 1. What was run

| | |
|---|---|
| Task | Chapter 1 room 1, from the canonical start at (19, 144) to the room 2 transition |
| Budget | 2,000,896 accepted transitions per seed (2,000,000 requested, rounded up to a rollout boundary) |
| Seeds | 0, 1, 2 |
| Algorithm | PPO (Stable-Baselines3 2.9.0), `MultiInputPolicy` with a small CNN over the semantic grid |
| Observation | `obs-v1`, schema fingerprint `b710a27fa4f1d96f` |
| Actions | `act-v1`, 24 on/off inputs, 21 enabled (pause, quick restart and journal disabled) |
| Reward | `rew-v1`: +1 success, -1 any failure, -1/60,000 per frame, no shaping |
| Hyperparameters | `n_steps` 2048, batch 512, 4 epochs, learning rate 3e-4, gamma 1.0, clip 0.2, entropy coefficient 0.01 |
| Deadline | 1,800 frames (30 seconds of normal play) |
| Hardware | One laptop, CPU inference, the game running minimised alongside |

The three runs are matched in every parameter that affects learning. They differ in one thing that does not:
seed 0 saved a checkpoint every 50,000 steps and seeds 1 and 2 every 250,000, so seed 0 has 40 saved
checkpoints and the others have 8. That changes what can be measured afterwards, not what was learned.

Each seed ran as a single uninterrupted session: **zero bridge faults, zero discarded rollouts and zero game
relaunches** across all three. Collection ran at 421 to 487 environment steps per second; each seed took
118 to 132 minutes of wall time, about 6.3 hours in total.

## 2. The result

| Seed | Accepted steps | Episodes | Endings | Evaluations | Wall time |
|---|---|---|---|---|---|
| 0 | 2,000,896 | 17,754 | all death | 7 x 0/20 | 130 min |
| 1 | 2,000,896 | 18,175 | all death | 7 x 0/20 | 118 min |
| 2 | 2,000,896 | 14,440 | all death | 7 x 0/20 | 132 min |

**Zero recorded clears in 50,369 training episodes and 21 evaluations.** Every training episode ended in
death; none reached the 30-second deadline.

## 3. Did the policy get further into the room?

Success rate cannot tell a policy that almost clears the room from one that never moves, so saved checkpoints
were replayed and their positions recorded. An untrained network of the same architecture and seed is included
as the control, which is what makes the answer meaningful.

Room 1 runs rightwards from the start ledge at x 19, over a spike pit of open air from x 40 to x 80 with
upward spikes at the bottom, up onto a platform, and out through the top of the room near x 261. Larger x is
progress.

Median furthest-right position, over 20 sampled episodes per checkpoint:

| Seed | Untrained | 25% | 50% | 75% | End of training | Best checkpoint |
|---|---|---|---|---|---|---|
| 0 | 76 | 76 | 76 | 76 | 75 | 76 |
| 1 | 76 | 76 | 75 | 75 | 76 | 76 |
| 2 | 76 | 76 | 76 | 75 | 76 | 76 |

Each column is the measured checkpoint nearest that fraction of training. For scale: on seed 1 the final
weights were sampled twice, as `step_000002000896.zip` and as `latest.zip`, and gave medians of 75.5 and 76.
That difference is sampling noise from 20 episodes, and it is larger than anything in the table.

**The median never moves.** After two million transitions the typical episode gets exactly as far as an
untrained network does, on all three seeds.

The frontier does not move either. Across 32 measurements (three seeds, their checkpoints and their untrained
controls), the furthest any sampled episode reached was exactly x 140 in 25 of them, and in the other 7 it was
lower. **Not one episode in any measurement passed x 144**, of the 261 needed. The x 140 ceiling appears at the
untrained controls too, which suggests it is a feature of the room's geometry under near-random input rather
than something that was learned.

Recorded ending positions cluster around (47, 168) and (60, 168): the spike floor at the bottom of the first
pit. The typical episode walks right, falls into the first pit and dies on the spikes.

## 4. Did the policy change at all?

Measured offline at a fixed set of 285 room 1 observations, for every saved checkpoint:

| Seed | Entropy at the first checkpoint | At the last | Share of maximum | Value head |
|---|---|---|---|---|
| 0 | 16.48 | 15.63 | 0.991 to 0.939 | -0.87 to -1.00 |
| 1 | 16.20 | 15.43 | 0.974 to 0.928 | -1.00 throughout |
| 2 | 16.00 | 15.51 | 0.962 to 0.932 | -1.00 throughout |

The maximum possible entropy for 24 independent on/off inputs is 24 ln 2 = 16.64. **The policy stayed within
7% of uniform coin flips for the whole run on every seed.** Averaged over the enabled inputs, the probability
of pressing each one stayed within about 0.1 of 0.5.

Individual inputs did drift, but not in a shared direction. The largest moves over training were jump 0.59 to
0.24 on seed 0, dash-only-up 0.54 to 0.29 on seed 1, and grab 0.70 to 0.34 on seed 2. These are seed-specific
wanderings of a nearly uniform policy, not a common strategy.

The value head is the clearest signal: it converged to almost exactly -1.00, the return of an immediate
failure. The critic learned the one thing the data could teach it, which is that every episode ends badly.

## 5. Episode lengths

Median episode length by quarter of training:

| Seed | First quarter | Second | Third | Fourth |
|---|---|---|---|---|
| 0 | 107 | 87 | 69 | 67 |
| 1 | 102 | 97 | 54 | 54 |
| 2 | 102 | 102 | 96 | 98 |

Seeds 0 and 1 roughly halved their episode length. **Seed 2 did not**, staying flat across all four quarters.

A tempting reading is that the agent learned to die sooner, because `rew-v1` pays slightly more for failing
early: a failure at frame t costs `-1 - t/60,000`, so an immediate death scores about -1.00002 and a timeout
scores -1.03. That incentive is real and was a known, documented trade-off of the reward design. But this data
does not establish it as the cause. Two of three seeds shortened, one did not, and nothing here separates
"learned to die sooner" from ordinary drift in a policy that is still almost uniform. **It is a hypothesis, not
a finding.** It is also cheap to remove, which is what the next experiment does.

## 6. What this does and does not show

It shows that PPO with this reward, this observation and this budget does not clear room 1, and that on this
evidence it was not on the way to doing so: the policy neither got further into the room nor left the random
regime.

It does not show:

- **Anything about other starts.** Every episode in this campaign began at the same canonical start. Nothing
  here speaks to generalisation over entry states.
- **A success rate below 5%.** Each evaluation sampled 20 episodes, which cannot distinguish a rate of 0 from
  one of a few percent. Evaluations now use 50 episodes.
- **A trend in pit crossings.** Across checkpoints, 0 to 7 of 20 sampled episodes crossed the pit, including 2
  to 3 of 20 for the untrained control. At that sample size this is noise, and no trend should be read into it.
  The median furthest-right position is the number to trust here, because it never moves at all.
- **That the shortening is causal.** See section 5.
- **A memory leak.** The game's private bytes ranged over roughly 200 MiB within each run, with a fitted drift
  between -6 and +9 MiB per million steps. There is no consistent direction across seeds, so this is normal
  variation, not literally flat and not a leak.

The checkpoint replays carry their own limits: 20 episodes per checkpoint, only summary statistics saved rather
than per-episode values, each checkpoint sampled once from the random state it happened to carry, and unequal
checkpoint counts between seed 0 and the others. The conclusion drawn from them is deliberately limited to the
one thing robust to all of that, which is that the median is identical to the untrained control everywhere.

## 7. What was ruled out, and what changes next

Before changing the reward, the cheaper explanation was tested: perhaps the network or the observation simply
cannot represent this task. The recorded 285-frame clear of room 1 was replayed through the environment and the
training policy was fitted to those observation and action pairs. It reached 99.87% per-input accuracy and
99.30% exact-frame accuracy, against an always-zero baseline of 89.02% and 0.00%.

**The network and the observation can represent a solution.** The failure is not capacity or features, so it is
exploration and reward.

Two candidate causes remain, and the next experiment addresses both at once because fixing either alone would
waste a run:

1. **The reward paid for failing early.** `rew-v2` charges every failure for the deadline it did not use, so a
   failure costs exactly -1.03 whenever it happens. The time cost still applies from the first frame and there
   is no survival bonus.
2. **The reward said nothing about where to go.** `rew-v2` adds potential-based shaping over a spike-weighted
   distance to the exit, validated offline against the recorded clear and a recorded death before any training.
   With a terminal potential of zero it sums to a per-start constant over any episode, so it cannot change
   which ending the agent prefers.

The measurements this baseline lacked now exist as well: every episode records how far it got, and every
checkpoint can be read for entropy and per-input behaviour. If the next run fails, it will fail legibly.

## 8. Reproducing this

```
.venv-rl/Scripts/python.exe scripts/train_room1.py --seed 0 --reward-version rew-v1
.venv-rl/Scripts/python.exe scripts/checkpoint_progress.py --run-dir runs/train/<run>   # needs the game
.venv-rl/Scripts/python.exe scripts/checkpoint_policy.py  --run-dir runs/train/<run>   # offline
```

Training refuses to start from a dirty working tree or a runtime that differs from
`config/pinned_runtime.json`, so a result can always name the code and the game build that produced it. Run
outputs are kept out of version control; the records behind this report are the `manifest.json`, `progress.csv`,
`episodes.jsonl` and `evaluations.jsonl` of each run directory.

## Addendum, 2026-09-18: three things this report leaves out

Nothing above is false, and the numbers are unchanged. But a day of follow-up experiments and an external
review found three places where the report is overstated by omission, and it is better to say so here than to
quietly rewrite the original.

**1. "The failure is not capacity or features, so it is exploration and reward" leaves out the learner.**
Section 7 draws that conclusion from the capacity check, and it excludes a third possibility that turned out to
matter. The action space is 24 independent on or off inputs, and the action head's bias starts at zero, so
every input starts at probability 0.5 and a fresh policy holds about **twelve buttons at once, every frame**. A
recorded clear holds 2.31. The near-uniform policy the report describes is partly a property of how the policy
is parameterised and initialised, not only of the reward it was given. The entropy bonus then keeps it there,
because maximum entropy over 21 independent inputs is exactly that twelve-button behaviour.

**2. The entropy figures in section 4 are measured on states the policy does not visit.** They come from 285
observations along a recorded clear, and about two thirds of those are past x 140, which the policy in this
campaign never reached. The figures are correct as stated, and they are not measured where the agent actually
spends its time.

**3. The description of the shaped reward in section 7 needs its other half.** The report says potential-based
shaping "cannot change which ending the agent prefers", which is true and is the property that makes it safe.
The consequence not stated is that the unspent-deadline charge, by making every failure cost the same whenever
it happens, also removes any reason to prefer one failure to another. Measured afterwards: across 8,146
episodes of shaped training, every failing episode from the canonical start returned the same number to six
decimal places, and the critic learned the shaped value function to within 0.03, so the shaping term cancels in
the temporal-difference error and contributes nothing to the policy gradient. With no successes to learn from
either, there was nothing in those returns for the policy to improve on. That is a better account of why the
shaped experiments were inert than anything in the original text.

**What has happened since, in brief.** Starting episodes from states the agent had already reached produced
the first clears of room 1, though never from the canonical start. Supervised cloning of five independently
searched routes then produced a policy that clears the canonical start about 6% of the time, and PPO
fine-tuning from that policy reached **82.5% to 98.5% across three seeds**. That work is reported separately in
[phase3b-demonstration-comparison.md](phase3b-demonstration-comparison.md).

None of it changes the finding above, which stands: PPO with this reward, this observation and this budget did
not clear room 1 without demonstrations. What the later work does establish is that the task was learnable by
the same network, observation and action space all along, so the barrier here was neither capacity nor the
environment.
