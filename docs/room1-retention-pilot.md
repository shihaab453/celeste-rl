# Room 1 retention pilot

A descriptive pilot with four runs and one training recipe. It is not a confirmatory result.

## Question

When a policy that already clears Room 1 is taught Room 2, how much of its Room 1 skill remains?

## What was done

The plan was declared before any pilot data (`config/retention-pilot.json`).

- **Starting policies:** the four lowest-numbered seeds (3 to 6) of the Phase 3B varied-start Room 1 policies. The
  choice was fixed in advance.
- **Teaching Room 2, step 1 (imitation):** each policy was cloned on the Room 2 demonstrations, starting from its
  own weights: fitted on 16 of the 21 routes (5 held back to check the fit, as for the Room 2 clones), 300 epochs on
  Room 2 frames only, with every weight free to change.
- **Teaching Room 2, step 2 (fine-tuning):** PPO on Room 2 from the canonical start for 500,000 steps, with the
  training-only stall ending (see [Room 2 training stability](room2-training-stability.md)).
- **Room 1 measure:** the frozen 200-state Room 1 held-out set from Phase 3B, played once from every state after
  cloning and at every 100,000-step checkpoint. That set already carries the published Phase 3B result, so only
  aggregate success is reported here, never per-route or per-state results.
- **Floor:** some Room 1 test states are close to the exit, where generic movement can succeed. So the same test was
  run on policies that never saw Room 1: four Room 2 clones, and the four Room 2 policies fine-tuned from them. The
  floor is what "no Room 1 skill at all" scores.

## Result

Room 1 held-out success (route-macro: each route counts equally):

| Starting policy | Before | After imitation | 100k | 200k | 300k | 400k | 500k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 3 | 76% | 21% | 23% | 28% | 32% | 31% | 29% |
| Seed 4 | 77% | 15% | 21% | 19% | 14% | 12% | 9% |
| Seed 5 | 77% | 17% | 24% | 32% | 28% | 28% | 30% |
| Seed 6 | 66% | 17% | 25% | 31% | 28% | 33% | 31% |
| Floor: never saw Room 1 | | 10% to 17% | | | | | 12% to 30% |

The "Before" values are the published Phase 3B results for these policies; they reproduced exactly when replayed
before this pilot.

**Teaching Room 2 by imitation erased almost all of the Room 1 skill.** Room 1 success fell from between 66% and 77%
to between 15% and 21% after the imitation step alone, close to policies that never saw Room 1 (10% to 17%). Measured against the floor,
2% to 12% of the skill remained. Through fine-tuning on Room 2 it stayed between 9% and 33%. Policies that never saw
Room 1 score 12% to 30% after the same fine-tuning, probably because Room 2 training teaches general movement that
helps a little anywhere. So the later rise is not Room 1 skill coming back, and this pilot cannot tell "nothing retained"
apart from "about 15% retained".

The policies still learned Room 2 about as well as clones that started from fresh weights: their final checkpoints
cleared 46 to 50 of 50 episodes from the Room 2 start, against 39 to 48 for the fresh clones (training side).

## Limits

- This describes one recipe: imitation on one room's frames, with every weight free to change. It does not show
  that forgetting is unavoidable. Mixing both rooms' demonstrations into the imitation step is the obvious next
  test, with Room 1 measured again after fine-tuning, since seed 4 kept losing Room 1 skill during Room 2
  fine-tuning (21% at 100k to 9% at 500k).
- Four runs, descriptive only.
- Each Room 1 number is one sampled episode per state over 200 states, so it carries a few points of sampling noise
  (about 3 points at 20% success). Single-checkpoint wiggles mean little.
- Seed 4 ended at 9%, below every floor policy.
- The Room 1 value estimate was copied over with the weights, so fine-tuning started from a different value
  estimate than for fresh clones.
- The Room 1 held-out set was reused from Phase 3B (aggregates only). A fresh Room 1 set will be generated before any
  confirmatory retention experiment.
- Room 2 was measured on the training side only. The fresh Room 2 held-out set was not used.

The per-checkpoint numbers, the floor for every run and the input hashes are in
[`docs/results/retention-pilot.json`](results/retention-pilot.json), written by
`scripts/analyze_retention_pilot.py`.

## Second pilot: mixing the Room 1 demonstrations back in

Also descriptive, four runs, one recipe, declared before any data (`config/mixed-imitation-pilot.json`).

The same four starting policies, seeds and settings, with one change: the imitation step used the Room 1 and Room 2
demonstrations together, each room counting equally in the loss. The plan also added a reference: clones fitted on
the Room 1 demonstrations alone, from fresh weights, scored on the same Room 1 test. It fixed in advance how each
outcome would be read.

| Starting policy | Before | After mixed imitation | 100k | 200k | 300k | 400k | 500k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 3 | 76% | 17% | 24% | 11% | 12% | 21% | 11% |
| Seed 4 | 77% | 14% | 22% | 30% | 25% | 22% | 13% |
| Seed 5 | 77% | 21% | 25% | 20% | 16% | 18% | 14% |
| Seed 6 | 66% | 22% | 22% | 17% | 13% | 12% | 12% |
| Floor: never saw Room 1 | | 10% to 17% | | | | | 12% to 30% |
| Room 1 demonstrations only | | 10% to 15% | | | | | |

**Mixing the demonstrations back in did not keep the Room 1 skill.** After mixed imitation, Room 1 success was
between 14% and 22%, and between 11% and 14% after fine-tuning on Room 2. The reference suggests why: clones fitted
on the Room 1 demonstrations alone scored only 10% to 15% on the Room 1 test, no better than policies that never saw
Room 1. On their own, as a policy, the demonstrations carry almost nothing that the Room 1 test measures; their value
in Phase 3B was as a starting point for reinforcement learning. The starting policies' 66% to 77% came from that
reinforcement learning, and re-imitating the demonstrations did not bring it back. A hint, from four runs: all four
mixed policies ended fine-tuning below the policies that never saw Room 1 (11% to 14%, against a floor of 12% to 30%
with a mean of 21%), as if re-imitating the demonstrations pulled toward weaker Room 1 play. By the rule fixed in
advance, the next recipe should instead use the starting policy's own play as
the imitation target.

Fitting the demonstrations was not the problem. The mixed clones matched the held-back Room 2 routes as well as
Room 2-only clones did, and matched the held-back Room 1 routes better than the Room 1-only reference clones did (51%
to 69% of whole frames against 33% to 40%), yet they still scored near the floor on the Room 1 test. They also still
learned Room 2: their final checkpoints cleared 48 to 50 of 50 episodes from the Room 2 start (training side).

Limits: as for the first pilot. In addition, the floor was reused from the first pilot (the same checkpoints and
command, with the evaluation code unchanged), and balanced mixing also changed the Room 2 side of the imitation step
(half the loss weight, about 26% more frames per epoch), so Room 2 differences between the pilots cannot be put down
to Room 1 being present.

The numbers, the reference clones and the input hashes are in
[`docs/results/mixed-imitation-pilot.json`](results/mixed-imitation-pilot.json), written by
`scripts/analyze_mixed_imitation_pilot.py`.

## Third pilot: can a copy of the policy play like it?

Also descriptive, four runs, one recipe, declared before any data (`config/coverage-ceiling-pilot.json`).

The second pilot suggested that the demonstrations were the wrong thing to imitate. Before mixing anything else in,
this pilot asked a narrower question: if a network is trained only to copy what a starting policy itself does, can
it play Room 1 like that policy? If even that failed, imitation could not rebuild the skill from scratch.

- **Copying target:** the starting policy's own probability for each button, on each frame, rather than a single
  "right answer" per frame.
- **Copies:** fresh networks, sharing no weights with the policy they copy, trained with the same settings as the
  earlier imitation steps.
- **Two sets of frames to copy on:**
  - **Demonstration frames:** the Room 1 demonstration routes (5 of the 7 routes fitted, about 2,000 frames).
  - **The policy's own play:** 25 episodes of each starting policy playing Room 1 from its normal start (19 fitted,
    4,300 to 8,600 frames). No test state was used as a start.
- **Measure:** the same Room 1 test, floor and reference as before; nothing was fine-tuned afterwards.

| Starting policy | Before | Copy on demonstration frames | Copy on its own play |
|---|---:|---:|---:|
| Seed 3 | 76% | 64% | 69% |
| Seed 4 | 77% | 60% | 74% |
| Seed 5 | 77% | 70% | 83% |
| Seed 6 | 66% | 50% | 56% |
| Floor: never saw Room 1 | | 10% to 17% | 10% to 17% |
| Room 1 demonstrations only | | 10% to 15% | |

**A copy of the policy plays Room 1 almost as well as the policy.** Measured against the floor, copies trained on
the policy's own play reached 81% to 109% of the policy's skill (median 92%), and copies trained on the demonstration
frames reached 69% to 88% (median 76%). Seed 5's copy scoring above its policy (83% against 77%) is within sampling
noise; read it as about equal.

The clearest comparison is on the demonstration frames. There, copying the policy's own probabilities gave 50% to
70%, while copying the demonstrated buttons on exactly the same frames, split and settings gave 10% to 15%. So what
the copy is trained to reproduce matters a great deal: copying the policy you want to keep works, and copying the
demonstrations does not. This pilot cannot say *why*. Two things changed together: whose choices are copied (on the
demonstration frames, all of the policy's most likely buttons match the demonstrated ones on only 20% to 30% of
frames), and probabilities instead of one right answer per frame. It also does not show that the choice of frames is
unimportant. Copies trained on the policy's own play did better in all four pairs, by 5 to 14 points, but they also
had two to four times as many frames.

**This is not yet a retention result.** Nothing learned Room 2 here; the copies only show that the policy's own
probabilities can carry its Room 1 skill. Whether they keep it while Room 2 is learned is the next pilot's question.

Sampling noise and the reused test set, as for the first pilot, and in addition:

- **This shows copying along known paths, not far from them.** Most Room 1 test starts (59% to 76%, depending on the
  copy) lie within 4 pixels of a frame the copy was trained on; the median distance is 1 to 2 pixels. This compares
  position only, not speed or timers. A copy that plays well only near familiar paths would score the same.
- Four starting policies from one training recipe (Phase 3B, varied starts).
- The Room 1 test set has now been used by Phase 3B and three pilots, so it serves as a development set. A fresh Room
  1 set will be generated before any claim beyond "descriptive".
- The policy's own play was recorded only from the normal Room 1 start.

The next pilot tests this: while a policy learns Room 2, keep copying its own Room 1 probabilities on its recorded
Room 1 play.

The numbers, the copies' agreement with their policies and the input hashes are in
[`docs/results/coverage-ceiling-pilot.json`](results/coverage-ceiling-pilot.json), written by
`scripts/analyze_coverage_ceiling.py`. The button-match and distance figures are in
[`docs/results/coverage-ceiling-descriptive.json`](results/coverage-ceiling-descriptive.json), written by
`scripts/describe_coverage_ceiling.py`.

## Fourth pilot: copying the policy's own play while learning Room 2

Also descriptive, four runs, one recipe, declared before any data (`config/mixed-self-distillation-pilot.json`).

The third pilot showed that a copy trained on a policy's own probabilities can carry its Room 1 skill. This pilot
used that in the retention setting: the same four starting policies were taught Room 2 exactly as in the second
pilot, with one change. The Room 1 part of the imitation step was the policy's own recorded play, fitted to its own
probabilities, instead of the Room 1 demonstrations. Fine-tuning on Room 2 then ran unchanged.

| Starting policy | Before | After mixed copying | 100k | 200k | 300k | 400k | 500k | Room 2 final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed 3 | 76% | 64% | 36% | 14% | 11% | 8% | 8% | 44 of 50 |
| Seed 4 | 77% | 63% | 47% | 48% | 43% | 40% | 26% | 46 of 50 |
| Seed 5 | 77% | 64% | 53% | 55% | 29% | 26% | 7% | 41 of 50 |
| Seed 6 | 66% | 52% | 41% | 39% | 19% | 15% | 7% | 35 of 50 |
| Floor: never saw Room 1 | | 10% to 17% | | | | | 12% to 30% | |

**Copying the policy's own play kept the Room 1 skill through the imitation step.** After mixed copying, Room 1
success was 52% to 64%, a median 78% of each policy's margin over the floor, where mixing in the demonstrations had
kept almost none (14% to 22%). Room 2 imitation was as good as before. Copying on the demonstration frames instead
of the recorded play kept a median 69%.

**Fine-tuning on Room 2 then wore it away.** Over 500,000 steps, Room 1 success fell to 7% to 26%. Measured in the
same unit (each policy's original margin over the floor), the median run dropped 0.87 of that margin against the
0.78 it had kept; three of the four dropped more than they had kept and ended below every policy that never saw
Room 1. Room 2 also paid: only one of four
final policies cleared the Room 2 start 45 or more times out of 50, against 48 to 50 in the second pilot. On a
held-out set of Room 2 starts they did about as well as the earlier recipes: 76% to 85% (median 81%), against 80% to
86% (median 82%) for the second pilot and 79% to 86% (median 81%) for the first.

**What went wrong looks like growing randomness rather than overwritten choices.** During fine-tuning these policies
became much less decisive, in both rooms. Randomness here is measured as how many fair coin tosses a policy's button
choices amount to per frame (0 means it always presses the same buttons in a given situation). For the seed 3 run it
rose from 5.3 to 9.5 in Room 1 and from 0.4 to 5.8 in Room 2, while the second and first pilots' fine-tuned policies
stayed at 1.4 or below. Of the choices the starting policy made confidently in Room 1, the fine-tuned policies still
made 55% to 83% the same way. Only 2% to 5% flipped to the opposite choice; the rest became uncertain, and 149 to 185
of the 200 Room 1 test episodes ended in a death. The run whose starting policy was least random (seed 4) stayed the
least random and did best in both rooms.

A possible cause, not yet tested: copying soft probabilities left the network less decisive than imitating single
answers does (its raw outputs on Room 2 frames were about half as large), and fine-tuning's small bonus for staying
random may then have pushed it further. The next pilot tests this directly, beside a term that keeps the policy close
to its starting self in Room 1.

Sampling noise, the reused Room 1 test set and the copied value estimate, as for the first pilot, and in addition:

- The randomness explanation is a hypothesis drawn from these four runs, not a measured cause.
- The two Room 2 measures disagree (the canonical start says worse, the held-out starts say about the same), so the
  Room 2 cost is reported, not settled.
- The Room 2 held-out set was used by the Room 2 matched fine-tuning result before, so it serves as a development
  set; only aggregates are reported.
- The starting policies differ a lot in how random they are to begin with (Room 1: 1.1 to 5.3 coin tosses per
  frame), and the least random one did best; four runs cannot separate that from chance.

The numbers are in [`docs/results/mixed-self-distillation-clone.json`](results/mixed-self-distillation-clone.json)
and [`docs/results/mixed-self-distillation-ppo.json`](results/mixed-self-distillation-ppo.json), written by
`scripts/analyze_mixed_self_distillation.py` and `scripts/analyze_mixed_self_distillation_ppo.py`; the randomness,
confident-choice, output-size and ending figures are in
[`docs/results/mixed-self-distillation-descriptive.json`](results/mixed-self-distillation-descriptive.json), written
by `scripts/describe_mixed_self_distillation.py`.

## Fifth pilot: keeping Room 1 with an anchor, or without the entropy bonus

Also descriptive, four runs per arm, one recipe, declared before training
(`config/ppo-anchor-pilot.json`). The Room 1 and Room 2 v1 sets are development sets;
only aggregates are reported. Neither fresh v2 set was used.

The fourth pilot kept the Room 1 skill through mixed copying, then lost most of it
while learning Room 2. Its growing randomness suggested two remedies:

- **E0:** turn off PPO's entropy bonus, the small reward for keeping button choices
  random. Everything else in the Room 2 fine-tuning stays the same.
- **A1 and A10:** keep that bonus, but add an anchor that penalizes departures from
  the frozen original donor's Room 1 button probabilities, with weights of 1 and
  10 respectively. It uses the donor's recorded Room 1 play: the same nineteen
  episodes the mixed copy was fitted on, with six held back for descriptive
  measurements.

All three arms start from the same four mixed copies as the fourth pilot and learn
Room 2 for 500,000 steps. The fourth pilot's unchanged fine-tuning runs serve as the
control, with entropy coefficient 0.01 and no anchor. They were run earlier and
sequentially; the remedy runs ran three at a time, with ten torch threads per job.

The loss measure is the drop from the mixed copy to the final checkpoint, divided
by the original donor's margin over the Room 1 floor of 14.2%. A loss of 0.25 means
losing a quarter of that original margin, not 25 percentage points of success.
The two floors remain those measured in the first pilot: 14.2% for a clone and
20.46% for a final checkpoint. The same Room 1 test and route-macro measure are
used throughout.

| Arm | Median loss of the original Room 1 margin | Room 2 v1 median | Room 2 finals at least 45 of 50 | Declared branch |
|---|---:|---:|---:|---|
| Control: bonus, no anchor | 0.871 | 80.6% | 1 of 4 | Reference |
| E0: no bonus, no anchor | 0.210 | 85.7% | 4 of 4 | Holds |
| A1: bonus, anchor weight 1 | 0.072 | 85.3% | 3 of 4 | Holds |
| A10: bonus, anchor weight 10 | 0.024 | 84.3% | 2 of 4 | Holds with a Room 2 cost |

The branch labels follow the declared screen. The Room 2 v1 threshold is 76.76%,
five points below the earlier mixed-imitation recipe's median of 81.76%.
Holds requires median loss at most 0.25, that Room 2 v1 threshold, and at least
three of four canonical finals clearing 45 of 50 episodes. Holds with a Room 2
cost meets the loss threshold but fails a Room 2 condition. Partial means median
loss above 0.25 and at most 0.6; Does not hold means loss above 0.6.
The canonical Room 2 measure uses 50 episodes from its normal start and is on the
training side. Room 2 v1 is a reused development set, so neither measure supports
a fresh generalization claim.

**All twelve remedy runs lost less Room 1 skill than any control run.** The largest
remedy loss was 0.362, against the smallest control loss of 0.586. Turning off the
entropy bonus alone prevented most of the decay seen in the control. The anchor
also held the skill under this screen, while keeping Room 1 randomness close to
the donor's level. The anchor frames are near the Room 1 test paths: the earlier
coverage measurement found 59% to 76% of test starts within four pixels of a fitted
frame, comparing position only. E0 has no such rehearsal term.

| Donor seed | Room 1 after mixed copying | Control final | E0 final | A1 final | A10 final |
|---|---:|---:|---:|---:|---:|
| 3 | 64% | 8% | 57% | 58% | 61% |
| 4 | 63% | 26% | 44% | 61% | 63% |
| 5 | 64% | 7% | 63% | 58% | 57% |
| 6 | 52% | 7% | 33% | 49% | 59% |

**The declared reading is a tie between E0 and A1.** Their median losses differ by
0.138 and their Room 2 v1 medians by 0.004, inside the declared tie bands of 0.15
and 0.05. E0's per-run losses were 0.116, 0.305, 0.010 and 0.362; A1's were 0.106,
0.021, 0.086 and 0.058, in donor order. E0 therefore has two runs above 0.25 and
A1 none, but four runs per arm do not establish an advantage between the arms
(the exact two-sided test on those counts gives about 0.43). On donor 5's copy,
E0 scored above A1 at every measured checkpoint. A1's smaller observed losses
must also be read beside the anchor's rehearsal near the test paths, described
above. This pilot does not show that either arm retains better than the other.

The Room 2 screen also needs care. A1's run from donor 3 cleared 40 of 50 episodes,
so there is no basis to claim no Room 2 cost for A1. A10's two finals below the
45-clear cut scored 44 and 43; its branch label records that screen, rather than
establishing a Room 2 cost from differences of one or two clears.

The descriptive frame measurements support the randomness explanation without
settling it. On the six held-back Room 1 episodes, seed 3's donor had 5.325 bits
of button-choice randomness per frame, its mixed copy 4.840, and its control
final 9.630. The remedy finals were E0 4.987, A1 5.261 and A10 5.103. Across the
four donors, A1's final Room 1 randomness was within 0.25 bits of the donor's;
E0 was less consistent. Of the choices the donor made confidently, A1 still
made 92% to 99% confidently the same way, A10 93% to 99%, E0 86% to 96%, and
the control 54% to 83%. These control shares are remeasured on this pilot's
held-back frames, rather than the frame set used for section 4. Room 2 randomness
rose from the mixed copy in every arm and every run, including E0. Removing the
bonus does not remove all randomness drift.

The owner provisionally chose E0 to carry forward, following the declared tie
choice by simplicity and cost. That is a choice, not a result establishing E0
as better. A small E0 versus A1 tie-break was then declared using new PPO seeds
and the same four copies, before any confirmation on fresh donors and the fresh
Room 1 v2 set. Its thresholds were chosen after seeing this pilot. That tie-break
has since finished and selected A1 under its declared rule; see the
[tie-break result](results/ppo-anchor-tiebreak.json). This section reports the
pilot only.

Sampling noise, the copied value estimate and the reused development sets remain
limits, as in the earlier pilots. In addition:

- Four donors from one recipe cannot establish generalization to new donors.
- Each copy's Room 1 starting score is one shared 200-start evaluation. Its
  measurement error moves every arm's loss on that copy together. As a rough
  scale, binomial approximations put one evaluation's standard error at about
  0.06 to 0.08 loss units per run. This is not a route-macro uncertainty estimate
  or an uncertainty interval for the arm comparison.
- Anchor rehearsal near test paths limits comparisons with E0, and the control
  and remedy campaigns ran at different times and with different concurrency.
- The critic is not anchored and shares features with the actor, so its updates
  can also change the Room 1 policy.
- A1-k0, A1-k1 and A1-k2 were stopped on purpose at 401,408 steps and resumed in
  a second process. Weights, optimizer state, counters and the anchor batch
  schedule carried over exactly. The resumed process restarted its random
  streams from the run seed and began from a fresh episode; A1-k2 also
  recollected one rollout, having been killed during the update after its saved
  checkpoint. The 100k through 400k checkpoints are unaffected. Only these A1
  runs were resumed, so interruption is a confound in the final-arm comparison.
- The resume amendment was committed as `9a3ee81`, with the training code
  identical to `54c9556`. The pilot declaration was already hash-pinned and
  therefore could not record that amendment in place. Analyzer record checks
  were tightened in `8add862` after all twelve training records existed,
  including the anchor's per-update held-back Room 1 agreement, and before
  evaluation results were read. They were refined in `90b9b47` to accept
  replayed anchor rows explained by a resume, check additional pins and refuse
  blank values. These changes are recorded separately in the tie-break
  declaration and did not change the pilot's numerical results or reading.

The numbers and input hashes are in
[`docs/results/ppo-anchor-pilot.json`](results/ppo-anchor-pilot.json), written by
`scripts/analyze_ppo_anchor_pilot.py`. The randomness and confident-choice
measurements are in
[`docs/results/ppo-anchor-pilot-descriptive.json`](results/ppo-anchor-pilot-descriptive.json),
written by `scripts/describe_anchor_pilot.py`. The tie choice and subsequent
tie-break declaration are in `config/ppo-anchor-tiebreak.json`.
