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
