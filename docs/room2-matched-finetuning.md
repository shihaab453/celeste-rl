# Room 2 matched fine-tuning comparison

## Question and fixed design

Does varied-start fine-tuning improve Room 2 success over canonical-start fine-tuning? Both arms started from the same clone of seven replay-verified Room 2 demonstrations. The clone was trained on five whole routes and checked on two whole routes. It was not evaluated on the frozen Room 2 starts before fine-tuning.

The training plan was committed as `813a4d0` before fine-tuning. Six matched seeds (20 through 25) ran in A/B order. Arm A used canonical starts. Arm B used success-weighted starts from its own reached-state archive, with a 25% canonical-start floor. Both arms used `rew-v2`, shaping scale 2.0, and 500,000 requested transitions. Every run completed 501,760 accepted transitions with zero discarded rollouts and zero game relaunches. The only declared held-out checkpoint from each run was its final `latest.zip`.

The evaluation plan was committed as `1913b4d` before any held-out episode. It pinned all twelve final checkpoint hashes. Each checkpoint played one stochastic episode from each of 200 frozen, validated Room 2 states, for 2,400 evaluated episodes. The states came from twelve unique complete routes searched with seeds 100 through 111 and had no full-route overlap with the seven demonstrations. Each checkpoint evaluation used seed 20260922. The held-out manifest SHA-256 is `ce57def1b3daed78a5aa3e454ab20cc12858a50b764bd0d94f80a1b2d734f8b0`.

## Predeclared analysis and result

Within each matched seed, success was averaged equally across the twelve complete-route clusters. The primary estimate is the mean of the six paired B minus A route-macro differences. The exact two-sided sign-flip test enumerated all 64 assignments of the six paired differences. The 95% crossed seed-route bootstrap independently resampled matched seed pairs and complete routes in 100,000 draws, retaining A/B pairing and all states within each selected route. The predeclared rule required a positive estimate, a bootstrap interval excluding zero, and p <= 0.05 to support B.

| Primary measure | A, canonical starts | B, varied starts | B minus A |
|---|---:|---:|---:|
| Route-macro success | 86.8% | 80.9% | -5.8 points |
| Crossed seed-route 95% interval | | | -18.7 to +3.4 points |
| Exact paired sign-flip test | | | p = 0.50 |

**The predeclared rule was not met.** The observed mean favoured A, but the interval includes zero and the paired test does not establish a difference. Varied starts improved three matched seeds and reduced success in three. Seed 25 had the largest negative B minus A difference and remains in the analysis.

With six seeds the exact test has only 64 sign assignments, so it can reach p <= 0.05 only when all six seed differences have the same sign (p = 2/64 = 0.031); a single seed with the opposite sign gives at least 4/64 = 0.0625. The Room 1 comparison had the same property.

## Matched-seed detail

| Seed | A route-macro | B route-macro | B minus A, points | A successes | B successes |
|---:|---:|---:|---:|---:|---:|
| 20 | 86.9% | 77.7% | -9.2 | 173/200 | 154/200 |
| 21 | 93.0% | 91.4% | -1.6 | 185/200 | 183/200 |
| 22 | 86.9% | 91.7% | +4.7 | 173/200 | 183/200 |
| 23 | 87.1% | 91.1% | +4.0 | 174/200 | 182/200 |
| 24 | 84.3% | 86.4% | +2.0 | 169/200 | 172/200 |
| 25 | 82.2% | 47.3% | -35.0 | 166/200 | 94/200 |

The state-weighted totals are descriptive: A cleared 1040/1200 (86.7%); B cleared 968/1200 (80.7%). These totals do not replace the route-macro matched-seed analysis because states from one route are adjacent trajectory prefixes.

In seed 25, B scored 47.3% route-macro success against A's 82.2%, and B timed out on 57 of 200 starts. This is a descriptive observation, not a reason to remove the seed or select a different checkpoint.

## Seed 25 B in its training records

Everything in this section is descriptive. It does not replace the predeclared analysis, which includes seed 25.

Training records, which never touch the held-out states, show that B seed 25 stopped clearing from the canonical start early in training. It cleared 2 of its first 36 canonical-start episodes (the first 50,000 accepted transitions) and 0 of the remaining 150. Of those 150, 65 timed out, 45 of them ending at world x 320 to 329, and 85 died further right, at world x 372 to 443. In the final fifth of training the other five B runs cleared 72% to 89% of their canonical-start episodes and the six A runs cleared 83% to 96%; B seed 25 cleared none of its 45.

On the held-out set, 37 of B seed 25's 57 timeouts never got past world x 320 to 329, the same place where 45 of its 65 training timeouts ended. Its failures are concentrated in starts early in the room:

| Start depth, frames into Room 2 | States | A seed 25 | B seed 25 | A seeds 20-24 | B seeds 20-24 |
|---|---:|---:|---:|---:|---:|
| 0 to 99 | 39 | 31/39 | 0/39 | 172/195 | 170/195 |
| 100 to 199 | 43 | 27/43 | 3/43 | 153/215 | 155/215 |
| 200 to 299 | 49 | 40/49 | 27/49 | 209/245 | 216/245 |
| 300 and later | 69 | 68/69 | 64/69 | 340/345 | 333/345 |

B seed 25 cleared 3 of the 82 starts in the first 200 frames, against A seed 25's 58 of 82. Across seeds 20 to 24 the mean B minus A route-macro difference was 0.0 points, and pooled over those five seeds the two arms were within 3 points of each other at every start depth. Six seeds cannot estimate how often a varied-start run fails in this way.

Starts 300 or more frames into the room were near ceiling for both arms: 67 to 69 of 69 in each A run and 64 to 69 of 69 in each B run. A's overall Room 2 success of 86.8% also left much less room to improve than Room 1, where A scored 52.2%.

## Where training episodes fail

Everything in this section is descriptive. It uses training records, the room's map data and the seven demonstrations, not the held-out states.

Both arms fail in the same short stretch of the room. In every one of the 12 training runs, 93.5% to 99.6% of training deaths ended between world x 360 and 459, and no training death ended left of x 372. The game's map data for this room (`1-ForsakenCity.bin`, room `lvl_2`) shows what that stretch contains: a bottomless pit from x 368 to 407, a block from x 400 to 439 whose top (y -72) is covered in upward spikes, and a gap from x 432 to 463 with upward spikes along its floor.

Just before the pit, on a raised step at x 352 to 367, the map places a spring, an object that bounces the player upward. All seven demonstrations touch it, 11 times in total. On every touch the player's speed becomes exactly (0, -185) pixels per second in that frame, which is what the game's spring bounce sets, and 10 of the 11 touches also gave back a used dash (in the other the dash was still available). After its last bounce, every demonstration starts a dash between x 364 and 400 and passes above the spiked block.

The policy's observation has no channel for springs. Its local grid shows the step as a plain solid block with empty cells above it. The policy does receive its own position, speed and dash count, so in this single fixed room it could still learn where the spring is. Whether the missing spring contributes to the failures in this stretch is a hypothesis that these runs do not test.

## Scope and audit

The fixed Room 2 task replays a pinned Room 1 exit fixture to the first controllable Room 2 frame. These results measure Room 2 competence from that setup and its held-out Room 2 states. Natural Room 1-to-Room 2 retention has not been tested.

The analyzer verified all twelve checkpoint hashes, the clean evaluation commit and pinned runtime, all result and episode-file hashes, and exactly one valid row for every frozen state in each run. A separate task audit confirmed the Room 2 task identity, zero stale starts, and 2,400 complete rows. In every run, `latest.zip` and `step_000501760.zip` differ in bytes: the only differing archive member is `data`, and within it only the serialized `observation_space` entry, which decodes to an equal space. Policy weights and optimizer state are byte-identical, and `latest.zip` is the file whose hash the evaluation plan pinned. The machine-readable analysis, including per-route results and artifact hashes, is in `docs/results/room2-heldout-ab-200.json`. Live run records remain local in `runs/campaign/20260923-033530-room2-heldout-ab-200-v1/`.
