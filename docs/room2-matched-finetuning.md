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

## Scope and audit

The fixed Room 2 task replays a pinned Room 1 exit fixture to the first controllable Room 2 frame. These results measure Room 2 competence from that setup and its held-out Room 2 states. Natural Room 1-to-Room 2 retention has not been tested.

The analyzer verified all twelve checkpoint hashes, the clean evaluation commit and pinned runtime, all result and episode-file hashes, and exactly one valid row for every frozen state in each run. A separate task audit confirmed the Room 2 task identity, zero stale starts, and 2,400 complete rows. The machine-readable analysis, including per-route results and artifact hashes, is in `docs/results/room2-heldout-ab-200.json`. Live run records remain local in `runs/campaign/20260923-033530-room2-heldout-ab-200-v1/`.
