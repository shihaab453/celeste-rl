# Next experiment: design proposal (draft for review)

Status: proposal. Nothing here is implemented. It follows the unshaped no-demonstration campaign (three seeds,
2,000,000 accepted transitions each, zero clears) and the notes in `inspiration-trackmania-notes.md`.

The campaign result is kept as the declared baseline. Everything below is a separate, versioned experiment on top of
it, so the comparison stays honest: one baseline, then changes introduced in a stated order.

## 1. What the baseline actually showed

| Observation | Evidence |
|---|---|
| No clears at all | 3 seeds, about 54,000 episodes, 21 evaluations, all deaths |
| The agent never got past the first spike pit | Checkpoint replays: median furthest right x = 76 (the pit spans x 40 to 80), best 140, exit at x 261 near the top |
| Training changed progress not at all | An untrained network of the same architecture reaches the same range (median 76, best 97) |
| The only thing it learned was to end episodes sooner | Median episode length fell 107, 87, 69, 67 frames by quarter of the run |

The cause is not a bug: with completion the only positive reward, every episode scores about -1, so the only
gradient available points at the small time cost, and the shortest path to less time cost is dying sooner.

Two further facts frame the proposal:

- **Scale.** 2M frames is about 9 hours of game time. The Trackmania videos use 12 to 2,000 hours per track, and
  400 hours for a result "comparable to the top players". We are one to two orders of magnitude below that.
- **Determinism.** Celeste replays are frame-identical here, verified by probes. The consistency ceiling that limits
  the Trackmania agent (physics that behave unpredictably) does not apply to us.

## 2. The four concepts worth taking

1. **Train from many states, not one start.** Spawning across the map, and oversampling the hard part, is the most
   repeated technique in those videos. It turns one long credit-assignment problem into many short ones.
2. **Reward progress continuously, not only completion.** A per-step progress signal is what makes their agents move
   at all. Ours has none.
3. **Use temporary skill bonuses, then remove them.** Pay for a skill until it appears, expect the bonus to be
   gamed, gate it, then delete it and keep only the task reward.
4. **Reward shape decides risk appetite.** Their agent avoided a risky trick until the reward was shifted so that
   only the risky behaviour paid. Ours prefers dying early for the same kind of reason.

## 3. The proposal, in the order it should be tried

Each stage is a separate experiment with its own version tag and run directory, and each one changes as little as
possible from the stage before.

### Stage A: record progress during training (no behaviour change)

Per episode, log furthest right, highest point, furthest progress value, and the position at the end. Costs nothing,
needs no game change, and removes the blind spot we hit when the baseline finished: we had to replay checkpoints to
learn where the agent died.

**Done when:** the training records answer "how far did it get" without a separate analysis run.

### Stage B: varied start states (`starts-v1`)

The agent still trains on the same task, but episodes begin from many legal states rather than only the canonical
start.

- **Source of states.** Replay a stored action prefix from the canonical start: the game is deterministic, a reset
  is about 5 ms, and 300 frames of replay is about 0.1 s at current speed. No new savestate machinery.
- **Which states.**
  1. **Agent-reached states:** positions the current policy actually reaches, archived during training. This is the
     honest, no-route-information source.
  2. **Route-derived states:** points along the legal exit route we already have from the Go-Explore style search.
     These carry route information, so a run using them is labelled "route-assisted" and reported separately.
- **Distribution.** A configurable mixture, with the section before the exit oversampled, as in the pipes video
  (90% near the finish while learning the jump).
- **Evaluation is unaffected:** always from the canonical start, plus the held-out reachable entry states the
  roadmap already requires. Training starts never enter evaluation.

**Risks to watch:** starts that are unreachable in a real attempt (would flatter the result), and an agent that
learns only the last section. Both are caught by canonical-start evaluation.

**Done when:** clears appear from starts near the exit, and the success rate from progressively earlier starts is
measurable.

### Stage C: progress reward (`rew-v2`)

A potential-based shaping term, off in the baseline (decision D3), turned on here.

- **Form:** `F = gamma * potential(next) - potential(current)`, gamma 1.0, potential of any terminal state 0, scale
  about 0.2. This form sums to a constant over any complete episode, so it cannot be farmed by pacing back and
  forth, which the Trackmania bonuses repeatedly were.
- **Potential:** distance to the exit measured along legal movement, not straight-line distance. Celeste needs
  vertical movement and sometimes moving away from the exit, so a straight-line measure would actively mislead.
  Two candidates:
  1. **Route projection:** progress along the recorded legal route by nearest point. Simple, obviously
    route-derived, must be disclosed.
  2. **Reachability graph:** breadth-first distance over reachable tiles or platforms, built from the room's
    collision data. Less hand-made, more work, and only a heuristic for real Celeste movement.
- **Reward data never enters observations.** The potential is computed in the reward path only, which the current
  separation already enforces and a test already asserts.

**Done when:** the shaped run reaches further than the baseline from the canonical start, measured by Stage A's
records, not by the shaped return.

### Stage D: fix the early-death incentive (`rew-v2`, same tag)

Options, in order of preference:

1. **Time cost zero during the reliability stage,** reintroduced later for speed. This matches "reliability first,
   then speed" and makes dying early no better than trying.
2. **Shift the reward** so that progress pays and mere survival does not, the Trackmania fix, if a time cost has to
   stay for the owner's constraint.

Whichever is chosen, it is one change, applied at a stated point, with the baseline preserved.

### Stage E: temporary skill bonuses, if still needed

Only if B, C and D leave the agent unable to perform a specific move (for example a dash across the first pit).

- Pay for the move under tight conditions (for example a dash that gains horizontal distance while airborne), watch
  the component logs for gaming, gate the condition when gamed, and **remove the bonus once the move appears**.
- Every bonus is a separate version tag, and each is reported with the episode where it was added and removed.

### Stage F: scale, and a final no-learning attempt phase

- **Budget:** raise the per-seed budget from 2M once the shaped setup shows progress, in one declared step (for
  example 10M, about 11 hours at current speed), rather than extending a running campaign.
- **Attempt phase:** freeze the policy and take many attempts, reporting best-case clears separately from the
  reliability numbers. The Trackmania videos do exactly this, and it is also how the roadmap's held-out evaluation
  should be read.

## 4. What is deliberately not taken

| Their technique | Why not |
|---|---|
| Adding walls to the map to force exploration | Modifying the game is a hard constraint. Our legal equivalents are start-state choice and temporarily restricting the action set |
| A scripted helper the policy can hand control to (auto-drift) | Would weaken the claim that the agent plays the game; possible later as a declared, disclosed experiment |
| Rewarding the agent to follow a reference path | Reported as disappointing in the A01 video; if we use our recorded clear it should be start states or behavioural cloning |
| Brute force action search | Already exists here as a fixture generator; it is not a learned policy and must never be presented as one |

## 5. Open questions for review

1. Is the staged order right, or should the progress reward (C) come before varied starts (B)? C is the smaller
   change; B is the one their videos rely on most.
2. Potential from a route projection or from a reachability graph: which gives an honest result with least
   hand-made room knowledge, and how should either be disclosed?
3. Does the potential-based form remove the farming risk in a platformer, where moving away from the exit is
   sometimes necessary, or does the potential itself need to encode that?
4. Is removing the time cost during the reliability stage acceptable, given the owner's constraint that a small
   time penalty exists from the start, or is the reward shift the better fit?
5. Agent-reached start states risk a feedback loop: the agent practises where it already goes. Is an archive with
   coverage weighting needed from the start?
6. What must be recorded for the comparison against the baseline to be credible, beyond Stage A's per-episode
   progress?
7. Is a 10M budget per seed the right next step, or should the first shaped run be short and diagnostic?
