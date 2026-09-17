# Notes from Yosh's Trackmania reinforcement learning videos

Source: five transcripts supplied by the owner (tactiq exports). Timestamps refer to each video. These are notes on
someone else's approach, taken as inspiration, not as instructions. Section 7 is the only part that proposes
anything for this project, and every proposal is checked against our hard constraints.

Videos:

| Ref | Title | Length |
|---|---|---|
| V1 | Training an unbeatable AI in Trackmania | 20 min |
| V2 | AI beats multiple World Records in Trackmania (pipes) | 37 min |
| V3 | AI exploits a bug in Trackmania (noseboost) | 23 min |
| V4 | I Trained an AI to Beat This Absurd World Record (A06) | 15 min |
| V5 | AI just Broke Trackmania's Greatest World Record (A01) | 29 min |

## 1. The setup he uses everywhere

- **Learning:** reinforcement learning from scratch, no prior knowledge, trial and error, a neural network picking
  actions from a handful of numbers (V1 01:00-02:19, V2 00:16-01:45).
- **Decision rate:** every tenth of a second in V1 (10 Hz); 20 times per second in V5 (05:02, 14:00); "it can take
  100 actions per second" in V3 (04:26). He raises it when reactions matter and notes the limit as a technical
  constraint, not a design choice (V5 13:18-13:34).
- **Observations are numbers, not pixels:** speed, position relative to the road centreline, orientation, which
  wheels touch the road and whether they slide (V1 07:07-08:48); velocity, rotation, wheel info (V3 02:28-02:39);
  "a few numbers describing the state of the car and how it is positioned on the road" (V5 02:01-02:08).
- **Layout lookahead is added only when needed:** inputs encoding the next three corners once the track stops
  being repetitive (V1 07:21-07:35); distance to the next corner and its direction on the pipe maze (V2 10:20-10:31);
  extra inputs to locate itself on the track for the noseboost run (V3 18:21).
- **Base reward:** progress along the track. "The faster the AI progresses along the track the higher the reward"
  (V1 01:49-02:02, V2 01:12-01:18 with "as long as it doesn't fall off", V4 01:07, V5 01:22-01:30).
- **Training scale:** 12 hours of driving on the simple pipe (V2 01:55); 35 hours before beating him on a road
  track (V1 09:49); 100 hours on the hardest pipe (V2 29:53); "an equivalent of 400 hours on A01, comparable to
  the top players" (V5 05:53); **2,000 hours of training on A06** (V4 09:35). Videos took 5 months to a year of
  work (V2 34:51, V3 09:55-10:50).

## 2. Exploration: starts, constraints and forced situations

- **Random spawns across the map.** "In reality, the AI regularly spawns anywhere on the map during its training.
  This prevents it from focusing too much on the first turns" (V1 07:35-07:44; V2 11:19-11:38).
- **Oversampling the hard part.** He makes it spawn more often just before the finish (V2 11:39-11:45), and later
  "the AI will spend 90% of its training in the finish area" (V2 26:51-26:56).
- **Starting from a human-driven prefix.** He drives the first seconds himself and hands over to the AI, both to
  reach an awkward state (driving backwards, V2 24:20-24:28) and to force a body position (on the nose, V3 03:37-03:44).
- **Simplify the action space at first.** He disables the brake early on: "every simplification of the decision
  making space tends to make the problem easier and quicker to solve" (V1 04:53-05:24); later re-enables it and
  gets a better result (V1 13:14-14:03). In V2 he forces permanent acceleration with steering as the only control,
  which finds a faster strategy than braking ever did (V2 06:44-08:26).
- **Constraints that end the run.** For the noseboost: if the car tilts too much or rises too high, rewards stop
  and the run ends, which forces balancing on the front wheels (V3 05:24-05:51).
- **Changing the map to block a local optimum.** He adds walls behind the ramps so the middle line is impossible,
  forcing the ramp cut, then more walls so only the flip clears the jump (V4 04:13-04:27, 06:14-06:30).
- **Practising from reference positions.** Alongside the walls, "the AI will regularly practice driving from one of
  these flip positions" (V4 06:34-06:47).

## 3. Reward shaping, and how the AI games it

This is the richest thread across the videos.

- **Short-term against long-term.** Smashing into a wall "collects more rewards initially, and it's only later that
  it turns out to be a bad decision" (V1 03:23-03:41).
- **A temporary bonus to unlock a skill.** The AI would not drift, so: "from now on, the AI will get a big reward
  bonus whenever it's drifting", detected by the car pointing away from its direction of travel (V1 15:33-15:52).
  - **It gamed it immediately:** "it found a way to constantly trigger the reward bonus, just by spamming these
    weird action patterns at low speed" (V1 16:08-16:27).
  - **Patch:** the bonus only counts above a speed threshold (V1 16:20-16:27).
  - **It gamed the patched version too,** chaining pointless drifts in straight lines for reward (V1 16:55-17:02).
  - **Removal:** "let's continue the training without the bonus. Now that it discovered how to drift, the AI
    shouldn't forget it" (V1 17:02-17:09). It then drifted "only where it saves time" (V1 17:33-17:42).
- **A section-specific bonus.** On A01 he gives extra reward for drifting only inside one corner section
  (V5 08:14-08:22). The AI gets slower first, then finds the technique and crushes the record (V5 08:34-09:43).
- **Changing the goal near the finish.** In the finish area the reward becomes distance to the finish "regardless of
  whether it's following the path or not", plus "a massive bonus" for crossing the line (V2 12:08-12:31).
- **Staged rewards.** Noseboost: reward raw speed until the trick is learned, then "replace this signal with a new
  one that only rewards the AI when it's making progress along the road" (V3 18:05-18:21).
- **Risk appetite is a reward-shape property.** With reward equal to current speed, the AI refused the risky
  noseboost: "instead of taking that risk, the AI probably learned it's better to aim for smaller rewards, as long
  as it can accumulate them over a longer time" (V3 07:20-07:57). Fix: **shift the reward** so that ordinary speed
  earns almost nothing and only the boost pays. "This is not a big fix. Just one small change in the AI code...
  Yet, it changed everything" (V3 07:59-08:29).
- **Other gaming he hit:** sliding upside down to collect speed reward (V3 04:47-04:56), and finding ways to land on
  four wheels to collect rewards safely (V3 05:01-05:14).
- **Punishing failure harder did not help consistency** on the pipes (V2 15:26-15:33).
- **Rewarding imitation of a reference run disappointed:** "I tried rewarding the AI for following the TAS path. But
  the results were quite disappointing" (V5 17:59-18:07).

## 4. What the AI is good and bad at

- **Good:** precision and consistency over long tasks. "I think that's what makes it so strong, in this kind of
  endurance scenario" (V1 12:12-12:18); "more precise, more consistent" (V5 ~19:07 in V1 numbering).
- **Bad: creativity and long-delayed payoffs.** It never discovered turning the car around when forced to drive
  backwards: "the AI would need to perform a precise sequence of actions by chance, without immediate positive
  feedback. The payoff would only come a long time after. This is quite unlikely to happen, without clear and
  guided indications" (V2 25:23-25:41). Same reasoning for not finding the world-record strategy on the hardest
  pipe and for needing the brake forced off (V2 30:34-30:48).
- **It optimises the average case, not the best case.** On A06: "I suspect the AI isn't aiming for the fastest
  approach, but rather for something that works reasonably well on average" (V4 12:36-12:45). Humans vary their
  lines, fail more, and occasionally get a "freak accident" that is faster (V4 11:44-12:23).
- **Local optima are sticky.** On A06 it avoided the ramp sides because early on it was not precise enough to profit
  from them, "and from there it just kept refining that line, without ever getting any hint that something better
  exists" (V4 03:44-04:09).
- **Several training runs find different strategies**, some better than others (V2 05:42-06:28; V3 15:19-15:46).

## 5. Consistency, determinism and chaos

- Trackmania is deterministic, so he **injects a tiny steering perturbation in the first tenth of a second** to make
  runs differ, which is how he gets many attempts from one policy (V2 16:18-17:00).
- Tiny perturbations change outcomes completely, and a perturbation seconds before a fall usually removes the fall
  (V2 17:51-18:25). He links this to chaos theory (V2 33:18-34:02).
- Per-corner success on the pipe maze was about **97.3%**, which over many corners makes a clean run rare, so he
  "just needs one good run" out of thousands (V2 22:13-22:39).
- On A06, identical approaches with identical actions gave different jump outcomes: "my guess is that this jump is
  pretty much random" (V4 10:14-10:49).
- **Final phase without learning:** "there's no further training. The AI has to stick with the weaker strategy it
  chose. And with that, hope there's still some room for a lucky run" (V4 13:20-13:39).

## 6. Beyond plain RL

- **Hybrid control.** He writes a small program that holds the optimal drift angle, and lets the policy hand over
  control during a drift and take it back (V5 13:43-14:13). The AI then used it in every run.
- **Segmented runs.** Drive segment one many times, keep the best, use it as the start of segment two, repeat, with
  a small random steering change at each segment start (V5 15:46-16:33).
- **Brute force search.** The TAS method: take an action sequence, apply small random changes, replay, keep
  improvements, guided by a measured quantity such as the car's height (V5 21:16-21:55, 23:48-24:07). This found a
  previously unknown hole in the road on the most studied track in the game (V5 23:06-23:29).
- **Borrowing from other projects.** Progress came from rewriting the learning algorithm after seeing Linesight, a
  similar open project (V3 13:01-13:18).

## 7. What transfers to this project, and what does not

Our hard constraints: game physics never modified; observations are game state with no route hints; actions are
only what the controls menu allows; reliability before speed; an honest no-demonstration baseline first.

**Transfers well**

1. **Varied starting states.** His single most repeated technique, and the one we lack entirely: we train only from
   the canonical start. Celeste equivalent: savestates at many legal, reachable points in the room, sampled during
   training, with the hard section oversampled. This also gives Phase 3's held-out entry states.
2. **A progress signal.** Every one of his videos rewards progress, not just completion. Our unshaped baseline has
   no such signal, which is exactly why 2M steps produced nothing. Celeste differs from a race track: progress is
   not one-dimensional, and sometimes you move away from the exit to reach it, so the measure has to come from
   reachable-platform distance rather than straight-line distance to the exit.
3. **Temporary skill bonuses, then removal.** The drift story maps directly onto dashing, wall jumps and climbing:
   reward the skill briefly to get it discovered, verify it is used, then remove the bonus and keep only the task
   reward. Removal is what stopped his AI drifting pointlessly.
4. **Expect the reward to be gamed, and instrument for it.** He was gamed three times in one video. We already log
   reward components per episode; the test is whether a component rises while clears do not.
5. **Risk appetite is a property of the reward shape.** His "shift the rewards" fix is the same phenomenon as our
   measured "learns to die faster": with every ending worth about -1, the only gradient is the time cost. Options
   are to remove the time cost during the reliability stage, or to offset the reward so that surviving and
   progressing is worth more than ending the episode.
6. **Staged rewards.** Reward the sub-skill first, then switch to task progress once it exists.
7. **A final no-learning attempt phase.** Freeze the policy and take many attempts. This is how we should report
   best-case clears, separately from the reliability numbers.
8. **Sampling variation from one policy.** He perturbs the first action because Trackmania is deterministic. Our
   policy samples actions, so we get variation for free, but the same idea gives cheap "many attempts from one
   checkpoint" evaluation, and our greedy evaluation is one repeated trajectory, as we already noted.
9. **Scale.** He trains 12 to 2,000 hours of game time per track. Our 2M frames is about 9 hours of game time.
   We are one to two orders of magnitude below his budgets, which is important context for "PPO cannot do it".

**Transfers with care**

10. **Constraints that force exploration.** He adds walls to the map. We cannot modify the game. Our legal
    equivalents: restricting the action set (he disables the brake; we could temporarily disable inputs), starting
    states that make the easy route impossible, and ending episodes early on a condition.
11. **Hybrid control (auto-drift).** A scripted helper the policy can call would need disclosing and would weaken
    the claim that the agent plays the game. Possible later as a declared experiment, not in the baseline.
12. **Brute force search.** We already have a Go-Explore style route finder for fixtures. Useful for demonstrations
    and for speed phases, but it is not a learned policy and must never be presented as one.

**Does not transfer**

13. **Chaos and random physics.** Trackmania's pipe and ramp behaviour is effectively unpredictable, which caps his
    consistency. Celeste is deterministic and our replays are frame-identical, verified by the probes. Our
    consistency problems will be our own, not the game's.
14. **Imitating a reference path by reward.** He reports this as disappointing. If we use our recorded clear, use
    behavioural cloning or start states from it, not a reward for following it.
