# Celeste RL: evidence, decisions, and phased roadmap

Prepared 16 September 2026, incorporating both supplied documents.

## Recommendation

Build a state-based, closed-loop policy for the real Steam game, with a Gymnasium interface and PPO as the initial learning algorithm. Keep one shared policy architecture from room 1 onward. Use Stable-Baselines3 for the first trusted baseline, then study or adapt a small, attributed CleanRL-style loop only when that helps understanding or an experiment.

The immediate deliverable is a measured, trustworthy game interface. This is a separate milestone from training. The live probes demonstrated existing JSON state export, single-frame advancement, and savestate restoration without custom C#. Across 1,000 measured resets, mean latency was 25.99 ms, median 24.01 ms, and p99 51.97 ms. The neutral one-frame loop averaged 32.63 ms, about 30.6 operations/s before learning costs. We also found a headless compatibility crash and corrected a probe startup race. The detailed measurements and their limits are in the [companion feasibility report](feasibility-report-2026-09-16.md).

Do not commit to full-game completion or competitive speedrunning on a calendar yet. A reliable first room is a credible first result. A continuous Chapter 1 run is already a substantial portfolio project. The main A-side story comes next; Core, B/C-sides, and Farewell remain later research stages.

## 1. Prior art and verified tooling

### 1.1 Everest, CelesteTAS, and Speedrun Tool

These are three separate components. Everest loads instrumentation mods. CelesteTAS provides precise input playback, stepping, inspection, and speed control. Speedrun Tool supplies the savestate implementation used by CelesteTAS.

| Capability | Evidence and implication |
|---|---|
| Steam installation | Everest officially supports Steam. The scratch test used your installed Steam executable and assets. |
| State from Python | CelesteTAS exposes `/tas/game_state` as JSON over Everest's localhost DebugRC server. This was exercised locally. |
| Additional internal fields | `/tas/info`, custom info templates, and target queries expose more than the JSON schema. Availability of each proposed field still needs a schema test. |
| Precise inputs | TAS files express frame counts and combinations, including crouch dash, alternate buttons, and movement/dash directions. Do not equate all TAS commands with legal controller inputs. |
| Frame advancement | `/tas/sendhotkey?id=FrameAdvance` exists and worked in the rendered probe. An HTTP acknowledgment alone does not mean the frame has completed. |
| Playback | `/tas/playtas?filePath=...` starts a local input file. This is useful for verification and demonstration capture. |
| Savestates | CelesteTAS calls Speedrun Tool. The rendered probe used a `***S` breakpoint and restart to restore it. This is an in-memory checkpoint, not a durable training checkpoint. |
| Speed control | CelesteTAS's update loop executes repeated normal-sized simulation steps. Increasing wall-clock throughput must never be implemented by enlarging the physics timestep or enabling Assist Mode speed changes. |
| Headless | Everest contains a headless installation mode; CelesteTAS's sync checker uses it. The tested combination crashed in a Speedrun Tool audio hook, so headless is not currently a validated configuration for this project. |
| Studio protocol | Shared-memory communication, binary serialization, approximately 16 ms polling, and fixed shared names exist. A Python port is possible in principle, but porting this whole editor protocol is not the first choice. |
| Multiple instances | Investigate per-worker HTTP ports, separate save directories, and disabling Studio communication. The default shared-memory names cannot safely be assumed to isolate workers. This has not been benchmarked. |

Sources: [Everest installation](https://everestapi.github.io/), [CelesteTAS HTTP endpoints](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/CelesteTAS-EverestInterop/Source/EverestInterop/DebugRcPage.cs), [JSON state schema](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/StudioCommunication/GameState.cs), [Studio transport](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/StudioCommunication/CommunicationAdapterBase.cs), [savestate integration](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/CelesteTAS-EverestInterop/Source/Playback/SavestateManager.cs), [playback loop](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/CelesteTAS-EverestInterop/Source/Playback/Core.cs), [headless installer](https://github.com/EverestAPI/Everest/blob/dev/MiniInstaller/Program.cs), [input documentation](https://github.com/EverestAPI/CelesteTAS-EverestInterop/wiki/Input-File).

**Near-zero C# is a hypothesis with a credible starting point, not a promise.** Try existing endpoints first. If file reloads, missing fields, synchronization, or per-instance isolation make this fragile, build a small Everest adapter. Its only responsibilities should be input delivery, exact step acknowledgment, coherent state export, checkpoint restoration, and process health. Keep rewards, observation construction, curricula, learning, and analysis in Python. Do not rewrite savestate cloning or physics.

The bridge should return an episode ID and monotonic request/step ID with each observation. The current HTTP endpoints are inspection conveniences, not an atomic `step(action) -> observation` protocol. Pause or otherwise synchronize before reading. Reject stale replies and preserve the terminal observation before resetting.

### 1.2 Existing Celeste learning projects

| Project | What its authors report | What to take from it |
|---|---|---|
| [Celeste-NEAT](https://github.com/hdrien0/Celeste-NEAT) | Simple Steam-game rooms, approximately 10 generations of 150 agents for the first room; difficulty with anticipation and complex interactions. | Real full-game instrumentation precedent. Its objective angle/distance inputs violate your observation constraint, so its result is not directly comparable. |
| [Celeste-Instrumentation](https://github.com/hdrien0/Celeste-Instrumentation) | A C#/Python socket interface, state exchange each frame, configurable speed and graphics. Installation patches the XNA executable using dnSpy and Harmony. | Study the request/response design. Do not adopt it directly: it is a different integration route from Everest/CelesteTAS and needs a behavior audit. |
| [Project Zoran: CelesteBot](https://projectzoran.com/assets/2019/04/2b307-project-zoran-2.pdf) | An Everest NEAT bot with a 30x30 geometry grid, checkpoint guidance, and 10x execution; about 20 hours over 1,200 generations of 30 organisms, reaching the third room. | Geometry caching, serialization, and throughput mattered. Checkpoint guidance did not by itself solve chapter-long exploration. |
| [Celeste-AI report](https://git.betalupi.com/Mark/celeste-ai/raw/branch/master/report/main.pdf) | A DQN agent cleared the first stage after about 4,000 episodes, around eight hours, with checkpoint rewards. | This is **Celeste Classic**, not your Steam target. Useful reward/normalization lessons, not an environment or sample-count estimate for this project. |
| [Madeline's Policy Climb](https://github.com/dhrumilp15/madelines-policy-climb) | Public DQN/NEAT code and a project report exist. | A research lead. I did not verify its report's outcomes, so I do not count it as evidence of chapter or game completion. |

NEAT evolves populations of networks through selection and mutation. It is a different learning approach from the gradient-based reinforcement learning you want to study. The reports above are author-reported results, not reproductions performed here. I found no verified end-to-end solution of your full target in the sources reviewed; that is not a claim that none exists.

### 1.3 Comparable work

- [PyTorch's Mario tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html): useful for seeing how game interaction, reward, and network updates fit together. Its DQN stores past transitions in a replay buffer, a dataset reused for later updates. Read for concepts; do not import its emulator assumptions or old dependencies into Celeste.
- [CoinRun](https://arxiv.org/abs/1812.02341): platformer research explicitly separating training levels from unseen test levels. Adopt the evaluation distinction: solving trained rooms does not establish generalization.
- [Go-Explore](https://arxiv.org/abs/2004.12919): preserve interesting reached states, return to them, and explore onward. This is a later fallback for exploration, with a separate requirement to produce a policy that works without restores at evaluation. Do not implement the full method before basic PPO fails.

### 1.4 Python compatibility and library choice

The local interpreter check confirmed Python **3.14.6**, torch **2.14.0+cu130**, and CUDA available. That verifies your current PyTorch environment, not the entire RL stack.

| Tool | Verified metadata | Decision |
|---|---|---|
| Gymnasium 1.3.0 | Published requirement is Python >=3.10; published classifiers list through 3.13. The development branch adds 3.14 explicitly. | Use its standard environment interface. Do not claim a published 3.14 validation from development metadata. |
| Stable-Baselines3 2.9.0 | Published requirement Python >=3.10, torch >=2.8 and <3, Gymnasium >=0.29.1 and <2; classifiers list through 3.13. Inspected CI also listed through 3.13. | Use PPO as the initial reference implementation. A permissive version bound does not prove all Windows extras work on 3.14. |
| CleanRL | Published 1.2.0 requires Python <3.11 and pins old dependencies. Inspected development metadata also excludes >=3.11 and pins a different stack. | Do not `pip install cleanrl` into this project. Study individual implementations; any adapted file becomes a project-maintained implementation with its own tested dependencies. |

Sources: [Gymnasium published metadata](https://pypi.org/pypi/gymnasium/json), [Gymnasium development metadata](https://github.com/Farama-Foundation/Gymnasium/blob/main/pyproject.toml), [SB3 published metadata](https://pypi.org/pypi/stable-baselines3/json), [SB3 CI](https://github.com/DLR-RM/stable-baselines3/blob/master/.github/workflows/ci.yml), [CleanRL metadata](https://github.com/vwxyzjn/cleanrl/blob/master/pyproject.toml).

**Pin the new RL environment to Python 3.12, using an available security-patched 3.12 release and recording its exact patch version in the lock/setup instructions.** Keep the existing exercise environment intact. Start dependency resolution with Gymnasium 1.3.0 and SB3 2.9.0. Select and lock a PyTorch CUDA wheel that passes a real GPU operation on both machines; do not copy the laptop's virtual environment to the desktop. The current 2.14/cu130 installation is a candidate, not a verified 3.12/desktop lockfile. Avoid installing all optional Atari, MuJoCo, and Box2D dependencies.

This is a conservative compatibility decision, not a claim that Gymnasium/SB3 categorically cannot run on 3.14. No RL dependency installation or training was performed during this planning work.

## 2. Proposed learning design

### 2.1 The vocabulary and algorithm

A **policy** is the neural network's rule for choosing actions from observations. An **episode** is one attempt, ending in completion, death, or the agreed task deadline. A **reward** is feedback after acting. A **return** adds rewards across an attempt. A **rollout** is a collected sequence of observations, actions, rewards, and ending signals.

Use **PPO, Proximal Policy Optimization**: collect experience with the current policy, estimate which actions worked better than expected, then make several restrained updates. It is an **actor-critic** method: the actor chooses actions, while the critic estimates future return. The **advantage** estimates how much better an action was than the critic expected. **GAE, Generalized Advantage Estimation**, combines short- and longer-horizon estimates to reduce noise. Its mixing parameter is a tuning choice, not a game rule. The PPO probability ratio compares how likely an action is under the new and previous policy. Clipping limits the incentive for excessively large changes; it does not guarantee perfect stability.

PPO is a sensible first baseline because it supports combinations of discrete inputs, is well studied, and has readable implementations. It is not proven to be the most sample-efficient algorithm for Celeste. Its **on-policy** nature means it mainly trains on freshly collected experience. A replay-based alternative could become attractive if stepping remains expensive, but changing algorithms before fixing exploration and interface errors is unlikely to help. Sources: [PPO paper](https://arxiv.org/abs/1707.06347), [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [CleanRL PPO explanation and implementations](https://docs.cleanrl.dev/rl-algorithms/ppo/).

Start with SB3 so the bridge and reward are the main unknowns. Read its rollout collection and PPO update with an annotated walkthrough. If a readable standalone learning loop is only a modest extra step after that baseline, adapt one with attribution and compare it against SB3 on CartPole and the same Celeste experiment. Do not debug two unvalidated algorithms simultaneously.

### 2.2 Observations and a shared policy

Initial proposed observation, subject to the bridge audit:

- Player-relative and room-relative position, including subpixel remainder; velocity, facing, grounded/contact state, stamina, dash availability, and movement-state identifier.
- Current jump/dash/grab-related timers and input buffer state where accessible and relevant. Include previous held inputs so press/release edges can be understood.
- A player-centered semantic grid, initially 32x32 cells at native tile scale, with separate channels for solid terrain, one-way platforms, oriented hazards, moving solids, and relevant interactables. Preserve sub-tile positions in the scalar/entity features.
- A coarse whole-current-room geometry view if the local crop hides necessary layout. This shows existing geometry, not a recommended exit, route, checkpoint, or future trajectory.
- Current dynamic entity positions/velocities and states in a bounded, masked entity list when a raster alone is inadequate. Define and test overflow behavior; do not silently omit nearby hazards.
- Elapsed attempt time if the task has a genuine finite deadline. Time is current information, not a route hint.

The grid is justified because scalar player state cannot tell a transferable controller where walls or spikes are. A semantic channel means physical categories, not treating texture IDs such as `1`, `3`, and `7` as quantities. Validate the conversion against actual collision information. A small convolutional encoder can recognize local patterns; concatenate its output with the scalar features before a small policy/value network. Begin with roughly two 128-unit hidden layers after encoding, then profile. This is a starting engineering scale, not a tuned result.

Never provide desired movement direction, distance to a selected objective, future hazard positions, reference actions, checkpoint index, or a room-specific target embedding to the policy. Keep privileged training rewards in a separate code path, outside automatic observation flattening. A room ID may be logged but should not be a default policy feature.

Exact internal-state access must be disclosed. It makes this a state-based controller with privileged observations. It does not make it pixel-based or human-equivalent play. Omitting route hints also does not guarantee transfer: coordinates and geometry can still be memorized. History may be necessary when the same observation admits different required actions. Start with a short stack of recent observations; add a recurrent network, which carries memory between steps, only when failure evidence supports it.

### 2.3 Inputs

Define one physical control configuration and audit every vanilla bind exposed by the installed game. Represent held button states and analog axes, if used, then let the game's ordinary bindings resolve them. Include crouch dash and its combinations, release/repress behavior, alternate bindings, menu inputs, and independent movement/dash bindings where physically available.

Do not quietly collapse the task into left/right/jump/dash only. Do not treat debug commands, savestate controls, speed control, `Set`, `Invoke`, or teleport as policy actions. Physically pressable opposing keys should follow the game's actual resolution rules; avoid inventing a prohibition or allowing impossible independent virtual axes.

A factorized action distribution predicts several button/axis choices rather than enumerating every possible combination. Its simplification can make coordinated combinations hard to discover; diagnose that before increasing network size. Test that each legal binding remains representable. Keep menus available but do not count leaving gameplay as success. Pausing must not freeze the experiment's deadline indefinitely.

Start at one decision per simulation frame. Holding an action for 2-4 frames may help exploration or throughput later, but fixed four-frame repeats would remove some frame-perfect possibilities. If introduced, retain one-frame actions and account for the actual elapsed frames in rewards and discounting. Changing action duration is an experimental condition.

### 2.4 Reward: reliability first, then speed

Use an episodic objective, not an informal reward-per-wall-clock-second ratio. Training speed and in-game performance are different quantities.

Proposed initial room objective:

- Completion: +1, exactly once on the verified forward room transition.
- Death or failure at a 30-second simulation deadline: -1.
- Time: -0.001 per simulated second, approximately -0.000016667 per ordinary frame.
- Progress shaping: a bounded potential difference, initially on a scale of 0.2.

These are starting values, not empirically tuned constants. At the 30-second limit, the direct time cost is 0.03, far smaller than the 2-point success/failure difference. No positive survival reward. Do not assert that this guarantees no stalling: exploration or shaping errors can still create bad behavior.

A **potential** is a score estimating progress from the current state. Add `gamma * potential(next) - potential(current)` to the base reward, with correct terminal handling. For an initial finite room task, consider `gamma = 1`, meaning future rewards in the attempt are not discounted. Then the shaping terms cancel across a complete trajectory when terminal potential is zero. This helps avoid rewarding repeated loops. It preserves the underlying objective under the assumptions of potential-based shaping; function approximation and incomplete observations still make learning difficult. [Reward shaping theory](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

Do not inherit PPO's common `gamma = 0.99` without thinking: over 600 one-frame steps it multiplies a delayed reward by about 0.0024. That is a substantial preference for early outcomes. If discounting later improves learning, document the changed objective and use the same gamma in shaping.

For room 1, a small hand-checked potential based on reachable platforms is acceptable. For expansion, generate candidate platform connectivity from collision geometry, test reachability with real legal action rollouts, and compute graph distance to the training target for reward only. A rough geometric graph is not a faithful Celeste motion planner: dash resources, one-way surfaces, momentum, and moving hazards can invalidate it. Budget 10-25 hours to build and validate a semi-automatic generator, plus sampled review and exception handling. Do not promise zero room-specific work.

An alternative is an archive of states actually reached by the agent. It avoids requiring a demonstration but introduces an exploration mechanism that must be named in the experiment. If shaped progress grows while completion stays flat, inspect trajectories before adding more reward.

Initially rank checkpoints by completion rate, then by successful-attempt time. Increase time cost only after the held-out entry-state test reaches the reliability target: try 0.005 and then 0.01 per simulated second, retaining a reliable reference policy and rejecting changes that fall below the target. A weighted sum cannot enforce reliability perfectly; the evaluation constraint does the final selection.

Death ends an individual room attempt. Measure successful clear times separately from failed-attempt durations; do not report failed attempts as fast clears. Also report deaths, success rate, and expected total effort to obtain a clear. Continuous chapter evaluation includes all deaths, respawns, transitions, and pauses, and reports both real elapsed time and the game's timer with their definitions.

### 2.5 Demonstrations, only after the declared baseline

Declare a no-demonstration campaign of **three training seeds, two million one-frame transitions per seed**, six million total. A seed controls learning randomness. This is an experiment budget, not a forecast that PPO will succeed. Allow at most 200,000 additional transitions for instrumentation smoke tests, counted separately. Freeze observation, reward, and action versions before the campaign. Abort invalid runs for concrete bugs, preserve their logs, and explain any restart.

At 0.25M and 1M steps per seed, inspect completion and state coverage, but do not quietly keep extending the budget. If there are no clears after the campaign, preserve that result and introduce demonstrations. If clears exist but are unreliable, finish the same budget and compare a demonstration-assisted branch rather than claiming demonstrations were necessary.

Start with 5-20 legal room-1 demonstrations, varied in timing and recovery if possible. **Behavioral cloning** is supervised learning from observation/action pairs, closely related to your sine training loop except the target is an action. Validate a cloned policy alone, then fine-tune with PPO. A TAS file supplies actions; replay it through the same instrumentation to record aligned observations. Do not put a planned future action or route into the observations. Split evaluation by whole trajectories or entry states, not adjacent frames from one recording.

Cloning can fail after small mistakes because it encounters states absent from its demonstrations. Collect corrective examples or broaden training starts. Account separately for demonstration collection, expert information, and RL steps. Describe the result as **demonstration-initialized, state-based RL**. For the no-demo condition, say **RL from scratch with engineered rewards**, if that is what was used. A TAS replay alone is not a learned closed-loop policy.

## 3. Throughput requirements derived from the experiment

No source establishes the number of steps this exact observation/action/reward setup needs. Use roughly **1-10 million transitions per seed as a planning envelope**, with the smaller, explicit two-million-step pilot budget above. Millions of transitions are not millions of attempts: at 300 transitions per attempt, two million transitions are about 6,667 attempts.

For N transitions, average step time s, average reset time r, mean attempt length L, and learning-update time U:

`total_seconds ~= N * (s + r/L) + U`

For several workers, measure aggregate throughput; do not multiply the one-worker rate by the process count without testing contention.

| Sustained aggregate transitions/second, including resets and learning | Time for the six-million-transition campaign |
|---:|---:|
| 30 | 55.6 hours |
| 60 | 27.8 hours |
| 200 | 8.3 hours |
| 1,000 | 1.7 hours |

These are arithmetic scenarios, not hardware predictions. At 20 training hours/day, even 30 transitions/s means roughly three days for the pilot. Human-speed execution is inconvenient, but it is not automatically fatal to a first-room experiment.

**Initial engineering target:** at least 30 sustained transitions/s for the honest pilot, preferably 100-200 or more. At mean L=300, target reset p50 <=30 ms, p99 <=100 ms, and mean <=50 ms. These thresholds are provisional service targets, not physics requirements. At 30 steps/s, a 50 ms average reset contributes only 0.17 ms per transition. At L=5, the same reset adds 10 ms per transition and becomes significant.

One correction to your answers: a p99 ten times the median does not by itself halve throughput. If 99% of resets take 50 ms and 1% take 500 ms, mean reset time is 54.5 ms. Record mean, p50, p99, maximum, timeout count, total reset time, and episode lengths. The entire distribution and reset frequency determine the impact.

Before training, benchmark at least 1,000 resets, 10,000 action/observation steps, and a sustained 30-minute loop, followed by a longer soak before unattended runs. Use a genuinely controllable post-intro state. Test normal, unfocused, minimized, and headless configurations separately. Include observation decoding, changing actions, policy-shaped inference cost, and actual restore completion. Measure dynamic hazard restoration later; identical player coordinates alone do not prove full-state restoration.

**Fallback order:** existing HTTP controls; a small synchronized local adapter; modest process parallelism; renderer simplification if behavior-equivalent; headless only after the observed crash is resolved and traces match. If no setup sustains 30 transitions/s, calculate a revised campaign duration from the measured rate. Do not silently replace the Steam game with Classic or a reimplemented simulator. Your revised reset constraint permits slower training if the experiment remains worthwhile.

## 4. Phased roadmap

Estimates are active hours including your learning, review, debugging, and write-up. Unattended training is additional wall-clock time. Expect overlap and uncertainty; these are planning ranges, not delivery promises.

### Phase 0: reproducible setup and feasibility, 12-25 hours

**Build:** initialize Git; ignore virtual environments, private saves, binaries, and bulky runs; preserve the existing exercises. Install a dedicated Everest/CelesteTAS/Speedrun Tool configuration with isolated saves. Pin versions and record hashes. Create the separate Python 3.12 environment. Keep scratch instrumentation outside the real package. Resolve the first-room identifier and reset state.

**Learn:** interpreter versus virtual environment; simulation time versus wall time; action/observation ordering. Read [Everest installation](https://everestapi.github.io/) and the [CelesteTAS controls](https://github.com/EverestAPI/CelesteTAS-EverestInterop/wiki/Controls).

**Done:** a version manifest, raw timing distributions, coherent state export, exact frame-count verification, legal input tests, repeated restore trace comparison, and a 30-minute stable loop. The report must explicitly distinguish rendered and headless results. No training code before this gate.

**Branch:** use existing interfaces if they pass. Otherwise authorize the minimal bridge implementation in the reviewed design. Allocate 8-20 additional hours if the adapter is needed. If basic fidelity cannot be demonstrated, stop and revise the project, rather than train on questionable physics.

**Portfolio:** a feasibility note showing the measured bottleneck, including the failed headless test and successful fallback.

### Phase 1: short RL teaching milestone, 6-10 hours

**Build:** CartPole with the selected PPO implementation, a readable walkthrough, and a saved/reloaded model. This occurs after the feasibility gate to honor your ordering constraint.

**Learn:** policy, return, critic, advantage, probability ratios, logarithms of probabilities, PPO clipping, and exploration entropy. Entropy describes how spread out the action probabilities are. Review just enough matrix multiplication and gradients to follow one minibatch update. Read [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/), [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), and selected [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) sections.

**Done:** three independent runs reach a predefined evaluation target, for example mean return >=475 across 20 fresh episodes, with reload preserving evaluation behavior. You can explain why the critic's target changes as experience is collected. This target is our teaching criterion, not a claim about the official benchmark threshold.

**Branch:** if this fails, debug the baseline and ending semantics, not Celeste. Cap this at days of active work; do not turn it into an implementation-from-scratch course.

### Phase 2: trustworthy Celeste environment, 15-30 hours

**Build:** a Gymnasium wrapper around the approved bridge, versioned observation/action schemas, one-frame stepping, reward components, and explicit episode boundaries. Cache static geometry; update dynamic state per step. Run legal scripted probes and a random policy only for validation.

**Learn:** the difference between **termination**, the task really ending, and **truncation**, data collection stopping before the task ends. A deliberately defined 30-second room deadline is task termination and needs elapsed-time information. An external collection interruption is different and may require the critic to estimate continuation. Read [handling time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) and [custom environments](https://gymnasium.farama.org/introduction/create_custom_env/).

**Done:** environment checks pass; recorded input sequences replay consistently; crouch dash and release/repress behavior work; success is a real forward transition; death and timeout are distinct; reset returns a ready state. Reward cannot enter policy inputs. Pause/journal cannot produce unlimited episodes. Compare normal and accelerated execution traces across movement, death, and transition probes.

**Branch:** missing dynamics require extra state fields or short history, not a room-specific network. Unexpected trace divergence blocks training until explained.

### Phase 3: room-1 campaign, 12-25 hours plus training

**Build:** PPO training, evaluation, atomic checkpoints, per-component rewards, episode summaries, seed/config manifests, and failure replay capture. Run the fixed six-million-transition campaign.

**Learn:** learning curves versus evaluation curves; variability between seeds; how a higher reward can coexist with worse gameplay. Read the [PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) selectively, especially rollout endings and advantage calculations.

**Done:** reproducible no-demo results for all three seeds, including failures. Reliability success means at least 99% observed success on 200 held-out, legally reachable entry states, with uncertainty reported. A deterministic policy on 200 identical savestates is one repeated trajectory, not 200 independent tests of robustness. Keep the canonical start result separate.

**Branch:** no clears at budget triggers the demonstration branch. Partial success triggers diagnosis of bottleneck states. Reliability achieved triggers speed experiments. A single lucky clear is a milestone, not completion of this phase.

**Portfolio:** first real gameplay clip and an experiment report with all seeds, failed runs, and a baseline comparison.

### Phase 3B: demonstration-assisted comparison if needed, 8-18 hours

**Build:** aligned state/action recordings, behavioral cloning, held-out trajectory evaluation, then PPO fine-tuning. Use the same evaluation protocol as Phase 3.

**Learn:** distribution shift, meaning the policy reaches states its training examples did not cover. Compare cloned-only, RL-only, and cloning-plus-RL.

**Done:** dataset provenance, quantities, split method, behavior checks, and a fair comparison at matched RL budgets. Report demonstration cost separately.

**Branch:** if cloning cannot follow a basic demonstration, investigate alignment, actions, and missing observations before more RL. If cloning works only from one exact state, broaden starts and corrective examples.

### Phase 4: several rooms with one policy, 20-40 hours

**Build:** a sampler over 3-5 representative rooms, mixing previously learned rooms into new training so the network does not forget them. Use the same weights and schema. Begin automating reward potentials and reached-state archives. Attempt a pair of naturally connected rooms early.

**Learn:** a **curriculum** changes the mix of training situations over time. **Catastrophic forgetting** means new learning destroys previous skills. Study [CoinRun](https://arxiv.org/abs/1812.02341) for evaluation design.

**Done:** one checkpoint solves the selected rooms; per-room tables reveal regressions; at least one two-room segment runs without an artificial reset. Held-out starts differ from those used for training and tuning.

**Branch:** if performance collapses when mixing rooms, adjust sampling/representation and evaluate memory before adding separate expert networks. If reward generation needs extensive manual annotation per room, address that before scaling to the chapter.

### Phase 5: Chapter 1 individual coverage and continuous completion, 30-70 hours

**Build:** expand the curriculum to all required route rooms. Train on entry states produced by the current upstream policy, including their speed, dash state, input holds, and hazard phases. Mix one-room, two-room, and longer segments. Keep some canonical resets to diagnose regressions.

**Learn:** an entry-state distribution is the range of situations in which the controller actually arrives. Multiplying isolated-room success rates is unreliable when failures and entries are dependent. Even with independence, 99% success over each of 20 rooms is only about 82% for a no-death sequence.

**Done:** a frozen policy completes Chapter 1 through normal transitions and normal death handling, with no savestate, teleport, altered physics, or speed control during the demonstration. Report completion over at least 20 complete attempts, deaths, total real time, and in-game time. Set a preliminary target of >=90% chapter completion within a declared deadline, then raise it. Do not claim a precise 99% chapter reliability estimate from 20 runs.

**Branch:** if isolated-room success is high but chapter success is poor, train from recorded natural entries and longer segments. A recurrent policy may be justified where local state/history remains ambiguous. Do not solve stitching by teleporting to memorized starting positions.

### Phase 6: main A-side story, roughly 80-200+ additional hours

**Build:** add chapters by mechanics, not just chapter number: moving hazards, wind, interactables, changing resources, and chapter-specific state. Maintain a regression suite and mixed-room training. Handle Prologue and non-platforming sequences with legal inputs, documenting any scripted menu/dialogue handling separately.

**Learn:** identify missing state from counterexamples; distinguish unseen-mechanic failure from ordinary tuning. Revisit [Go-Explore](https://arxiv.org/abs/2004.12919) only if reached-state exploration remains the limiting factor.

**Done:** chapter-level results and then a continuous Prologue-to-Summit run under the stated rules. If training has included every room, call this mastery of the trained campaign, not unseen-level generalization.

**Branch:** new mechanics may require new observation channels or memory. Version these changes and re-evaluate older chapters. Whole-game success is a research goal; the effort range has substantial upside risk.

### Phase 7: faster play, then optional harder content, open-ended

**Build:** optimize time subject to a reliability floor; study legal speed techniques; use richer demonstrations when justified; compare full time distributions and recovery behavior. Main story completion precedes Core, B/C-sides, and Farewell work. Extras may require normal collection/unlock progression or an explicitly disclosed evaluation save; do not silently grant access or collectibles in a claimed full progression run.

**Learn:** the fastest successful replay can be a very unreliable policy. Optimize expected complete-run performance as well as best time.

**Done:** repeated real-time runs improve speed without violating the selected reliability floor. Define a comparison category before using the phrase competitive: room/chapter/game, route, timing convention, deaths, and information/input privileges. A state-reading, frame-perfect agent is not a human leaderboard submission by default.

**Branch:** training may need recurrent memory, demonstration improvements, or exploration research. Competitive full-game times cannot be responsibly estimated before Chapter 1 evidence exists.

**Planning total:** approximately 95-200 active hours to a strong Chapter 1 result if major gates pass, plus optional adapter and imitation branches. At 10-15 hours/week, think roughly 2-5 months, with integration failures capable of extending that. A first useful room-level portfolio result should arrive much earlier. These ranges include documentation and learning, not just coding time.

## 5. Risks and evidence that reveals them

| Risk | Observable symptom | Response |
|---|---|---|
| Sparse reward | No clears, narrow state coverage, mostly identical deaths | Fixed campaign budget, bounded shaping, reached-state curriculum, then declared demonstrations. Do not assume random exploration can never clear the first room. |
| Shaping exploitation | Progress reward rises while clear rate stays flat; oscillation near a checkpoint | Plot reward components, inspect replay, use potential differences and terminal corrections, remove repeatable bonuses. |
| Premature death | Agent dies quickly to avoid further time penalties | Compare return of death, timeout, and plausible clears; keep failure penalty large and time small; address exploration. No scalar reward entirely eliminates local optima. |
| Menu/pause loopholes | Simulation timer stops while the process remains in an episode | Count decision frames including pause, add an external liveness timeout, and report the ending cause. Keep legal inputs but define the task deadline coherently. |
| Wrong success trigger | Agent gets completion reward by restarting, leaving a menu, or loading a level | Verify transition identity and forward destination once; checkpoint evaluator independent of reward code. |
| Partial reset | Same player coordinates, different hazard phase or held input | Replay identical action traces from restored states and compare all relevant fields/events, including buffers and randomness. |
| Observation race | State mixes two frames, disappears on death, or belongs to an earlier request | Frame IDs, paused/coherent reads, explicit terminal snapshots, watchdog and protocol tests. |
| Privileged information leakage | Removing reward metadata destroys behavior | Separate observation construction from reward/evaluation data; assert feature allowlists; log schema. |
| Room overfitting | Canonical starts succeed, natural entries fail | Reachable varied starts, mixed-room training, early two-room evaluation, and no room-specific network dependency. |
| Incorrect environment model | Reward graph says an impossible platform is reachable | Validate edges with actual rollouts; treat geometric reachability as a heuristic. |
| Headless or accelerated desynchronization | The same legal inputs produce different states | Matched trace tests and real-time rendered evaluation. The observed headless crash is an immediate compatibility item. |
| Long-run corruption | NaNs, growing RAM, repeated identical stale samples | Process health metrics, finite-value checks, bounded recovery retries, checkpoint rollback, and soak tests. |
| Weak statistical claims | One successful video, best seed only, many identical starts | Predeclared budgets, all-seed curves, fixed evaluation sets, confidence intervals, and explicit limits. |

A normal death is not proof that every relevant state variable resets identically. Chapters can contain persistent session flags and other state. Use measured reset equivalence, not that assumption.

## 6. Hardware, persistence, and cloud

### Local machines

Your 6 GB GPU is enough for the proposed small policy and state/grid observations in principle. The likely first bottleneck is the game/bridge CPU loop, not neural-network capacity. Benchmark CPU inference and training for small networks; GPU launch/transfer overhead can outweigh its benefit. SB3 specifically notes that non-convolutional PPO is often better run on CPU. [SB3 guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

Begin with one game process, then two, then four only if aggregate throughput rises and memory stays comfortable. With 16 GB system RAM, keep several GB free for Windows, PyCharm, and filesystem cache. Store rollout arrays on CPU, use compact grid encodings, and move minibatches to GPU if useful. Monitor laptop thermals, sustained clocks, RAM, VRAM, and disk growth. Do not assume sixteen CPU threads imply sixteen useful workers.

Use the desktop's 1-3 hour windows for independent seeds, short ablations, evaluation, or demonstration processing. Install its own pinned Python environment and game instrumentation. Prefer separate runs and transferring immutable artifacts over distributed on-policy training across intermittent machines. Test both GPU capability and numerical results rather than assuming the matching CUDA label guarantees compatibility.

In PyCharm, use named run configurations, the explicit project interpreter, breakpoints for short diagnostic runs, and profiling for bridge versus network time. Run overnight training outside the debugger. The documented SSH interpreter feature does **not** support a Windows remote host, so do not design the desktop workflow around that. Run locally on that desktop or use an independently configured remote-debug workflow later. [PyCharm SSH limitation](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html).

### Checkpoints and recovery

A durable training checkpoint should contain model and optimizer state, learning-rate schedule position, normalization statistics, global step/update counters, Python/NumPy/PyTorch random states, curriculum/archive version, and the complete configuration plus Git commit. Save at update boundaries every 5-10 minutes; write a temporary file and atomically rename it. Keep latest, previous, best-by-evaluation, and a small daily history.

On a game crash, terminate that worker, recreate its ready state, and discard its unfinished rollout. On learner restart, load a complete checkpoint and resume from fresh episodes. This is operational recovery, not bit-for-bit continuation of an in-memory game. Exact continuation would require additional environment state and validation, and is not necessary initially. Force a worker crash and a learner crash before trusting overnight runs.

Keep active runs outside a live-synced folder. Copy completed checkpoints and closed logs to your available Drive storage, with hashes and retention. Git tracks code/configuration; it is not the checkpoint store. Exclude game assets from the public repository and publish reconstruction/setup instructions instead. Preserve third-party licenses and attribution for reused code.

### Cloud decision: optional, zero essential spend

Google's current plans list monthly cloud credits, and its FAQ says credits can apply to Cloud products. This does not verify redemption, expiry, GPU quota, regional availability, or eligible SKUs on your account. Do not assume paid Colab access. [Plans](https://developers.google.com/program/plans-and-pricing), [benefits FAQ](https://developers.google.com/profile/help/benefits).

For scale, the listed T4 GPU-only price is $0.35/hour: $10 buys at most about 28.6 GPU-hours before CPU, RAM, disk, networking, and any Windows licensing. Region and configuration matter. This is not a quote for a complete usable Celeste VM. [GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en).

Google now documents spend-cap budgets for selected API services, but explicitly says persistent compute/storage charges continue; Compute Engine VMs are not a general hard-capped GPU-training solution through that feature. Alerts-only budgets also do not enforce a cap. Given your absolute $10 ceiling, use **no billable cloud training** in this roadmap. Optional cloud work is free-tier notebook analysis or a later, separately verified bounded offering. Your owned hardware is the default. [Spend-cap scope and limitations](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps).

## 7. Portfolio deliverables

Treat every milestone as a small publishable result, without overselling scope:

1. **Feasibility:** measured latency distributions, source findings, the headless failure, and the renderer fallback. A diagram of the actual game-to-Python boundary.
2. **Room 1:** 30-60 second authentic gameplay clip, observation/action specification, learning curves against steps and wall time, all seeds, and checkpoint reproduction instructions.
3. **Experimental comparison:** change one factor at a time, such as shaping, time penalty, or demonstration initialization. An **ablation** removes or changes one component to test its contribution. Use matched budgets and fixed evaluation sets. Keep exploratory results separate from held-out final evaluation.
4. **Chapter 1:** uncut continuous run, deaths and timing, per-room failures, natural-entry evaluation, and comparison with canonical-start results.
5. **Write-up:** hypothesis, protocol, outcome, uncertainty, failure interpretation, and what changed next. Include bad runs. Plot successful clear-time distributions alongside success rates so survivorship bias is visible.

For a data-science internship reviewer, the strongest story is that you formulated measurable hypotheses, built a trustworthy data-generating system, tested interventions, and explained why results changed. A failed no-demo campaign followed by a well-controlled improvement is useful evidence, not something to hide.

README opening claim should be exact, for example: "A learned state-based controller clears the first Forsaken City room with X/Y successes on held-out reachable starts, using vanilla gameplay physics and frame-level legal inputs." Disclose training resets, privileged observations, engineered rewards, and demonstrations if any. Do not use generated visuals as evidence of gameplay. Use real captures and reproducible measurements.

## Immediate next milestone after direction review

Complete Phase 0's synchronization/fidelity and sustained-throughput gate, decide whether existing controls suffice or a small adapter is warranted, then approve Phase 1 and the environment implementation. The successful reset microbenchmark lowers one risk; it does not substitute for the complete gate.
