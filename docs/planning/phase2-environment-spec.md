# Phase 2: Celeste environment specification (draft)

Status: draft for review. Nothing here is implemented yet. Once reviewed, the observation, action and reward definitions get version tags (`obs-v1`, `act-v1`, `rew-v1`) and are frozen before the Phase 3 campaign. Any later change gets a new tag.

Scope of v1: **Chapter 1, room `1`, from the canonical start** (TAS frame 300, player at (19, 144)), succeeding on the transition into room `2`. Varied, legally reachable entry states for held-out evaluation are Phase 3 work, but nothing below should assume a single start.

## 1. Architecture

```
policy  <-- observation (obs-v1) ---  CelesteRoomEnv  <-- Observation + extras --  LockstepBridge  <-- TCP -->  game
        --- action (act-v1)      -->                  --- input line          -->
                                       |
                                       +-- reward (rew-v1), ending cause: returned separately, never in the observation
```

| File | Role |
|---|---|
| `celeste_rl/actions.py` | `act-v1`: action vector to and from `(buttons, dash_only, move_only)`, then `format_input_line` |
| `celeste_rl/observation.py` | `obs-v1`: pure functions from one bridge `Observation` plus cached room geometry to arrays. Never receives reward or evaluation data |
| `celeste_rl/reward.py` | `rew-v1`: reward components and ending classification, kept outside observation construction |
| `celeste_rl/env.py` | `CelesteRoomEnv(gymnasium.Env)`: reset, step, episode boundaries, schema versions in `info` |
| `mod/CelesteRLLockstep` | Adds the player fields the current export lacks (section 3.2) |
| `tests/test_env.py` and friends | Offline tests with a fake bridge that replays recorded traces |
| `scripts/env_probes.py` | Live probes against the real game (section 7) |

The environment talks only to `LockstepBridge`. One environment owns one game process.

## 2. Actions (`act-v1`)

`MultiBinary(24)`: one independent on/off choice per input, in this fixed order.

| Index | TAS letter | Meaning |
|---|---|---|
| 0-3 | `L R U D` | movement directions |
| 4-5 | `J K` | jump (two bindings) |
| 6-7 | `X C` | dash (two bindings) |
| 8-9 | `Z V` | crouch dash (two bindings) |
| 10-11 | `G H` | grab (two bindings) |
| 12 | `S` | pause |
| 13 | `Q` | quick restart |
| 14 | `N` | journal / talk |
| 15 | `O` | confirm (second binding) |
| 16-19 | `A` + `L R U D` | dash-only directions |
| 20-23 | `M` + `L R U D` | move-only directions |

- One action is one game frame. No action repeat in v1.
- **Every vector maps to a legal input line, and every legal line maps to exactly one vector.** Dash-only or move-only with no direction set is simply omitted. Opposing directions (L and R together) are allowed and resolved by the game, as on a keyboard.
- Two bindings of the same action matter: holding one jump key and pressing the other re-presses jump without a release frame. Both stay.
- Press and release edges are learnable because the previous action is part of the observation (section 3.4).
- A factorised distribution (one Bernoulli per input) is the intended policy head. Its known weakness is coordinated combinations; diagnose that before changing it.

**Menu inputs (open decision D1).** `S`, `Q` and `N` open menus. Some pause-menu sequences make CelesteTAS stop the TAS, which ends the bridge session (section 4.4). The environment supports all 24 inputs. A config option `disabled_inputs` forces chosen indices to 0, reported in `info` and run manifests. Whether the Phase 3 campaign uses all 24 is decided after probe P7 measures how often random play hits a session-ending sequence.

## 3. Observations (`obs-v1`)

`Dict` space. All floats are `float32` and roughly in [-1, 1] or [0, 1]; scales are fixed constants in the schema, not learned statistics.

### 3.1 `player` (vector): from the existing export

| Feature | Encoding |
|---|---|
| Position in the room | `(X - Bounds.X) / Bounds.W`, `(Y - Bounds.Y) / Bounds.H` |
| Subpixel remainder | `PositionRemainder.X`, `.Y` (already in [-0.5, 0.5]) |
| Speed | `Speed.X / 400`, `Speed.Y / 400` |
| Flags | `OnGround`, `IsHolding`, `AutoJump` |
| Timers | `JumpTimer` (var jump) in frames / 12, `MaxFall / 320` |
| Movement state | one-hot over the vanilla player state names (`StNormal`, `StClimb`, `StDash`, ...), plus one "other" slot |

### 3.2 `player` (vector): new fields exported by the mod

The CelesteTAS game state lacks resources and timers that decide what an input will do. The mod adds a `player_extra` object to every reply, next to `state` (so `state` stays comparable with the HTTP bridge). Read-only; no game value is written.

| Field | Why |
|---|---|
| `Dashes`, `MaxDashes` | whether a dash is available |
| `Stamina` | climbing and grabbing run out |
| `Facing` | dash and climb direction without a held direction |
| `dashCooldownTimer`, `dashRefillCooldownTimer`, `dashAttackTimer` | dash timing |
| `jumpGraceTimer` | coyote time |
| `varJumpSpeed` | with `JumpTimer`, how a held jump continues |
| `wallSlideTimer`, `wallSpeedRetentionTimer`, `wallSpeedRetained` | wall interactions |
| `forceMoveX`, `forceMoveXTimer`, `climbNoMoveTimer` | frames where input is overridden |
| `InControl`, `Dead` | whether input does anything; death |
| `LiftBoost` | moving platform momentum |
| Buffered presses: jump, dash, crouch dash, grab (`bufferCounter`) | an input pressed a few frames ago can still fire |

Timers are encoded in frames divided by a fixed maximum. The final list is confirmed by reading `Player` in the pinned game build; fields that do not exist there are dropped, not faked.

### 3.3 `grid` (image): player-centred local geometry

`uint8` array of shape `(C, 32, 32)`: 32 x 32 cells of 8 px (one tile), centred on the player's tile. Channels:

| Channel | Source |
|---|---|
| solid | `SolidsData` tiles that are not `0`, plus `StaticSolids` rectangles |
| jump-through (up, left, right, down) | `JumpThrus` with direction |
| spikes (up, down, left, right) | `Spikes` with direction |
| other hazards | `Spinners`, `Lightning` |
| outside room | cells beyond `Level.Bounds` (edges are where exits and pits are) |

- A cell is 1 if any part of the object overlaps it. Exact sub-tile positions are in the `player` vector, not the grid.
- Static geometry (`SolidsData`) is parsed once per room and cached; entity lists are re-read every step.
- Room exits and pits are not marked. The policy sees walls, hazards and room edges, never which edge is the goal.
- Validation: across random rollouts the player hitbox never overlaps a solid cell (probe P8), and the grid for the start frame is checked by hand against the room.

### 3.4 `context` (vector)

| Feature | Why |
|---|---|
| Previous action (24 values) | held buttons, so press/release edges are learnable |
| Elapsed frames / deadline | the deadline ends the task, so the policy may know how much time is left |

### 3.5 Deliberately excluded

Room name or ID, target exit or direction, distance to the goal, reward or its components, potential value, checkpoints, future hazard positions, recorded actions, and all bridge `diagnostics` and transport metadata. A test asserts the exact set of observation keys and feature names. History beyond the previous action is not included in v1; add a short frame stack only if failures show it is needed.

## 4. Episodes

### 4.1 Reset

`bridge.reset()`, then check: room `1`, player present, not transitional, frame 300. Otherwise raise (section 4.4). `info` contains the schema versions, the start frame and the disabled inputs.

### 4.2 Step and ending causes

Each step sends one input line and receives the resulting frame. The step counter includes every frame: freeze frames, paused frames and frames spent in menus.

| Cause | Condition | `terminated` | Base reward |
|---|---|---|---|
| `success` | `RoomName` becomes the target room (`2`) with a player present | True | +1 (exactly once) |
| `death` | no player in a Level scene, or `Dead` true | True | -1 |
| `timeout` | step counter reaches the deadline (1,800 frames, 30 s) | True | -1 |
| `left_level` | a non-Level scene or a transitional observation after the start (for example save and quit) | True | -1 |
| `wrong_room` | any other room change | True | -1 |

- The deadline is part of the task, so hitting it is **termination, not truncation**; the time left is in the observation. Pausing cannot extend an episode because paused frames still count.
- `truncated` is only ever set by an outside caller stopping collection. The environment itself never truncates.
- Death is decided on the first frame with no player, the same boundary the death fixture already records.
- `info["ending"]` names the cause; reward components are in `info["reward_components"]`.

### 4.3 Things to confirm with probes

Quick restart (`Q`) and the pause menu's retry count as deaths in the game; check which cause they produce here. Check that a death during a freeze frame or on the deadline frame is classified as `death` (a real ending beats the deadline, as in Phase 1).

### 4.4 Bridge failures are not outcomes

A timeout, dropped connection, error reply or ended TAS session raises `BridgeFault`. The episode is discarded: no reward, no terminal transition. The caller decides how to recover. The training loop's worker restart policy is Phase 3 work, but the environment never turns a fault into an ordinary step, and a failed step is never retried.

## 5. Reward (`rew-v1`)

```
reward = completion (+1 once) + failure (-1 on death, timeout, left_level, wrong_room)
       + time (-0.001 per simulated second = -1/60,000 per frame)
       + shaping (scale * (gamma * potential(next) - potential(current)), potential of a terminal state = 0)
```

- These are the roadmap's starting values, not tuned constants. There is no survival bonus.
- `gamma` for shaping is an environment setting that must equal the learning algorithm's gamma (open decision D2: `1.0` for the finite room task, as the roadmap suggests, or the PPO default `0.99`).
- **Potential (open decision D3).** v1 ships with shaping off (potential always 0) so every probe and test checks the unshaped task first. A hand-checked room 1 potential based on reachable platforms is added as a separate, versioned component before Phase 3, and is kept out of observation code.
- Every component is reported separately in `info` so learning curves can be split by component.

## 6. Offline tests (no game)

- `gymnasium.utils.env_checker.check_env` against the environment with a fake bridge.
- Action mapping: every single input and random combinations round-trip, and every produced line matches the mod's input regex.
- Observation: shapes, dtypes and bounds; exact key and feature allowlist; grid for the recorded start frame matches a checked-in expected grid.
- Endings and rewards on recorded traces: exit route gives `success` at the recorded transition step with +1 exactly once; death route gives `death` on the first no-player frame; an idle trace gives `timeout` at 1,800; rewards sum as specified.
- A bridge fault raises and produces no transition.

## 7. Live probes (`scripts/env_probes.py`)

| ID | Probe | Pass |
|---|---|---|
| P1 | Exit route through the environment | `success` once, at the recorded step |
| P2 | Death route | `death` at the first no-player frame |
| P3 | Neutral inputs | `timeout` at 1,800 |
| P4 | Pause spam | still `timeout` at 1,800 |
| P5 | Crouch dash; jump held versus released and re-pressed; alternate jump binding re-press | expected state differences |
| P6 | Same action sequence twice | identical observations |
| P7 | Random policy, 1,000 episodes, all 24 inputs | only classified endings; count `BridgeFault`s and ending causes |
| P8 | Random policy | no NaN; observations within bounds; player hitbox never overlaps a solid cell |
| P9 | Throughput | environment steps/s including observation encoding, versus raw bridge |
| P10 | Actions from P1 and P7 played back as plain TAS | states match the environment's (existing export comparison) |

## 8. Build order

1. Mod `player_extra` export, rebuilt and checked against the existing equivalence checks.
2. `actions.py` and `observation.py` with offline tests.
3. `reward.py` and `env.py` with the fake bridge and `check_env`.
4. Live probes P1 to P10, results saved under `runs/env-probes/`.

Each step is committed separately.

## 9. Open decisions

| ID | Decision | Proposed default |
|---|---|---|
| D1 | Menu inputs (`S`, `Q`, `N`) in the Phase 3 action space | Support all 24; decide after P7 |
| D2 | Gamma for the room task | Decide with the Phase 3 PPO settings; the environment takes it as a setting |
| D3 | Room 1 potential for shaping | Off in v1; add a hand-checked version before Phase 3 |
| D4 | Frame stack | None in v1; previous action only |
