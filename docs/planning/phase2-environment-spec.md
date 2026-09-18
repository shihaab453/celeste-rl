# Phase 2: Celeste environment specification (v2)

Status: all five implementation steps done, and live probes P1 to P11 pass (section 11.1). The version tags `obs-v1`, `act-v1` and `rew-v1` are assigned in code (`celeste_rl/schema.py`, `celeste_rl/reward.py`) and are frozen before the Phase 3 campaign. Any later change gets a new tag.

Changes from the first draft: ending detection moves from state snapshots to game events captured by the mod (restarts hidden by loading were not detectable before); the observation gains control, collider and wall-boost state, a four-step history, exact units and a defined terminal encoding; the action mapping claim is narrowed to canonical input lines; the grid gets exact coordinates and an independent collision check; Stable-Baselines3 integration and bridge-fault recovery are specified; open decisions are decided; probes get pass thresholds.

## 1. Scope

- **Task:** Chapter 1, room `1`, from the canonical start (TAS frame 300, player at (19, 144)), succeeding on the natural transition into room `2`.
- **`obs-v1` is a room-1 schema.** It is designed not to depend on room 1's layout, but it does not promise that later rooms work without changes. Mechanics outside room 1 (refills, moving and falling blocks, holdables, wind, larger rooms) get a new observation version before the agent trains on them (section 12).
- Varied, legally reachable entry states for held-out evaluation are Phase 3 work. Nothing here assumes a single start, but reset validation for them needs its own configuration (section 6.1).

## 2. Architecture

```
policy <-- observation (obs-v1) --  CelesteRoomEnv  <-- state + extras + events --  LockstepBridge <-- TCP --> game + mod
       --- action (act-v1)      -->                 --- canonical input line   -->
                                        |
                                        +-- reward (rew-v1), ending cause, events: returned in info, never in the observation
```

| File | Role |
|---|---|
| `celeste_rl/schema.py` | Single source of truth for `obs-v1` and `act-v1`: feature table (name, source, unit, scale, bounds), channel indices, action order, schema fingerprint |
| `celeste_rl/actions.py` | Action vector to canonical input line, disabled inputs, applied action |
| `celeste_rl/observation.py` | Pure functions from one reply (state, extras) plus cached room geometry to arrays. Never receives events, reward or evaluation data |
| `celeste_rl/endings.py` | Ending classification from events and state |
| `celeste_rl/reward.py` | `rew-v1` and `rew-v2` components |
| `celeste_rl/potential.py` | the progress potential `rew-v2` shapes with |
| `celeste_rl/env.py` | `CelesteRoomEnv(gymnasium.Env)` |
| `celeste_rl/training/` (Phase 3, smoke-tested in Phase 2) | SB3 feature extractor and the rollout supervisor (section 9) |
| `mod/CelesteRLLockstep` | Adds `extras`, `events` and a validation-only collision query (section 4) |
| `scripts/env_probes.py` | Live probes (section 11) |

The environment requires the lockstep bridge. The HTTP bridge has no extras or events; it remains a reference for `state` only.

## 3. Actions (`act-v1`)

`MultiBinary(24)`, one on/off choice per input, one action per game frame, no action repeat.

| Index | TAS letter | Meaning |
|---|---|---|
| 0-3 | `L R U D` | movement directions (separate keys) |
| 4-5 | `J K` | jump (two bindings) |
| 6-7 | `X C` | dash (two bindings) |
| 8-9 | `Z V` | crouch dash (two bindings) |
| 10-11 | `G H` | grab (two bindings) |
| 12 | `S` | pause |
| 13 | `Q` | quick restart |
| 14 | `N` | journal / talk |
| 15 | `O` | confirm (second binding) |
| 16-19 | `A` + `L R U D` | dash-only directions |
| 20-23 | `M` + `L R U D` | move-only directions (separate keys) |

### 3.1 Canonical lines

- Every vector maps to exactly one **canonical** input line: letters in the fixed order above, each at most once, `A` or `M` omitted when none of its directions is set. Different vectors give different canonical lines. The bridge only ever sends canonical lines.
- The mod's regex also accepts non-canonical text (repeated or reordered letters). Those are not part of `act-v1`; a parser for recorded lines must canonicalise or reject them.
- A unique line does not mean a unique effect. For example, holding both jump bindings behaves like holding one, except for press edges (see 3.2).

### 3.2 How inputs reach the game

- Movement (`L R U D`), move-only (`M`) directions and all buttons are fed as separate keys. Opposing movement keys are allowed and resolved by the game.
- **Dash-only directions are not four independent keys.** CelesteTAS resolves them into one dash aim: left wins over right and down wins over up, before the game sees them. So pressing the opposite dash-only direction while holding one does not create a new key event. Confirmed by probe P5. The game reads the dash aim when the dash begins, after its opening freeze (4 frames after the press), so the direction held then decides the dash, not the one held on the press frame.
- Alternate bindings (`J`/`K`, `X`/`C`, `Z`/`V`, `G`/`H`) have separate press edges, so holding one jump key and pressing the other re-presses jump. Both stay.
- Talk and Cancel can only be pressed together with Dash or Journal, and analog stick angles are excluded. Both are CelesteTAS limits, stated as scope.
- Physical binding set: the one exercised by `scripts/binding_audit.py`.

### 3.3 Disabled inputs and the applied action

- A run setting `disabled_inputs` forces chosen indices to 0 before the line is built. It is recorded in `info` and the run manifest and cannot change during a run.
- **Decision D1:** Phase 3 training starts with `S`, `Q` and `N` disabled (indices 12, 13, 14). All 24 inputs remain supported and are probed (P7). Menu inputs are enabled for training only after menu-caused endings are proven to be classified as failures, never as bridge faults.
- The policy still outputs 24 values. Disabled values are ignored by the environment; this is a declared action transformation. `info["applied_action"]` holds what was actually sent, and the observation's action history uses the applied action.

## 4. Mod additions

All additions are read-only: no game value is written, no input is consumed. A noninterference probe (P11) compares runs with the additions enabled and disabled.

### 4.1 `extras` (on every reply)

Sampled at the same boundary as `state`. `null` when there is no Level or no player. Every member is resolved when the mod loads; if any is missing, the mod refuses all commands with an incompatibility error rather than sending a shorter object.

| Group | Members (type in the pinned build) |
|---|---|
| Resources | `Dashes` (int), `MaxDashes` (int property), `Stamina` (float) |
| Body | `Facing` (enum, sent as -1 or 1), `Ducking` (bool), collider and hurtbox rectangles relative to `Position` |
| Dash | `dashCooldownTimer`, `dashRefillCooldownTimer`, `dashAttackTimer` (float seconds), `DashDir` (vector) |
| Jump | `jumpGraceTimer`, `varJumpSpeed`, `AutoJumpTimer` (float) |
| Wall | `wallSlideTimer`, `wallSpeedRetentionTimer`, `wallSpeedRetained`, `wallBoostTimer` (float), `wallBoostDir` (int) |
| Forced movement | `forceMoveX` (int), `forceMoveXTimer`, `climbNoMoveTimer` (float) |
| Other | `LiftBoost` (vector property), `InControl`, `Dead`, `JustRespawned` (bool) |
| Input buffers | private `bufferCounter` (float seconds) of `Input.Jump`, `Input.Dash`, `Input.CrouchDash`, read by field. Never through `Pressed` or `Check`, which can consume the buffer. A buffered press survives only while its button stays held: releasing it clears the buffer (probe P5). Grab has no buffer (its buffer time is zero); its held state is in the action history |
| Level | `Level.Paused`, `Level.InCutscene` (bool), `Engine.FreezeTimer` (float seconds) |

The final field list is fixed in implementation step 1 by reading the pinned `Celeste.dll`, and recorded with the assembly hash.

### 4.2 `events` (on every reply)

The mod subscribes to Everest events and latches them from the moment a command is taken until its reply is sent, **including every automatic loading update in between**. Each reply carries the list, in order, and the latch is cleared. Events from a reset are reported on the reset reply and never count toward the new episode.

| Event | Everest source | Payload |
|---|---|---|
| `death` | `Events.Player.OnDie` (fires only for an accepted death) | room |
| `transition` | `Events.Level.OnTransitionTo` | from room, to room, direction |
| `load_level` | `Events.Level.OnLoadLevel` (every room entry: transition, respawn, restart) | room, intro type, `isFromLoader` |
| `level_exit` | `Events.Level.OnExit` | exit mode |
| `pause`, `unpause` | `Events.Level.OnPause`, `OnUnpause` | none |

Events are for ending classification and logging only. They never enter the observation.

### 4.3 `query_solids` (validation only)

A command that takes world rectangles and returns, for each, whether it collides with any `Solid` in the current level, using the game's own collision check. Used only by probes (P8) as an independent collision oracle. The environment never calls it.

## 5. Observations (`obs-v1`)

A `Dict` with five entries. Arrays are returned as fresh copies so stored samples are never mutated later.

| Key | Space | Content |
|---|---|---|
| `player` | `Box(float32, (4, F))` | player and control features (5.1) for the current decision and the previous three |
| `actions` | `MultiBinary((4, 24))` as `int8` | applied actions of the previous four decisions |
| `history_valid` | `MultiBinary(4)` | 1 where a history row is real, 0 where it is reset padding |
| `grid` | `Box(uint8, (11, 32, 32), 0 to 1)` | local geometry around the current player position (5.2) |
| `context` | `Box(float32, (2,))` | elapsed decision frames / 1,800; `player_present` |

**Decision D4:** four decisions of dynamic history for `player` and `actions`. Static geometry is not stacked.

### 5.1 `player` features

Timers are in seconds as exported and divided by a scale; they are not converted to frames. Positions and speeds are scaled but **never clipped**; their Box bounds are unbounded and values outside the nominal range are logged. Flags and one-hots are 0 or 1. NaN or infinity in any input fails the step as a schema violation (a `BridgeFault`, section 7).

| Feature | Source | Scale (nominal range) |
|---|---|---|
| Position in room | `(X - Bounds.X) / Bounds.W`, `(Y - Bounds.Y) / Bounds.H` using exported bounds (room 1 is 320 x 180) | [0, 1], exceeds it at room edges |
| Subpixel remainder | `PositionRemainder` | [-0.5, 0.5] |
| Speed | `Speed / 400` px/s | about [-1, 1]; supers and lift boosts can exceed it |
| Lift boost | `LiftBoost / 250` | [-1, 1] |
| Flags | `OnGround`, `IsHolding`, `AutoJump`, `Ducking`, `InControl`, `Dead`, `JustRespawned`, `wallSpeedRetained != 0` | 0 or 1 |
| Resources | `Dashes / 2`, `MaxDashes / 2`, `Stamina / 110` | [0, 1] |
| Direction | `Facing` (-1 or 1), `DashDir` x and y, `wallBoostDir`, `forceMoveX` | [-1, 1] |
| Collider | collider and hurtbox height and top offset relative to `Position`, / 11 | [-1.5, 1] |
| Jump | `JumpTimer / 0.25`, `jumpGraceTimer / 0.1`, `varJumpSpeed / 160`, `AutoJumpTimer / 0.1`, `MaxFall / 320` | about [-1, 1] |
| Dash timers | `dashCooldownTimer / 0.2`, `dashRefillCooldownTimer / 0.1`, `dashAttackTimer / 0.3` | [0, 1] |
| Wall | `wallSlideTimer / 1.2`, `wallSpeedRetentionTimer / 0.06`, `wallSpeedRetained / 400`, `wallBoostTimer / 0.2` | about [-1, 1] |
| Forced movement | `forceMoveXTimer / 0.2`, `climbNoMoveTimer / 0.1` | [0, 1] |
| Input buffers | jump, dash, crouch dash `bufferCounter / 0.08`; a value of 0 or below means no buffered press and is encoded as 0 | [0, 1] |
| Control | `Level.Paused`, `Level.InCutscene`, `Engine.FreezeTimer / 0.1` | [0, 1] |
| Movement state | one-hot over the 26 vanilla state indices (`StNormal` = 0 to `StIntroThinkForABit` = 25), plus one "other" slot | 0 or 1 |

- Scales come from game constants (dash speed 240, super horizontal 260, 0.08 s input buffer, 0.2 s var jump and dash cooldown, 0.1 s dash refill cooldown, 0.2 s wall boost window, 110 stamina, 250 lift boost cap). They are checked against the pinned binary in step 1 and fixed in `schema.py`; the table there, not this page, is authoritative once versioned.
- Feature order, the state enumeration and every scale are part of the schema fingerprint. A checkpoint records the fingerprint and refuses to load with a different one.

### 5.2 `grid`

- **Cells:** 8 x 8 px, aligned with the room's tile grid. Cell (row, col) of the room covers world x in `[Bounds.X + 8*col, Bounds.X + 8*col + 8)` and y likewise, half-open. Coordinates are floored, so negative positions work.
- **Anchor:** the cell containing the centre of the player's collider is at crop index (16, 16); the crop covers rows and columns -16 to +15 around it.
- **Occupancy rule:** a channel cell is 1 if the object's rectangle overlaps any part of the cell (half-open). This is a conservative raster, not an exact collision map; sub-cell positions of the player are in `player`.

| Channel | Content | Source |
|---|---|---|
| 0 | solid | `SolidsData` characters other than `0` (foreground solid tiles), plus `StaticSolids` rectangles, re-read every step because they can move |
| 1-4 | jump-through facing up, down, left, right | `JumpThrus` (only the types CelesteTAS exports: `JumpthruPlatform`, `SidewaysJumpThru`, `UpsideDownJumpThru`) |
| 5-8 | spikes pointing up, down, left, right | `Spikes` with `Direction` (0 up, 1 down, 2 left, 3 right), shifted to the hitbox: CelesteTAS exports an entity's position with its collider's size, and the game's spike hitboxes are 3 px above the position for up spikes and 3 px left for left spikes (confirmed from the pinned binary) |
| 9 | other hazards | `Lightning` rectangles shifted 1 px right and down to the hitbox (`Hitbox(w - 2, h - 2, 1, 1)`); `Spinners` exported as positions, stamped as a 16 x 16 box around the position, which contains the spinner's colliders (a radius 6 circle and a 16 x 4 box) |
| 10 | outside the room | any part of the cell outside `Level.Bounds`. Room 1's tiles span 184 px but its bounds are 180 px, so the bottom tile row is partly outside |

- Outside-room is geometry information, not solid. Exits and pits are not marked.
- Static tile geometry is parsed once and cached by the key (room name, bounds, hash of `SolidsData`), so a changed tile map or bounds invalidates it.
- **No player:** `grid` is all zeros.

### 5.3 Reset and terminal encoding

- **Reset:** row 0 of `player` is the start state; rows 1 to 3 and all `actions` rows are zeros with `history_valid` = `[1, 0, 0, 0]`. At the canonical start this is exact: the prefix's last frame is neutral, so no input is held. Varied starts must restore their real held inputs or declare a neutralisation, not invent zero history.
- **No player** (death frame): row 0 of `player` is zeros, `player_present` is 0, `grid` is zeros, history rows shift normally. The observation stays inside its space.
- `step` after an episode has ended raises until `reset`.

### 5.4 Deliberately excluded

Room name or ID, target exit or direction, distance to the goal, reward or its components, potential values, checkpoints, future hazard positions, recorded actions, events, and bridge `diagnostics` (loading counts, heap sizes, transport ids). `Level.Paused`, `InCutscene` and `FreezeTimer` are included because they are game facts the player can see, exported separately from diagnostics. A test asserts the exact keys, shapes, feature names and schema fingerprint.

## 6. Episodes

### 6.1 Reset

`bridge.reset()`, then validate: frame 300, room `1`, player at (19, 144), `StNormal`, `InControl`, not paused, `FreezeTimer` 0, `Dashes` = `MaxDashes` = 1, no `null` extras. The prefix itself is checked by the mod (input count and savestate breakpoint) and the bridge (every start must equal the first). Any failure raises `BridgeFault`. Events on the reset reply are logged and discarded. `info` holds the schema versions and fingerprint, `disabled_inputs` and the start configuration name (`canonical`).

### 6.2 The clock

The step counter counts **acknowledged decision frames**: one per `step`, each consuming one input. Freeze frames and paused frames count. Automatic loading updates inside one step do not add steps, and wall-clock time is never used. The deadline is 1,800 decision frames (30 seconds of normal play).

### 6.3 Ending causes and precedence

Each step is classified from that reply's events, then its state, in this order:

| Order | Cause | Condition | `terminated` | Base reward |
|---|---|---|---|---|
| 1 | (fault) | protocol error, invalid reply, schema violation, or an unexpected TAS stop | no transition; `BridgeFault` is raised | none |
| 2 | `death` | a `death` event | True | -1 |
| 2 | `restart` | a `load_level` event whose intro type is not `Transition` (chapter restart, reload, respawn), wherever it falls relative to other events | True | -1 |
| 2 | `left_level` | a `level_exit` event with no such `load_level` (a pause-menu chapter restart sends both and counts as `restart`) | True | -1 |
| 3 | `success` | a `transition` event from room `1` to room `2` in this episode, with no order-2 event in the same reply | True | +1 |
| 4 | `wrong_room` | a `transition` event to any other room | True | -1 |
| 5 | `timeout` | the step counter reaches 1,800 | True | -1 |

- Death, restart and leaving always beat success in the same reply. A real ending on the deadline frame beats the timeout.
- If the reply shows no player and carries no `death`, `restart` or `left_level` event, that is a fault, not a death: absence alone is never relabelled. Unknown or malformed events are faults too.
- A `transition` counts as `success` only if it is the reply's single transition, from the start room to the target room; anything else is `wrong_room`.
- The deadline is part of the task, so hitting it is **termination**; elapsed time is observed. The environment never sets `truncated`. The end of a PPO rollout buffer is not an episode ending; the learner bootstraps across it.
- A pause event alone does not end the episode; paused frames still spend the deadline.
- With menu inputs enabled, pause-menu retry is expected to produce `death`, restart chapter `restart` or `left_level`, and save and quit `left_level`. Probe P7 confirms each. Sequences that make CelesteTAS stop the TAS are a known bridge limitation; while they exist, menu inputs stay disabled for training (D1).
- `info["ending"]` names the cause and `info["events"]` lists the step's events.
- After `left_level` (for example "return to map"), the savestate reset over the socket cannot work. The lockstep bridge sees the level exit with no level loaded after it and makes the next reset a full recovery. From the map, CelesteTAS disables the first playback of the episode file right after it starts, and a second playback succeeds; the HTTP reset plays it once more in exactly that case (at most two attempts).
- Observed on the fixtures after step 1: the exit route's `transition` (1 to 2) arrives on step 285, whose `state` still says room `1`, and its `load_level` (intro `Transition`) on step 286. The death route's `death` event arrives one step before the first frame without a player. The pause-menu restart arrives as `level_exit` (mode `Restart`) plus `load_level` (intro `Jump`, from the loader) on the step whose reply follows the loading. Resets carry no events.

## 7. Bridge faults

`BridgeFault` covers timeouts, dropped connections, an unreachable game (for example after a crash, while resetting over HTTP), error replies, invalid replies, schema violations and unexpected TAS stops. For the environment it means: **no sample for this step, and this episode cannot continue.** The environment never returns a fabricated transition, never retries a step, and refuses `step` until `reset`. Recovery belongs to the learner (section 9.2).

## 8. Reward (`rew-v1`)

```
reward = +1 on success                      (once)
       + -1 on death, restart, left_level, wrong_room or timeout
       + -1/60,000 every step, including the final step   (-0.001 per second of normal play)
       + shaping, off in v1
```

- **Decision D2:** gamma = 1.0 for this finite task, in both the learner and any shaping term. With 0.99, a success at frame 1,800 would score about the same as a timeout.
- **Decision D3:** shaping is off for all Phase 2 validation and for an unshaped baseline. A bounded potential-based term `scale * (potential(next) - potential(current))` with terminal potential 0 may be added before the Phase 3 campaign as a separately versioned, disclosed component. At gamma 1 it sums to `-scale * potential(start)` over any complete episode, so it does not change which behaviour is best from a given start. A fault has no successor, so no shaping term is computed for it.
- **Decision D5, the time cost stays from the start** (owner constraint). Known trade-off: failing immediately scores about -1.00002 and timing out scores -1.03, so a policy certain to fail slightly prefers dying early. Trying for the full 30 seconds is still better whenever its success chance exceeds about 1.5%. This is accepted and disclosed, not claimed to be absent.
- Every component is reported separately in `info["reward_components"]`.

## 8.1 Reward (`rew-v2`), the shaped version

`rew-v1` above is frozen and stays selectable, so the unshaped baseline campaign remains reproducible. `rew-v2` is the separately versioned component decision D3 allowed for, and it is used from the first shaped Phase 3 experiment onwards. It makes two changes, which only work together:

```
reward = +1 on success                      (once)
       + -1 on death, restart, left_level, wrong_room or timeout
       + -(1,800 - t)/60,000 on those same failures at frame t   (the unspent deadline)
       + -1/60,000 every step, including the final step
       + scale * (potential(next) - potential(current)),  potential(terminal) = 0
```

- **The unspent-deadline charge.** Under `rew-v1` a failure at frame t costs `-1 - t/60,000`, so failing sooner pays slightly more (decision D5's known trade-off). Charging a failure for the deadline it did not use makes every failure total exactly **-1.03 whenever it happens**. The time cost still applies from the first frame, so D5 is kept, and there is no survival bonus: staying alive is worth nothing by itself, it only stops being cheaper to die early. The charge is reported as its own component, `unspent_deadline`.
- **Progress shaping.** The potential is the spike-weighted breadth-first tile distance to the room's exit (`celeste_rl/potential.py`): cost 1 per tile, plus 8 for a cell within 2 tiles of a spike and 1 per tile of open air below it, normalised so the exit is 1. It was chosen offline, before any training, by replaying a recorded clear of room 1 and a recorded death (`scripts/potential_check.py`): it rises along the real solution and falls on the way into the spike pit. The plain tile distance was rejected because it pays the agent to walk into that pit; a fall-weighted variant was rejected because it rates the pit floor above the start ledge.
- **The scale is a declared run parameter, not a constant of the reward.** Every run records the value it used in its manifest, the same way it records the learning rate or the entropy coefficient. The first runs used 0.2. Because a potential-based term telescopes to `-scale * potential(start)` over any complete episode, changing the scale changes how loud the progress signal is per step but never which ending is preferred, so it does not need a new reward tag. What the tag covers is the shape of the reward: its components and how each is computed.
- The potential is a function of the state alone, is computed outside the observation encoder, and never enters the observation. With gamma 1 and a terminal potential of 0, an episode's shaping sums to exactly `-scale * potential(start)`, a constant per start, so it cannot change which ending the agent prefers.
- Nothing here uses the recorded solution. A potential projected onto a recorded route is a demonstration method and belongs to Phase 3B.

## 9. Training integration (fixed now, built and smoke-tested in Phase 2)

### 9.1 Network input

Stable-Baselines3's default `Dict` handling would flatten the grid into 11,264 inputs, and its image network does not fit a 32 x 32 grid. Instead:

- `MultiInputPolicy` with a custom feature extractor: the grid, cast to float 0 or 1, through a small channels-first CNN (3 x 3 convolutions, 32 then 64 channels, stride 2, then a 128-unit linear layer); `player`, `actions`, `history_valid` and `context` flattened; all concatenated before two 128-unit layers for the policy and value heads.
- `normalize_images=False`, so the 0 or 1 grid is not divided by 255.
- The environment keeps `uint8` grids; the rollout buffer stores them as `uint8`.

### 9.2 Rollout supervisor

If any environment raises `BridgeFault` during rollout collection:

1. No sample is stored for the failing step.
2. **The whole current rollout is discarded**; no update uses it. Earlier completed updates are kept.
3. The failed game worker is restarted; every environment slot is reset; the rollout buffer, last observations and episode-start flags are replaced with fresh, validated ones.
4. Weights and optimizer state are those of the last completed update. Any statistics updated from the discarded rollout are restored.
5. Attempted, discarded and accepted transitions are counted separately. After 3 consecutive faults without a completed rollout the model (weights and optimizer state of the last update) is saved to the run's abort checkpoint path, training callbacks are ended, and `TrainingAborted` is raised.

Phase 3 starts with one environment. Implemented in `celeste_rl/training/supervisor.py` (`SupervisedPPO`) for a single-process `DummyVecEnv` without `VecNormalize`. A reset that faults during recovery is retried and counts toward the consecutive limit. The first reset inside `learn()` happens before collection and is not supervised. Offline tests inject faults mid-rollout, in episode-end resets and in recovery resets, and check that no update sees data from before a fault, weights and optimizer state are unchanged by discarded rollouts, the step budget counts accepted transitions only, and episodes finished inside a discarded rollout are not logged. Worker processes and their communication faults come with multi-worker training.

## 10. Offline tests (no game)

| Area | Checks |
|---|---|
| Environment | `gymnasium.utils.env_checker.check_env` with a fake bridge replaying recorded replies |
| Actions | all 24 single inputs; all 256 A x M direction combinations with buttons; seeded random vectors; each canonical line matches the mod regex; canonicalisation or rejection of non-canonical lines; disabled inputs and applied action |
| Observation | spaces, dtypes and shapes; exact key, feature and fingerprint allowlist; start-frame grid against a checked-in expected grid; grid anchor and flooring at negative coordinates and room edges; cache invalidation when tiles or bounds change; no-player encoding in space; returned arrays not aliased between steps |
| Endings | table-driven cases for every cause and every precedence conflict (death and success in one reply, death on frame 1,800, restart hidden by loading using the saved loading-route reply, no player without a death event raising a fault) |
| Reward | component sums per cause, including the time cost on the final step |
| Faults | a fault produces no transition and blocks `step` until `reset` |
| Learner smoke test | SB3 PPO with the custom extractor and `MultiBinary` actions builds, runs a short fake rollout and update, handles terminal autoreset, and passes supervisor fault injection with no optimizer step on discarded data |

Target: zero failures.

## 11. Live probes (`scripts/env_probes.py`)

Results are saved under `runs/env-probes/<timestamp>/`.

| ID | Probe | Pass |
|---|---|---|
| P1 | Exit route through the environment, 10 repeats | `success` once each, at the recorded transition step, identical observations across repeats |
| P2 | Death route, 10 repeats | `death` on the reply carrying the death event, identical across repeats |
| P3 | Neutral inputs | `timeout` at exactly 1,800 |
| P4 | A known safe pause and unpause cycle repeated | `timeout` at exactly 1,800 |
| P5 | Named mechanics, each with expected values and frame windows written into the probe: crouch dash; jump held versus released and re-pressed; alternate jump binding re-press; buffered jump within 0.08 s; crouch collider change; dash-only opposing directions resolving left and down | every expected value matches |
| P6 | 100 resets, then 10 repeats of P1, P2 and a movement route, including 5 / 50 / 500 ms client delays | encoded observations and events identical; reset validation passes 100/100 |
| P7 | Random policy, 1,000 episodes with all 24 inputs, and 1,000 with S, Q, N disabled | every ending classified; with S, Q, N disabled zero faults; with them enabled, every fault reproduced and explained; menu-caused endings tallied by cause |
| P8 | Random policy trajectories checked against `query_solids` | zero NaN or schema violations; for sampled cells, every channel-0 cell whose tile or solid covers the whole cell collides and every empty cell does not; the player's real collider never overlaps a colliding rectangle |
| P9 | Throughput, matched runs | environment keeps at least 80% of raw bridge steps/s; encoding p95 under 200 microseconds; p50, p95, p99 and reset latency published |
| P10 | Actions from P1, P2 and P7 replayed as plain TAS playback with export | all common `state` fields match on every expected frame; extras compared between lockstep runs with and without client delay |
| P11 | Noninterference: P1, P2 and a random trace with `extras` and `events` enabled versus disabled | `state` identical on every frame |

Zero passing does not prove absence: 1,000 fault-free episodes bound the fault rate at about 0.3% per episode (95%).

### 11.1 Results

Full run `runs/env-probes/20260917-160721` and P11 run `runs/extras-noninterference/20260917-160652`, both from a clean checkout of commit `741c533` on the pinned runtime (`config/pinned_runtime.json`: Celeste 1.4.0.0-fna, Everest 6531 stable, CelesteTAS 3.47.1, Speedrun Tool 3.27.21, mod DLL built from that commit). The live scripts refuse uncommitted changes and runtime mismatches unless told otherwise, and record the runtime manifest in `results.json`. All passing:

| ID | Result |
|---|---|
| P1 | `success` at step 285 in 10/10 repeats, identical observations, return 0.99525 |
| P2 | `death` at step 74 in 10/10 repeats, identical observations |
| P3 | `timeout` at 1,800, return -1.03 |
| P4 | 30 pause and unpause cycles, `timeout` at 1,800 |
| P5 | jump speed -105; held jump apex 20 px higher; buffered jump fires 3 frames before landing only while held, not after release and not from 8 frames before; alternate jump binding re-jumps while the first is held; ducking collider 6 px; crouch dash ducks; dash-only left + right dashes left while facing right and up + down dashes down |
| P6 | 100 resets identical; exit, death and movement routes identical across 10 repeats with 5, 50 and 500 ms delays |
| P7 | menus disabled: 1,000/1,000 episodes classified (all deaths, median 70 frames), zero faults. All inputs: 987 classified (246 death, 674 restart, 67 left_level); 13 faults, all CelesteTAS stopping the TAS during pause-menu sequences mid-episode, 13/13 reproduced exactly on replay. Menu inputs stay disabled for training (D1) |
| P8 | 236 sampled frames, zero collision mismatches, zero schema violations |
| P9 | environment 1,463 to 1,500 steps/s versus raw bridge 1,807 to 1,851 (81.7%, target 80%); environment overhead p50 118, p95 172, p99 246 microseconds; reset p50 5.0 ms, p95 14.0 ms. The ratio's margin over 80% is within run-to-run noise (81.1% to 81.9% across the saved passing runs); the overhead p95 under 200 microseconds is the robust half of this probe |
| P10 | exit (285), death (74) and three random episodes (46, 77, 294 frames) match plain TAS playback on every expected player frame |
| P11 | state identical with extras and events on and off across exit, death, restart and 600 random frames |

## 12. Later observation versions

Not in `obs-v1`, required before the agent trains on the mechanic:

- refill availability, moving and falling block velocity and phase, crumble blocks, holdables, wind (`WindDirection` is already exported)
- a typed, masked entity list with capacity and overflow rules
- a coarse whole-room geometry view for rooms larger than the 256 px crop
- longer history or a recurrent policy, coordinated action heads, action repeat

Each gets a new version, a migration plan for checkpoints and re-evaluation.

## 13. Build order

1. **Mod:** `extras`, `events`, `query_solids`, startup member checks; `LockstepBridge` validation of the new objects; existing equivalence checks still pass; P11.
2. **Schema, actions, observation** with offline tests.
3. **Endings, reward, environment** with the fake bridge and `check_env`; version tags assigned.
4. **Training integration:** extractor and rollout supervisor with the learner smoke test.
5. **Live probes** P1 to P10.

Each step is committed separately.

## 14. Decisions

| ID | Decision | Status |
|---|---|---|
| D1 | Menu inputs | All 24 supported; Phase 3 starts with `S`, `Q`, `N` disabled until menu endings are proven |
| D2 | Gamma | 1.0 |
| D3 | Shaping | Off for validation and the unshaped baseline; any later potential is separately versioned |
| D4 | History | Four decisions of player features and applied actions |
| D5 | Time cost | Kept from the start; early-death trade-off disclosed |
