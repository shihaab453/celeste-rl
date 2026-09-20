# celeste-rl

Building a reinforcement learning agent that learns to play [Celeste](https://www.celestegame.com/) (Steam version) from game state, on unmodified game physics.

The long-term goal is an agent that clears the first room of Chapter 1 reliably, then every room, then a continuous Chapter 1 run, then the main story, and eventually plays at speedrun pace.

## Status

**Room 1 of Chapter 1 is solved by the demonstration-assisted, varied-start method.** On a replacement frozen
set of 200 unique states from 11 search-route clusters, six matched fine-tuning seeds achieved a 74.8%
route-macro success rate with varied starts and 52.2% with canonical starts. The predeclared paired difference
was **+22.6 percentage points**, with a crossed seed-route 95% bootstrap interval from +6.7 to +43.5 points and
an exact two-sided paired sign-flip p-value of 0.03125. Varied starts won in all six matched seeds and on all
eleven route-level descriptive comparisons.

The fixed canonical start remains a poor measure of generalisation: two policies that scored 98.0% and 98.5%
there scored 59.7% and 84.6% on the earlier diagnostic state set. The project now proceeds to Room 2. Chapter 1
has 20 rooms; Room 1 is the first.

Two experiment reports, and the negative one came first and matters as much:

| Report | Result |
|---|---|
| [Phase 3: without demonstrations](docs/phase3-unshaped-baseline.md) | **0 clears** in 50,369 episodes over three seeds and six million transitions, with the diagnosis of why |
| [Phase 3B: with demonstrations](docs/phase3b-demonstration-comparison.md) | **Varied-start fine-tuning beats canonical-start fine-tuning by 22.6 points** under the predeclared matched, route-cluster-aware analysis |

The project committed in advance to attempting the room without demonstrations first, on a fixed budget, so
that the result could be described honestly either way. That attempt failed, and the report says so and
explains what was measured to work out why: the policy never left near-uniform play, the critic learned the
shaped value function exactly, and the reward left very little raw advantage variation among failures. PPO
normalises those advantages, so this is evidence of weak signal rather than literally zero policy gradient.

Confidence intervals are reported where their sampling assumptions are defensible. The held-out comparison
resamples matched training seeds and complete route clusters rather than treating every prefix as independent.

### The interface underneath

No learning happens until the environment is correct, fast enough for millions of frames, and stable enough to
run unattended.

| What | Result |
|---|---|
| Step speed | **About 2,600 frames per second** including resets, up from about 30 per second over CelesteTAS's HTTP interface (about 85x) |
| Physics fidelity | Complete game state identical frame by frame to the reference interface, and position, speed, state, room and timer identical to normal-speed TAS playback on fixed traces covering a room transition with 7 dashes (376/376 frames), spike deaths and respawns (263/263) and a pause-menu loading scene (287/287) |
| Stability | 30-minute run: 4,853,400 steps, 16,178 episodes, 808 of 808 periodic replay checks identical, no errors |
| Robustness | 13 live fault checks pass, including 100 rounds of replacing a client connection mid-reset and a client that stops reading replies |
| Inputs | Input bindings checked in the game (30 of 30 checks), including both bindings for jump, dash, crouch dash and grab, dash-only and move-only directions, and pause-menu confirm and cancel |
| Memory | Found and fixed a render target leak of 2 to 6 MB per reset (8 GB in 4 minutes); memory now stays flat |
| Provenance | Every training run records its commit, the game build's file hashes and the schema version, and refuses to start if any of them drifts |
| Tests | 281 unit tests, plus live checks against the game |

## How it works

The game runs with [Everest](https://everestapi.github.io/) (mod loader), [CelesteTAS](https://github.com/EverestAPI/CelesteTAS-EverestInterop) (frame-exact input playback) and [Speedrun Tool](https://github.com/DemoJameson/Celeste.SpeedrunTool) (savestates), in an isolated copy of the game.

**Reference bridge (Python, HTTP).** CelesteTAS has no endpoint that accepts inputs, so `celeste_rl/bridge.py` writes inputs into the TAS file CelesteTAS is playing. The file always ends at the frame the game is paused on, and the bridge appends exactly one input line per step. CelesteTAS refuses to advance past the end of the file until it has re-read it, so an input can never land on the wrong frame or be played twice. This path is limited by the game's 60 Hz clock to about 30 steps per second, and is used as the correctness reference.

**Lockstep bridge (C# Everest mod + Python client).** `mod/CelesteRLLockstep` hooks CelesteTAS's playback loop. When every input has been played, it sends the resulting state to Python over a local socket and waits for the next input, instead of waiting for the next 60 Hz tick. Each frame still runs through CelesteTAS's normal input playback and the game's normal update with the normal frame length; only the waiting changes. `celeste_rl/lockstep.py` exposes the same `reset()` and `step()` as the reference bridge.

**Resets** restore a savestate taken one frame before the episode start, which takes about 4 ms.

**Session safety.** Each connection is owned by a numbered session: replies only go to the connection that sent the command, a new connection must reset before stepping, and every pending command has a deadline. Any timeout, malformed reply or error ends the Python session, and recovery always goes through a full reset.

**Loading scenes.** If an input starts a loading scene, the mod lets loading finish before replying, so every observation is a frame the agent can act on.

## Verification

- `scripts/transition_check.py` plays a fixed input trace four ways (reference bridge, lockstep bridge, lockstep with 5, 50 and 500 ms client delays, and plain TAS playback at normal speed exported frame by frame) and requires them to agree, with full coverage of every expected frame.
- `scripts/lockstep_fault_check.py` misbehaves on purpose against the running game: stepping before reset, wrong start frames, replacing connections at random points during a reset, disconnecting mid-step, and clients that never read replies.
- `scripts/binding_audit.py` checks that each input binding does what it should in the game.
- `scripts/soak_test.py` and `scripts/memory_probe.py` run long sessions and isolate memory behaviour. The memory probe found that Everest's dash-trail render targets were never freed when a savestate replaced the level, and the mod now releases them before each reset.

## Ground rules

- **Physics are never modified.** Savestates and fast stepping are training scaffolding and will not be used in any demonstrated run.
- **Legal inputs only.** The agent can use exactly the bindings in the game's controls menu. Input lines are validated in both Python and the mod, so no TAS command can be injected.
- **State-based observations.** The agent reads game state (position, velocity, room geometry and so on), not pixels. This will be documented precisely alongside results.
- **No game files** are included in this repository.

## Repository layout

| Path | Contents |
|---|---|
| `celeste_rl/` | Reference bridge, lockstep client, game launcher, export comparison |
| `mod/CelesteRLLockstep/` | The Everest mod (C#) |
| `scripts/` | Environment check, live verification, fault, binding, soak and memory checks, route search |
| `tests/` | Unit tests and fixed input traces |
| `docs/planning/` | Phased roadmap and the first feasibility measurements |
| `experiments/`, `results/` | Early PyTorch exercises (tensors, a hand-written training loop fitting `sin(x)`) |

## Setup

Requires Windows, the Steam version of Celeste, and a separate copy of the game with Everest 1.6531.0, CelesteTAS 3.47.1 and Speedrun Tool 3.27.21 installed. Game files are not included.

The RL environment uses Python 3.12.13 in `.venv-rl`, created with [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12.13
uv venv .venv-rl --python 3.12.13 --seed
uv pip sync requirements-rl.lock --python .venv-rl/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
.venv-rl/Scripts/python.exe scripts/check_env.py
```

Pinned dependencies: PyTorch 2.14.0 (CUDA 13.0), Gymnasium 1.3.0, Stable-Baselines3 2.9.0. `check_env.py` verifies a real GPU calculation and trains PPO on CartPole briefly as a smoke test. Run the unit tests with `.venv-rl/Scripts/python.exe -m unittest`.

Build the mod into the game copy with the .NET SDK:

```bash
cd mod/CelesteRLLockstep
dotnet build -c Release -p:GameDir=<path to the game copy> -p:RefsDir=<folder with the CelesteTAS DLLs>
```

`RefsDir` holds `CelesteTAS-EverestInterop.dll`, `StudioCommunication.dll` and `MemoryPack.Core.dll` from the `bin` folder of the CelesteTAS release. The build installs the mod into the game copy's `Mods` folder.

To change Python dependencies, edit `requirements-rl.in`, delete `requirements-rl.lock` (an existing lock file makes uv keep old versions), then regenerate it:

```bash
uv pip compile requirements-rl.in -o requirements-rl.lock --python-version 3.12 --python-platform windows --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --generate-hashes --emit-index-annotation
```

Do not use `uv sync` or `uv run` in this repository: uv's project mode manages a venv named `.venv` and would rebuild the separate Python 3.14 environment used by `experiments/`.

## Next

Phases 0 to 3B are done: the interface, a PPO implementation studied on CartPole, the environment, the
no-demonstration attempt and the demonstration-assisted comparison.

1. **Room 2.** Generalise task starts beyond the room 1 savestate, find independent Room 2 routes, then repeat
   cloning, varied-start fine-tuning and frozen held-out evaluation without tuning against the test set.
2. **Two-room retention.** Mix Rooms 1 and 2 during training and measure both separately, so learning Room 2
   cannot silently destroy Room 1 competence.
3. **How much of the demonstration set is needed.** The cloned Room 1 policy scores below a
   repeat-your-last-action baseline on held-out routes. Reducing the route count is useful only after Room 2
   establishes that the full method transfers.
4. **Natural two-room play.** Evaluate one frozen checkpoint across the Room 1 to Room 2 transition without an
   artificial reset, as Phase 4 requires.

The full plan is in [`docs/planning/roadmap.md`](docs/planning/roadmap.md).
