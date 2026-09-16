# Celeste feasibility evidence

Date: 16 September 2026. Scope: source inspection and throwaway measurements, with no training, environment implementation, or reward implementation.

## What was tested

An isolated copy of the user's Steam Celeste installation was created under `celeste-research-scratch/game-probe`. The official Everest installer was applied there. The original Steam installation was not modded. The scratch process used its own `EVEREST_SAVEPATH` and localhost DebugRC port 32279. No personal save files were copied into the probe.

Versions exercised:

- Celeste 1.4.0.0, converted by the official Everest installer from the installed XNA build to FNA.
- Everest stable 1.6531.0.
- CelesteTAS 3.47.1.
- Speedrun Tool 3.27.21.
- Python 3.14.6 for the standard-library-only measurement client.
- Existing PyTorch 2.14.0+cu130 was separately imported and reported CUDA available. No model was trained.

Source checkouts inspected:

- CelesteTAS: `fd1e2670e5bbdf58c3796efd84965c8e3b7c5614`.
- Everest development source: `47d6a61fa918c17b8085b93b61ee59a7465968f8`.
- Speedrun Tool: `a1bb8dc43fb675da4cf2c1b30e98265e828ad766`.

Development source inspection and installed release behavior are different evidence types. The live tests above use the listed released binaries.

## Live findings

### State access and room identity

`/tas/game_state` returned player position, subpixel remainder, velocity, grounded status, movement-state name, current-room geometry, spike bounds/directions, and other fields. Chapter 1's initial room was reported as `1`, in `Celeste/1-ForsakenCity`.

The benchmark's reset checkpoint was after 300 neutral TAS frames, past the introductory animation. The player was in `StNormal`, grounded, at (19, 144), with zero velocity. A 60-frame checkpoint had still been in `StIntroJump`, and was rejected as the ready-state benchmark definition.

The JSON schema is not a complete RL observation. In particular, the inspected player JSON does not include every proposed dash/stamina/timer field. Additional inspection interfaces exist, but their per-field completeness and coherent per-step collection need validation.

### Rendered benchmarks, successful

The Python client used the existing HTTP endpoints and a neutral TAS file. No custom C# was written.

For each one-frame advance, timing started before sending the frame-advance request and ended after observing the expected paused TAS frame and decoding fresh JSON game state. For each reset, timing started before sending Restart and ended after observing the paused checkpoint frame, decoding JSON, and verifying the player fields and room matched the saved start.

| Operation | Measured samples after warm-up | Mean | Median (p50) | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Advance one neutral frame, initial run | 200 | 32.64 ms | 32.70 ms | 35.12 ms | 36.78 ms |
| Restore room-start savestate, initial run | 100 | 26.89 ms | 24.18 ms | 51.00 ms | 51.51 ms |
| Advance one neutral frame, corrected longer run | 1,000 | 32.63 ms | 32.66 ms | 36.13 ms | 54.14 ms |
| Restore room-start savestate, corrected longer run | 1,000 | 25.99 ms | 24.01 ms | 51.97 ms | 56.30 ms |

Twenty warm-up samples were excluded for each operation in each run. Raw samples for the corrected longer run remain in `celeste-research-scratch/probe_results.json`; the initial run is preserved in `celeste-research-scratch/probe_results_short.json`. The percentile estimator selects an ordered sample. Even 1,000 resets cannot establish rare-failure reliability or replace a long soak test.

The inverse mean advance latency is approximately 30.6 operations/second. This is not training throughput: it excludes policy inference, learning updates, changing-action file writes, and some loop bookkeeping outside the timed section. It is also not the engine's maximum simulation speed.

### Headless attempt, failed

The official installer accepted the `headless` argument. The game started, loaded the mods, reached the room, and served state. After a savestate and subsequent stepping, the game raised `System.ArgumentOutOfRangeException` in `FMOD.Studio.EventDescription.getPath`, through Speedrun Tool's `MuteAudioUtils.EventDescriptionOnCreateInstance`.

The preserved log is `celeste-research-scratch/headless-crash.log`. This establishes a compatibility failure in the tested combination, not that headless Celeste is fundamentally impossible. No physics changes or source patches were made to bypass it. Reinstalling the scratch copy with the normal renderer allowed the short benchmark to complete.

### Longer probe and initialization race

The first attempt to extend the benchmark to 1,000 measured steps/resets timed out while expecting an advanced frame; the game was still paused at frame 300. The script had allowed a previous playback's matching frame number to satisfy the new playback's startup condition. That is an asynchronous initialization flaw in the probe, not evidence of a measured slow reset.

The script was changed to wait for previous playback to stop before starting the new one. The corrected run then completed 1,020 steps and 1,020 resets, leaving 1,000 measured samples of each after warm-up. Every reset matched the initial exported player fields and room. This successful run is reported separately above; it does not erase the initialization failure or prove restoration of unexported state. A production bridge needs request and episode IDs, not just matching frame numbers.

## What is demonstrated, and what remains open

**Demonstrated:** the laptop can expose state to Python, advance neutral frames, and restore a ready room-start savestate with low tens-of-milliseconds latency using existing tools. The original guessed 50 ms reset target is in the right range for this microbenchmark, although sustained sample throughput matters more.

**Not demonstrated by these benchmarks:** robust dynamic action injection, full hidden-state equivalence, moving-hazard restore fidelity, real-time versus accelerated physics equivalence, multiple-instance isolation, complete binding coverage, a long unattended run, or end-to-end PPO sample throughput. The tested headless stack does not pass.

There is enough evidence to justify continuing the game-interface feasibility phase. There is not enough evidence to declare the complete training interface finished or to promise a zero-C# production design.

## Required next checks before training

1. Extend the corrected benchmark to 10,000 action/observation steps with varied inputs; preserve failures as well as successful samples. The 1,000-reset neutral benchmark is complete.
2. Change physical inputs from Python between steps; verify the intended action is consumed exactly once at the intended frame.
3. Verify every required input binding, including crouch dash and press/release edges, against ordinary gameplay.
4. Compare deterministic short action traces after restore, including player state, buffers, timers, and dynamic entities, not just position.
5. Compare rendered normal-speed and accelerated execution using the same physical inputs. Treat headless as a separate optional configuration until its crash is resolved.
6. Run a 30-minute continuous workload, then a longer soak and deliberate process-crash recovery test before unattended learning.
7. Benchmark two workers only after save, port, file, and Studio-channel isolation are explicit.

Do not time only the HTTP acknowledgment. It can precede completion of the requested game operation.

## Side effects and scope

Only scratch files and research documents were authored. The real project at `C:\Projects\celeste-rl` was inspected, not modified. No Git repository was initialized there, no RL packages were installed, and no training or reward implementation was written.

The scratch game process was stopped after measurement, and its final log was preserved as `celeste-research-scratch/rendered-probe-final.log`. The isolated game copy, downloaded releases, inspected source checkouts, and measurement scripts remain in the scratch directory for reproducibility. The copied game assets consume several gigabytes; they are local test material and must not be committed or redistributed with the portfolio repository.

The first launch of the copied Steam executable redirected to the original Steam installation before the scratch app-ID file was added. The original game process was left alone. This means background workload was not a tightly controlled benchmark condition; the reported timings are exploratory laptop measurements, not a clean performance certification.

Official sources used for the probe: [Everest installation](https://everestapi.github.io/), [CelesteTAS endpoints](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/CelesteTAS-EverestInterop/Source/EverestInterop/DebugRcPage.cs), [savestate implementation](https://github.com/DemoJameson/Celeste.SpeedrunTool), and [TAS command documentation](https://github.com/EverestAPI/CelesteTAS-EverestInterop/wiki/Commands).
