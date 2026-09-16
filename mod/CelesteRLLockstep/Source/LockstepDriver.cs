using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Celeste;
using Celeste.Mod;
using Monocle;
using MonoMod.RuntimeDetour;
using TAS;
using TAS.Communication;
using TAS.Input;

namespace CelesteRL.Lockstep;

/// Runs the game in lockstep with a Python client instead of waiting for the 60 Hz game clock.
///
/// Every frame still goes through CelesteTAS's normal input playback and the game's normal update with
/// the normal frame length. When CelesteTAS has played every input it has, this driver sends Python the
/// resulting state and waits for the next input line, which it appends to CelesteTAS's input list in
/// memory. While Python keeps answering, the playback speed is raised so many frames run per game
/// tick; CelesteTAS lowers it again about every 16 ms, which lets the game draw and handle window
/// events. Raising the playback speed also enables CelesteTAS's fast-forward optimizations (skipped
/// visual-only updates), so equivalence with normal-speed playback is checked, not assumed.
///
/// Protocol (newline-delimited JSON, loopback TCP):
///   {"id": 1, "cmd": "reset", "start_frame": 300}  restore the savestate one frame before start_frame
///                                                  and play to start_frame; required first on every
///                                                  connection before stepping
///   {"id": 2, "cmd": "step", "line": "1,R,J"}      play one frame holding those buttons
///   {"id": 3, "cmd": "observe"}                    report the current state without advancing
/// Replies: {"id": 1, "frame": 301, "state": {...}, "diagnostics": {...}} or {"id": 1, "error": "..."}.
/// "state" is serialized exactly like CelesteTAS's /tas/game_state endpoint, or null with no player.
internal static class LockstepDriver {
    private const string LogTag = "CelesteRLLockstep";
    private const int DefaultPort = 32280;

    // One frame: buttons a player can bind in the controls menu, then optional dash-only (A) and
    // move-only (M) directions, in the canonical order the Python bridge writes. Checking again here
    // means the socket can never inject TAS commands.
    private static readonly Regex InputLine = new(@"^1(,[LRUDJKXCZVGHSQNO])*(,A(?=[LRUD])L?R?U?D?)?(,M(?=[LRUD])L?R?U?D?)?\z", RegexOptions.Compiled);

    private static readonly JsonSerializerOptions JsonOptions = new() { IncludeFields = true };
    private static readonly PropertyInfo PlaybackSpeedProperty =
        typeof(Manager).GetProperty(nameof(Manager.PlaybackSpeed), BindingFlags.Public | BindingFlags.Static)!;
    private static readonly FieldInfo FastForwardsField =
        typeof(InputController).GetField("FastForwards", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;

    private const float SteppingSpeed = 100_000f;
    private static readonly TimeSpan CommandWait = TimeSpan.FromMilliseconds(20);
    private static readonly TimeSpan ResetTimeout = TimeSpan.FromSeconds(10);

    private enum Phase {
        Idle,                 // nothing owed to Python
        WaitingForBreakpoint, // reset requested; waiting for the savestate to restore
        Advancing,            // an input is being played; its result is owed to Python
    }

    private static Hook? managerUpdateHook;
    private static LockstepServer? server;

    // Everything below belongs to one connection (sessionGeneration) and is discarded when it changes.
    private static int sessionGeneration;
    private static bool sessionReady; // a reset has completed on this connection
    private static Phase phase = Phase.Idle;
    private static string? pendingId;
    private static string? pendingCommand;
    private static readonly Stopwatch resetTimer = new();
    private static int resetUpdates;

    public static void Load() {
        int port = int.TryParse(Environment.GetEnvironmentVariable("CELESTE_RL_LOCKSTEP_PORT"), out int parsed)
            ? parsed
            : DefaultPort;

        var update = typeof(Manager).GetMethod(nameof(Manager.Update), BindingFlags.Public | BindingFlags.Static)!;
        managerUpdateHook = new Hook(update, (Action<Action>) OnManagerUpdate);

        server = new LockstepServer(port);
        server.Start();
        Logger.Log(LogLevel.Info, LogTag, $"Listening on 127.0.0.1:{port}");
    }

    public static void Unload() {
        managerUpdateHook?.Dispose();
        managerUpdateHook = null;
        server?.Dispose();
        server = null;
    }

    /// Wraps CelesteTAS's Manager.Update, which runs at the start of every game update before physics.
    private static void OnManagerUpdate(Action orig) {
        int generation = server?.CurrentGeneration ?? 0;
        if (generation != sessionGeneration) {
            // A different connection (or none) now owns the socket. Whatever the old one was waiting
            // for is abandoned; the new one must reset before it can step.
            if (phase != Phase.Idle) {
                Debug($"abandoned {pendingCommand} {pendingId} from connection {sessionGeneration}");
            }
            sessionGeneration = generation;
            sessionReady = false;
            ClearPending();
        }

        if (generation == 0) {
            orig();
            return;
        }

        if (!Manager.Running) {
            if (phase != Phase.Idle) {
                Fail($"TAS playback stopped while {pendingCommand} was pending");
            }
            orig();
            return;
        }

        var controller = Manager.Controller;

        if (phase == Phase.WaitingForBreakpoint) {
            SetPlaybackSpeed(SteppingSpeed);
            resetUpdates += 1;
            // The restored savestate pauses on the breakpoint, one frame before the episode start.
            if (Manager.CurrState == Manager.State.Paused && controller.CurrentFrameInTas == controller.Inputs.Count - 1) {
                Debug("breakpoint reached, advancing to episode start");
                Manager.NextState = Manager.State.Running;
                phase = Phase.Advancing;
            } else if (resetTimer.Elapsed > ResetTimeout) {
                Fail($"reset did not reach the savestate breakpoint within {ResetTimeout.TotalSeconds:F0} s " +
                     $"(frame {controller.CurrentFrameInTas}/{controller.Inputs.Count}, state {Manager.CurrState})");
            }
        } else if (controller.CurrentFrameInTas >= controller.Inputs.Count) {
            // Every input so far has been played and simulated.
            if (phase == Phase.Advancing && pendingId != null) {
                if (pendingCommand == "reset") {
                    sessionReady = true;
                }
                SendObservation(pendingId);
            }
            ClearPending();
            HandleNextCommand(controller);
        }

        orig();
    }

    private static void HandleNextCommand(InputController controller) {
        if (!server!.TryTake(sessionGeneration, out string message, CommandWait)) {
            // Python has not answered yet: let CelesteTAS end this game tick so the window keeps drawing.
            SetPlaybackSpeed(1.0f);
            return;
        }

        string id = "null";
        try {
            using var document = JsonDocument.Parse(message);
            var root = document.RootElement;
            id = root.GetProperty("id").GetRawText();
            Debug($"command {message}");

            switch (root.GetProperty("cmd").GetString()) {
                case "step": {
                    if (!sessionReady) {
                        SendError(id, "This connection has not completed a reset; send reset before step");
                        return;
                    }
                    string line = root.GetProperty("line").GetString() ?? "";
                    if (!InputLine.IsMatch(line)) {
                        SendError(id, $"Rejected input line {JsonSerializer.Serialize(line)}");
                        return;
                    }

                    int before = controller.Inputs.Count;
                    controller.AddFrames(line, controller.FilePath, 0, 0);
                    if (controller.Inputs.Count != before + 1) {
                        SendError(id, $"CelesteTAS did not add exactly one frame for {line}");
                        return;
                    }

                    Manager.NextState = Manager.State.Running;
                    SetPlaybackSpeed(SteppingSpeed);
                    Begin(id, "step", Phase.Advancing);
                    return;
                }

                case "reset": {
                    sessionReady = false;
                    int startFrame = root.GetProperty("start_frame").GetInt32();

                    // Re-reading the file drops the in-memory inputs, leaving only the fixed prefix.
                    controller.RefreshInputs(forceRefresh: true);
                    if (controller.Inputs.Count != startFrame) {
                        SendError(id, $"TAS file has {controller.Inputs.Count} input frames, expected a prefix of {startFrame}");
                        return;
                    }
                    if (!HasSavestateBreakpoint(controller, startFrame - 1)) {
                        SendError(id, $"TAS file has no savestate breakpoint (***S) at frame {startFrame - 1}");
                        return;
                    }

                    // Disabling and re-enabling the run is what CelesteTAS's Restart hotkey does:
                    // EnableRun, at the start of Manager.Update, restores the breakpoint savestate, or
                    // replays the prefix to create it if no valid savestate exists.
                    Manager.DisableRun();
                    Manager.NextState = Manager.State.Running;
                    SetPlaybackSpeed(SteppingSpeed);
                    resetTimer.Restart();
                    resetUpdates = 0;
                    Begin(id, "reset", Phase.WaitingForBreakpoint);
                    return;
                }

                case "observe":
                    SendObservation(id);
                    return;

                default:
                    SendError(id, "Unknown command");
                    return;
            }
        } catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException or FormatException) {
            SendError(id, $"Malformed command: {e.Message}");
        }
    }

    private static bool HasSavestateBreakpoint(InputController controller, int frame) {
        // FastForwards is internal to CelesteTAS: frame -> FastForward record with a SaveState flag.
        if (FastForwardsField.GetValue(controller) is not IDictionary breakpoints || !breakpoints.Contains(frame)) {
            return false;
        }
        object? breakpoint = breakpoints[frame];
        return breakpoint?.GetType().GetProperty("SaveState")?.GetValue(breakpoint) is true;
    }

    private static void Begin(string id, string command, Phase next) {
        pendingId = id;
        pendingCommand = command;
        phase = next;
    }

    private static void ClearPending() {
        phase = Phase.Idle;
        pendingId = null;
        pendingCommand = null;
        resetTimer.Reset();
    }

    /// Report that the pending command failed, and require a new reset before stepping.
    private static void Fail(string error) {
        if (pendingId != null) {
            SendError(pendingId, error);
        } else {
            Logger.Log(LogLevel.Warn, LogTag, error);
        }
        sessionReady = false;
        ClearPending();
    }

    private static void SendObservation(string id) {
        string state;
        try {
            // During a death the Player entity is removed until respawn. GetGameState assumes a player
            // and would throw; /tas/game_state returns an empty body in that case, which the HTTP bridge
            // reads as no state. Report null here too so both bridges agree.
            state = Engine.Scene is Level level && level.Tracker.GetEntity<Player>() == null
                ? "null"
                : JsonSerializer.Serialize(GameData.GetGameState(), JsonOptions);
        } catch (Exception e) {
            // GetGameState assumes a player exists. An exception here must never take down the game.
            SendError(id, $"Could not read game state: {e.GetType().Name}: {e.Message}");
            sessionReady = false;
            return;
        }

        // Diagnostics for the transition, loading and freeze-frame checks. Not a policy observation.
        string diagnostics = JsonSerializer.Serialize(new Dictionary<string, object?> {
            ["loading"] = Manager.IsLoading(),
            ["freeze_timer"] = Engine.FreezeTimer,
            ["scene"] = Engine.Scene?.GetType().FullName,
            ["level_paused"] = (Engine.Scene as Level)?.Paused,
            ["reset_updates"] = pendingCommand == "reset" ? resetUpdates : null,
        });

        Debug($"reply {id} frame={Manager.Controller.CurrentFrameInTas}");
        server!.Send(sessionGeneration,
            $"{{\"id\":{id},\"frame\":{Manager.Controller.CurrentFrameInTas},\"state\":{state},\"diagnostics\":{diagnostics}}}");
    }

    private static void SendError(string id, string error) {
        server!.Send(sessionGeneration, $"{{\"id\":{id},\"error\":{JsonSerializer.Serialize(error)}}}");
        Logger.Log(LogLevel.Warn, LogTag, error);
    }

    private static readonly bool DebugLogging = Environment.GetEnvironmentVariable("CELESTE_RL_LOCKSTEP_DEBUG") == "1";

    private static void Debug(string message) {
        if (DebugLogging) {
            var controller = Manager.Controller;
            Logger.Log(LogLevel.Info, LogTag,
                $"{message} | connection={sessionGeneration} ready={sessionReady} phase={phase} running={Manager.Running} " +
                $"curr={Manager.CurrState} next={Manager.NextState} frame={controller.CurrentFrameInTas}/{controller.Inputs.Count} " +
                $"loading={Manager.IsLoading()}");
        }
    }

    private static void SetPlaybackSpeed(float speed) => PlaybackSpeedProperty.SetValue(null, speed);
}
