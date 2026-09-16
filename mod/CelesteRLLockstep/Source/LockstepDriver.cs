using System;
using System.Collections.Generic;
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
/// Physics are untouched: every frame still goes through CelesteTAS's normal input playback and the
/// game's normal update with the normal frame length. The only change is timing. When CelesteTAS has
/// played every input it has, this driver sends Python the resulting state and waits for the next
/// input line, which it appends to CelesteTAS's input list in memory. While Python keeps answering,
/// the playback speed is raised so many frames run per game tick; CelesteTAS lowers it again about
/// every 16 ms, which lets the game draw and handle window events.
///
/// Protocol (newline-delimited JSON, loopback TCP):
///   {"id": 1, "cmd": "step", "line": "1,R,J"}  play one frame holding those buttons
///   {"id": 2, "cmd": "reset"}                  restore the savestate and play to the episode start
///   {"id": 3, "cmd": "observe"}                report the current state without advancing
/// Replies: {"id": 1, "frame": 301, "state": {...}} or {"id": 1, "error": "..."}.
/// "state" is serialized exactly like CelesteTAS's /tas/game_state endpoint.
internal static class LockstepDriver {
    private const string LogTag = "CelesteRLLockstep";
    private const int DefaultPort = 32280;

    // One frame, and only buttons a player can bind in the controls menu. The Python bridge
    // enforces the same rule; checking again here means the socket can never inject TAS commands.
    private static readonly Regex InputLine = new("^1(,[LRUDJKXCZVGHSQNO])*$", RegexOptions.Compiled);

    private static readonly JsonSerializerOptions JsonOptions = new() { IncludeFields = true };
    private static readonly PropertyInfo PlaybackSpeedProperty =
        typeof(Manager).GetProperty(nameof(Manager.PlaybackSpeed), BindingFlags.Public | BindingFlags.Static)!;

    private const float SteppingSpeed = 100_000f;
    private static readonly TimeSpan CommandWait = TimeSpan.FromMilliseconds(20);

    private enum Phase {
        Idle,                 // nothing owed to Python
        WaitingForBreakpoint, // reset requested; waiting for the savestate to restore
        Advancing,            // an input is being played; its result is owed to Python
    }

    private static Hook? managerUpdateHook;
    private static LockstepServer? server;
    private static Phase phase = Phase.Idle;
    private static string? pendingId;

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
        if (server is not { Connected: true } || !Manager.Running) {
            if (phase != Phase.Idle) {
                Debug("released: client gone or TAS not running");
            }
            phase = Phase.Idle;
            pendingId = null;
            orig();
            return;
        }

        var controller = Manager.Controller;

        if (phase == Phase.WaitingForBreakpoint) {
            SetPlaybackSpeed(SteppingSpeed);
            // The restored savestate pauses on the breakpoint, one frame before the episode start.
            if (Manager.CurrState == Manager.State.Paused && controller.CurrentFrameInTas == controller.Inputs.Count - 1) {
                Debug("breakpoint reached, advancing to episode start");
                Manager.NextState = Manager.State.Running;
                phase = Phase.Advancing;
            }
        } else if (controller.CurrentFrameInTas >= controller.Inputs.Count) {
            // Every input so far has been played and simulated.
            if (phase == Phase.Advancing && pendingId != null) {
                SendObservation(pendingId);
            }
            phase = Phase.Idle;
            pendingId = null;
            HandleNextCommand(controller);
        }

        orig();
        if (!Manager.Running) {
            Debug("TAS stopped during this update");
        }
    }

    private static void HandleNextCommand(InputController controller) {
        if (!server!.TryTake(out string message, CommandWait)) {
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
                    pendingId = id;
                    phase = Phase.Advancing;
                    return;
                }

                case "reset":
                    // Re-reading the file drops the in-memory inputs, leaving only the fixed prefix.
                    // Disabling and re-enabling the run is what CelesteTAS's Restart hotkey does:
                    // EnableRun, at the start of Manager.Update, restores the breakpoint savestate.
                    controller.RefreshInputs(forceRefresh: true);
                    Manager.DisableRun();
                    Manager.NextState = Manager.State.Running;
                    SetPlaybackSpeed(SteppingSpeed);
                    pendingId = id;
                    phase = Phase.WaitingForBreakpoint;
                    return;

                case "observe":
                    SendObservation(id);
                    return;

                default:
                    SendError(id, "Unknown command");
                    return;
            }
        } catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException) {
            SendError(id, e.Message);
        }
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
            return;
        }
        Debug($"reply {id} frame={Manager.Controller.CurrentFrameInTas}");
        server!.Send($"{{\"id\":{id},\"frame\":{Manager.Controller.CurrentFrameInTas},\"state\":{state}}}");
    }

    private static readonly bool DebugLogging = Environment.GetEnvironmentVariable("CELESTE_RL_LOCKSTEP_DEBUG") == "1";

    private static void Debug(string message) {
        if (DebugLogging) {
            var controller = Manager.Controller;
            Logger.Log(LogLevel.Info, LogTag,
                $"{message} | phase={phase} running={Manager.Running} curr={Manager.CurrState} next={Manager.NextState} " +
                $"frame={controller.CurrentFrameInTas}/{controller.Inputs.Count} loading={Manager.IsLoading()}");
        }
    }

    private static void SendError(string id, string error) {
        server!.Send($"{{\"id\":{id},\"error\":{JsonSerializer.Serialize(error)}}}");
        Logger.Log(LogLevel.Warn, LogTag, error);
    }

    private static void SetPlaybackSpeed(float speed) => PlaybackSpeedProperty.SetValue(null, speed);
}
