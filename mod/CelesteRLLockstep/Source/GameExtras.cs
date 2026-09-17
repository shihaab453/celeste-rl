using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;
using System.Text.Json;
using Celeste;
using Celeste.Mod;
using Microsoft.Xna.Framework;
using Monocle;

namespace CelesteRL.Lockstep;

/// Read-only player, input-buffer and level facts that CelesteTAS's game state does not include, for the
/// environment's observation (docs/planning/phase2-environment-spec.md, section 4.1).
///
/// Nothing here writes game state or consumes input. Input buffers are read from VirtualButton's private
/// bufferCounter field, never through Pressed or Check, which can consume the buffer. Every member is
/// resolved once at load; if any is missing (a different game build), Incompatibility is set and the
/// driver refuses commands instead of sending a shorter object.
internal static class GameExtras {
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private const BindingFlags Static = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    // Player members read by name. Fields and properties both work; the getter is compiled once.
    private static readonly string[] PlayerMembers = {
        "Dashes", "MaxDashes", "Stamina", "Facing", "Ducking",
        "dashCooldownTimer", "dashRefillCooldownTimer", "dashAttackTimer", "DashDir",
        "jumpGraceTimer", "varJumpSpeed", "AutoJumpTimer",
        "wallSlideTimer", "wallSpeedRetentionTimer", "wallSpeedRetained", "wallBoostTimer", "wallBoostDir",
        "forceMoveX", "forceMoveXTimer", "climbNoMoveTimer",
        "LiftBoost", "InControl", "Dead", "JustRespawned", "hurtbox",
    };

    // Input buttons whose buffered press is exported. Grab has no buffer (its buffer time is zero).
    private static readonly string[] BufferedButtons = { "Jump", "Dash", "CrouchDash" };

    private static readonly Dictionary<string, Func<Player, object?>> playerGetters = new();
    private static readonly Dictionary<string, FieldInfo> buttonFields = new();
    private static FieldInfo? bufferCounterField;

    public static string? Incompatibility { get; private set; }

    public static void Load() {
        var missing = new List<string>();
        foreach (string name in PlayerMembers) {
            var getter = CompileGetter(name);
            if (getter == null) {
                missing.Add($"Player.{name}");
            } else {
                playerGetters[name] = getter;
            }
        }
        bufferCounterField = typeof(VirtualButton).GetField("bufferCounter", Instance);
        if (bufferCounterField == null || bufferCounterField.FieldType != typeof(float)) {
            missing.Add("VirtualButton.bufferCounter");
        }
        foreach (string name in BufferedButtons) {
            var field = typeof(Input).GetField(name, Static);
            if (field == null || !typeof(VirtualButton).IsAssignableFrom(field.FieldType)) {
                missing.Add($"Input.{name}");
            } else {
                buttonFields[name] = field;
            }
        }
        if (typeof(Level).GetField("InCutscene", Instance) == null && typeof(Level).GetProperty("InCutscene", Instance) == null) {
            missing.Add("Level.InCutscene");
        }
        Incompatibility = missing.Count > 0 ? $"game members not found: {string.Join(", ", missing)}" : null;
    }

    private static Func<Player, object?>? CompileGetter(string name) {
        MemberInfo? member = (MemberInfo?) typeof(Player).GetField(name, Instance)
                             ?? typeof(Player).GetProperty(name, Instance);
        if (member == null || member is PropertyInfo { GetMethod: null }) {
            return null;
        }
        var player = Expression.Parameter(typeof(Player));
        var access = Expression.MakeMemberAccess(player, member);
        return Expression.Lambda<Func<Player, object?>>(Expression.Convert(access, typeof(object)), player).Compile();
    }

    /// The extras object as JSON, or "null" when there is no Level or no player.
    public static string Serialize() {
        if (Incompatibility != null || Engine.Scene is not Level level || level.Tracker.GetEntity<Player>() is not { } player) {
            return "null";
        }

        var values = new Dictionary<string, object?>();
        foreach (var (name, getter) in playerGetters) {
            object? value = getter(player);
            values[name] = value switch {
                Vector2 v => new Dictionary<string, float> { ["X"] = v.X, ["Y"] = v.Y },
                Facings facing => (int) facing,
                Hitbox box => Rect(box),
                _ => value,
            };
        }
        values["Collider"] = player.Collider is Hitbox collider ? Rect(collider) : null;
        values["State"] = player.StateMachine.State;

        var buffers = new Dictionary<string, float>();
        foreach (var (name, field) in buttonFields) {
            buffers[name] = field.GetValue(null) is VirtualButton button ? (float) bufferCounterField!.GetValue(button)! : 0f;
        }

        return JsonSerializer.Serialize(new Dictionary<string, object?> {
            ["player"] = values,
            ["input_buffers"] = buffers,
            ["level"] = new Dictionary<string, object?> {
                ["Paused"] = level.Paused,
                ["InCutscene"] = level.InCutscene,
                ["FreezeTimer"] = Engine.FreezeTimer,
            },
        });
    }

    /// A hitbox relative to its entity's position.
    private static Dictionary<string, float> Rect(Hitbox box) =>
        new() { ["X"] = box.Position.X, ["Y"] = box.Position.Y, ["W"] = box.Width, ["H"] = box.Height };

    /// For validation probes only: whether each world rectangle collides with any Solid in the level,
    /// using the game's own collision check. Null when the scene is not a Level.
    public static bool[]? QuerySolids(JsonElement rects) {
        if (Engine.Scene is not Level level) {
            return null;
        }
        var results = new List<bool>();
        foreach (var rect in rects.EnumerateArray()) {
            var r = new Rectangle(rect[0].GetInt32(), rect[1].GetInt32(), rect[2].GetInt32(), rect[3].GetInt32());
            results.Add(level.CollideCheck<Solid>(r));
        }
        return results.ToArray();
    }
}

/// Game events latched between replies (spec section 4.2), so an episode ending that happens during
/// automatic loading, such as a chapter restart, is still reported on the reply of the step that caused it.
/// Handlers only record; they never change what the game does.
internal static class GameEvents {
    private static readonly List<Dictionary<string, object?>> latched = new();
    private static bool subscribed;

    // The room the current step started in, so a transition can name where it came from (the game has
    // already switched Session.Level when the transition event fires).
    public static string? RoomAtLastReply { get; set; }

    public static void Subscribe() {
        if (subscribed) {
            return;
        }
        Everest.Events.Player.OnDie += OnDie;
        Everest.Events.Level.OnTransitionTo += OnTransitionTo;
        Everest.Events.Level.OnLoadLevel += OnLoadLevel;
        Everest.Events.Level.OnExit += OnExit;
        Everest.Events.Level.OnPause += OnPause;
        Everest.Events.Level.OnUnpause += OnUnpause;
        subscribed = true;
    }

    public static void Unsubscribe() {
        if (!subscribed) {
            return;
        }
        Everest.Events.Player.OnDie -= OnDie;
        Everest.Events.Level.OnTransitionTo -= OnTransitionTo;
        Everest.Events.Level.OnLoadLevel -= OnLoadLevel;
        Everest.Events.Level.OnExit -= OnExit;
        Everest.Events.Level.OnPause -= OnPause;
        Everest.Events.Level.OnUnpause -= OnUnpause;
        subscribed = false;
    }

    public static void Clear() => latched.Clear();

    /// The latched events as a JSON array, then cleared.
    public static string TakeJson() {
        string json = JsonSerializer.Serialize(latched);
        latched.Clear();
        return json;
    }

    private static void Add(string type, Dictionary<string, object?>? payload = null) {
        var entry = new Dictionary<string, object?> { ["type"] = type };
        if (payload != null) {
            foreach (var (key, value) in payload) {
                entry[key] = value;
            }
        }
        latched.Add(entry);
    }

    private static void OnDie(Player player) =>
        Add("death", new() { ["room"] = (player.Scene as Level)?.Session.Level });

    private static void OnTransitionTo(Level level, LevelData next, Vector2 direction) =>
        Add("transition", new() {
            ["from"] = RoomAtLastReply,
            ["to"] = next.Name,
            ["direction"] = new Dictionary<string, float> { ["X"] = direction.X, ["Y"] = direction.Y },
        });

    private static void OnLoadLevel(Level level, Player.IntroTypes intro, bool isFromLoader) =>
        Add("load_level", new() { ["room"] = level.Session.Level, ["intro"] = intro.ToString(), ["from_loader"] = isFromLoader });

    private static void OnExit(Level level, LevelExit exit, LevelExit.Mode mode, Session session, HiresSnow snow) =>
        Add("level_exit", new() { ["mode"] = mode.ToString() });

    private static void OnPause(Level level, int startIndex, bool minimal, bool quickReset) =>
        Add("pause", new() { ["quick_reset"] = quickReset });

    private static void OnUnpause(Level level) => Add("unpause");
}
