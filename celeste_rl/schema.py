"""The environment's observation and action schemas (obs-v1, act-v1): the single source of truth.

docs/planning/phase2-environment-spec.md describes the design; the tables here are authoritative. Any
change to a feature, scale, channel, state list, action order or constant is a new schema version, and
changes FINGERPRINT, which checkpoints record and compare.

Scales come from the pinned game build's constants (Celeste.Player): DashSpeed 240, SuperJump horizontal
260, ClimbMaxStamina 110, VarJumpTime 0.2 (0.25 for super wall jumps), JumpGraceTime 0.1, DashCooldown
0.2, DashRefillCooldown 0.1, DashAttackTime 0.3, WallSlideTime 1.2, WallSpeedRetentionTime 0.06,
ClimbNoMoveTime 0.1, ClimbJumpBoostTime 0.2, BounceAutoJumpTime 0.1, MaxFall 160 and FastMaxFall 240,
lift boost caps 250, input buffer 0.08 s. Timers are in seconds, as the game stores them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

OBS_VERSION = "obs-v1"
ACT_VERSION = "act-v1"

# Task constants that are part of what the policy sees.
DEADLINE_FRAMES = 1800  # 30 s of normal play, counted in acknowledged decision frames
HISTORY = 4  # decisions of player features and applied actions, row 0 the most recent

# Actions: one on/off value per input, in this order. Plain letters are buttons (in the bridge's canonical
# letter order); "A"/"M" plus a direction are dash-only and move-only directions.
ACTION_INPUTS = (
    "L", "R", "U", "D", "J", "K", "X", "C", "Z", "V", "G", "H", "S", "Q", "N", "O",
    "AL", "AR", "AU", "AD", "ML", "MR", "MU", "MD",
)
MENU_INPUTS = ("S", "Q", "N")  # disabled for initial training (decision D1)

# Grid: 8 px cells aligned with the room's tile grid, 32 x 32, the cell containing the centre of the player's
# collider at index (16, 16).
CELL_SIZE = 8
GRID_SIZE = 32
GRID_ANCHOR = 16
GRID_CHANNELS = (
    "solid",
    "jumpthru_up", "jumpthru_down", "jumpthru_left", "jumpthru_right",
    "spikes_up", "spikes_down", "spikes_left", "spikes_right",
    "other_hazards",
    "outside_room",
)
SPINNER_BOX = 16  # spinners are exported as positions; stamped as a box this wide around the position
# CelesteTAS exports entity rectangles as the entity's position plus its collider's size, not the collider's
# location. Spike hitboxes are offset from the position by direction: up spikes' hitbox is 3 px above it,
# confirmed on the recorded room 1 spike deaths (the player's hurtbox, y 157-166, overlaps the shifted spike,
# y 165-168, not the exported one, y 168-171). The left offset mirrors it and is not yet confirmed on a trace;
# down and right are assumed to start at the position. Other exported solids, jump-throughs and lightning are
# assumed to have colliders at their position; probe P8 checks solids.
SPIKE_OFFSETS = {"up": (0, -3), "down": (0, 0), "left": (-3, 0), "right": (0, 0)}

# Vanilla Celeste.Player state indices 0-25 (StNormal = 0 ... StIntroThinkForABit = 25), then "other".
PLAYER_STATES = (
    "StNormal", "StClimb", "StDash", "StSwim", "StBoost", "StRedDash", "StHitSquash", "StLaunch",
    "StPickup", "StDreamDash", "StSummitLaunch", "StDummy", "StIntroWalk", "StIntroJump", "StIntroRespawn",
    "StIntroWakeUp", "StBirdDashTutorial", "StFrozen", "StReflectionFall", "StStarFly", "StTempleFall",
    "StCassetteFly", "StAttract", "StIntroMoonJump", "StFlingBird", "StIntroThinkForABit", "other",
)


@dataclass(frozen=True)
class Feature:
    """One entry of the player feature vector.

    kind:
      value    float(source) / scale, not clipped
      flag     1.0 if source is truthy, else 0.0
      nonzero  1.0 if source != 0, else 0.0
      buffer   max(source, 0) / scale (a counter at or below 0 means nothing is buffered)
      room_x   (state.Player.Position.X - state.Level.Bounds.X) / state.Level.Bounds.W, not clipped
      room_y   (state.Player.Position.Y - state.Level.Bounds.Y) / state.Level.Bounds.H, not clipped
    source is a path into {"state": <bridge state>, "extras": <mod extras>}.
    """

    name: str
    kind: str
    source: tuple[str, ...]
    scale: float = 1.0


P = ("extras", "player")
PLAYER_FEATURES = (
    Feature("position_x", "room_x", ("state", "Player", "Position", "X")),
    Feature("position_y", "room_y", ("state", "Player", "Position", "Y")),
    Feature("remainder_x", "value", ("state", "Player", "PositionRemainder", "X")),
    Feature("remainder_y", "value", ("state", "Player", "PositionRemainder", "Y")),
    Feature("speed_x", "value", ("state", "Player", "Speed", "X"), 400),
    Feature("speed_y", "value", ("state", "Player", "Speed", "Y"), 400),
    Feature("lift_boost_x", "value", (*P, "LiftBoost", "X"), 250),
    Feature("lift_boost_y", "value", (*P, "LiftBoost", "Y"), 250),
    Feature("on_ground", "flag", ("state", "Player", "OnGround")),
    Feature("is_holding", "flag", ("state", "Player", "IsHolding")),
    Feature("auto_jump", "flag", ("state", "Player", "AutoJump")),
    Feature("ducking", "flag", (*P, "Ducking")),
    Feature("in_control", "flag", (*P, "InControl")),
    Feature("dead", "flag", (*P, "Dead")),
    Feature("just_respawned", "flag", (*P, "JustRespawned")),
    Feature("wall_speed_retained_active", "nonzero", (*P, "wallSpeedRetained")),
    Feature("dashes", "value", (*P, "Dashes"), 2),
    Feature("max_dashes", "value", (*P, "MaxDashes"), 2),
    Feature("stamina", "value", (*P, "Stamina"), 110),
    Feature("facing", "value", (*P, "Facing")),
    Feature("dash_dir_x", "value", (*P, "DashDir", "X")),
    Feature("dash_dir_y", "value", (*P, "DashDir", "Y")),
    Feature("wall_boost_dir", "value", (*P, "wallBoostDir")),
    Feature("force_move_x", "value", (*P, "forceMoveX")),
    Feature("collider_top", "value", (*P, "Collider", "Y"), 11),
    Feature("collider_height", "value", (*P, "Collider", "H"), 11),
    Feature("hurtbox_top", "value", (*P, "hurtbox", "Y"), 11),
    Feature("hurtbox_height", "value", (*P, "hurtbox", "H"), 11),
    Feature("var_jump_timer", "value", ("state", "Player", "JumpTimer"), 0.25),
    Feature("jump_grace_timer", "value", (*P, "jumpGraceTimer"), 0.1),
    Feature("var_jump_speed", "value", (*P, "varJumpSpeed"), 160),
    Feature("auto_jump_timer", "value", (*P, "AutoJumpTimer"), 0.1),
    Feature("max_fall", "value", ("state", "Player", "MaxFall"), 320),
    Feature("dash_cooldown_timer", "value", (*P, "dashCooldownTimer"), 0.2),
    Feature("dash_refill_cooldown_timer", "value", (*P, "dashRefillCooldownTimer"), 0.1),
    Feature("dash_attack_timer", "value", (*P, "dashAttackTimer"), 0.3),
    Feature("wall_slide_timer", "value", (*P, "wallSlideTimer"), 1.2),
    Feature("wall_speed_retention_timer", "value", (*P, "wallSpeedRetentionTimer"), 0.06),
    Feature("wall_speed_retained", "value", (*P, "wallSpeedRetained"), 400),
    Feature("wall_boost_timer", "value", (*P, "wallBoostTimer"), 0.2),
    Feature("force_move_x_timer", "value", (*P, "forceMoveXTimer"), 0.2),
    Feature("climb_no_move_timer", "value", (*P, "climbNoMoveTimer"), 0.1),
    Feature("jump_buffer", "buffer", ("extras", "input_buffers", "Jump"), 0.08),
    Feature("dash_buffer", "buffer", ("extras", "input_buffers", "Dash"), 0.08),
    Feature("crouch_dash_buffer", "buffer", ("extras", "input_buffers", "CrouchDash"), 0.08),
    Feature("paused", "flag", ("extras", "level", "Paused")),
    Feature("in_cutscene", "flag", ("extras", "level", "InCutscene")),
    Feature("freeze_timer", "value", ("extras", "level", "FreezeTimer"), 0.1),
)
# The movement state one-hot follows the features above, one slot per PLAYER_STATES entry.
STATE_SOURCE = (*P, "State")
PLAYER_FEATURE_NAMES = tuple(f.name for f in PLAYER_FEATURES) + tuple(f"state_{s}" for s in PLAYER_STATES)
PLAYER_FEATURE_COUNT = len(PLAYER_FEATURE_NAMES)

CONTEXT_NAMES = ("elapsed", "player_present")


def _fingerprint() -> str:
    description = {
        "obs_version": OBS_VERSION,
        "act_version": ACT_VERSION,
        "deadline_frames": DEADLINE_FRAMES,
        "history": HISTORY,
        "action_inputs": ACTION_INPUTS,
        "grid": {"cell": CELL_SIZE, "size": GRID_SIZE, "anchor": GRID_ANCHOR, "channels": GRID_CHANNELS,
                 "spinner_box": SPINNER_BOX, "spike_offsets": SPIKE_OFFSETS},
        "player_features": [asdict(f) for f in PLAYER_FEATURES],
        "state_source": STATE_SOURCE,
        "player_states": PLAYER_STATES,
        "context": CONTEXT_NAMES,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:16]


FINGERPRINT = _fingerprint()
