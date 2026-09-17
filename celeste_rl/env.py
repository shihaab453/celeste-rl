"""CelesteRoomEnv: the Gymnasium environment for one Celeste room (Phase 2 spec).

One step is one game frame. The action is act-v1 (24 on/off inputs), the observation obs-v1, and the reward
whichever version `reward_config` selects: rew-v1 unshaped, or rew-v2 with the unspent-deadline charge and
progress shaping over `celeste_rl.potential`. The episode ends on success, death, restart, leaving the level,
entering the wrong room or the 30-second deadline; every ending is task termination, and the environment
never truncates.

Bridge faults: a transport or protocol failure, a malformed reply, or events the ending rules cannot explain
raise BridgeFault. That step produces no transition, the episode cannot continue, and step() refuses until
reset(). The caller discards the interrupted data; the environment never fabricates a transition and never
retries a step.

What leaves the environment besides the observation (reward components, ending cause, events, applied action,
frame) is returned in `info` and is never part of the observation.
"""
from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from celeste_rl.actions import apply_disabled, disabled_mask, to_parts
from celeste_rl.bridge import BridgeError
from celeste_rl.endings import EndingFault, RoomTask, classify
from celeste_rl.observation import ObservationBuilder, SchemaViolation, observation_space
from celeste_rl.potential import RoomPotential
from celeste_rl.reward import RewardConfig, reward_components
from celeste_rl.schema import ACT_VERSION, ACTION_INPUTS, FINGERPRINT, MENU_INPUTS, OBS_VERSION

# The canonical start: TAS frame 300 of the episode prefix, room 1, standing at (19, 144) with one dash.
CANONICAL_START = {"room": "1", "position": (19, 144), "dashes": 1}


class BridgeFault(RuntimeError):
    """The step or reset could not be completed or trusted. Discard the interrupted episode and call reset()."""


class CelesteRoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bridge, disabled_inputs: Iterable[str] = MENU_INPUTS, task: RoomTask = RoomTask(),
                 reward_config: RewardConfig = RewardConfig()):
        """`bridge` is a LockstepBridge (or anything with the same reset/step/close). `disabled_inputs` defaults to
        pause, quick restart and journal (decision D1); pass () to enable all 24 inputs."""
        self.bridge = bridge
        self.task = task
        self.reward_config = reward_config
        self.disabled_inputs = tuple(disabled_inputs)
        self._mask = disabled_mask(self.disabled_inputs)
        self.action_space = spaces.MultiBinary(len(ACTION_INPUTS))
        self.observation_space = observation_space()
        self._builder = ObservationBuilder()
        self._elapsed = 0
        self._ready = False
        # rew-v2 only: the room's progress potential, rebuilt only when the room changes, and its value at
        # the previous step. Both stay out of the observation.
        self._potential: RoomPotential | None = None
        self._potential_value = 0.0

    def _info(self, **extra) -> dict:
        return {
            "obs_version": OBS_VERSION,
            "act_version": ACT_VERSION,
            "reward_version": self.reward_config.version,
            "schema_fingerprint": FINGERPRINT,
            "disabled_inputs": self.disabled_inputs,
            "elapsed": self._elapsed,
            **extra,
        }

    def _check_start(self, observation) -> None:
        state, extras = observation.state, observation.extras
        problems = []
        if observation.tas_frame != getattr(getattr(self.bridge, "http", None), "start_frame", 300):
            problems.append(f"frame {observation.tas_frame}")
        if state is None or not isinstance(extras, dict):
            problems.append("no player or no extras")
        else:
            player, level = extras.get("player", {}), extras.get("level", {})
            position = state.get("Player", {}).get("Position", {})
            checks = {
                "room": (observation.room, self.task.start_room),
                "position": ((position.get("X"), position.get("Y")), CANONICAL_START["position"]),
                "state": (player.get("State"), 0),
                "in control": (player.get("InControl"), True),
                "paused": (level.get("Paused"), False),
                "freeze timer": (level.get("FreezeTimer"), 0),
                "dashes": ((player.get("Dashes"), player.get("MaxDashes")), (CANONICAL_START["dashes"],) * 2),
            }
            problems += [f"{name} {actual!r} (expected {expected!r})" for name, (actual, expected) in checks.items()
                         if actual != expected]
        if problems:
            raise BridgeFault(f"Episode start is not the canonical start: {'; '.join(problems)}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ready = False
        try:
            observation = self.bridge.reset()
            self._check_start(observation)
            obs = self._builder.reset(observation.state, observation.extras)
        except (BridgeError, SchemaViolation) as error:
            raise BridgeFault(f"reset failed: {error}") from error
        self._elapsed = 0
        self._potential_value = self._start_potential(observation.state)
        self._ready = True
        return obs, self._info(start="canonical", frame=observation.tas_frame, reset_events=observation.events,
                               potential=self._potential_value)

    def step(self, action):
        if not self._ready:
            raise RuntimeError("The episode has ended or faulted; call reset() before step()")
        # An invalid action raises ValueError here, before anything is sent, and the episode can continue.
        applied = apply_disabled(action, self._mask)

        # From here any failure leaves the game in an unknown or ended state.
        self._ready = False
        try:
            observation = self.bridge.step(*to_parts(applied))
        except BridgeError as error:
            raise BridgeFault(f"step failed: {error}") from error
        elapsed = self._elapsed + 1
        try:
            ending = classify(observation.events, observation.state, elapsed, self.task)
            obs = self._builder.step(observation.state, observation.extras, applied, elapsed)
        except (EndingFault, SchemaViolation) as error:
            raise BridgeFault(f"step at frame {observation.tas_frame} could not be interpreted: {error}") from error

        self._elapsed = elapsed
        terminated = ending is not None
        # The terminal potential is 0 by definition, so an episode's shaping sums to -scale * potential(start).
        value = 0.0 if terminated or self._potential is None else self._potential.value(observation.state)
        shaping = 0.0
        if self.reward_config.shaped:
            shaping = self.reward_config.shaping_scale * (self.reward_config.gamma * value - self._potential_value)
        self._potential_value = value
        components = reward_components(ending, self.reward_config, elapsed, shaping)
        self._ready = not terminated
        info = self._info(ending=ending, events=observation.events, reward_components=components,
                          applied_action=applied, frame=observation.tas_frame, potential=value)
        return obs, float(sum(components.values())), terminated, False, info

    def _start_potential(self, state: dict | None) -> float:
        """Build the room's potential if rew-v2 asked for shaping, reusing it while the room is unchanged."""
        if not self.reward_config.shaped:
            return 0.0
        try:
            if self._potential is None or not self._potential.matches(state):
                self._potential = RoomPotential(state)
            return self._potential.value(state)
        except (KeyError, TypeError, ValueError) as error:
            raise BridgeFault(f"the room's progress potential could not be built: {error}") from error

    def close(self):
        self._ready = False
        self.bridge.close()
