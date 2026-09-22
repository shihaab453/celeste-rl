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

Starts: the bridge always resets to the base Room 1 savestate. A later room may supply a `task_start` recipe
that is replayed after every base reset. With a `start_sampler`, an episode may instead begin from a state the
agent reached after that task start. Replayed setup frames are not transitions and earn no reward. `elapsed`
counts from the task start, so a later room receives the full episode deadline. A stale sampled entry falls
back to the task start with the reason in `info`; a stale task start is a bridge fault.
"""
from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from celeste_rl.actions import apply_disabled, disabled_mask, parse_line, to_line, to_parts
from celeste_rl.bridge import BridgeError
from celeste_rl.endings import SUCCESS, WRONG_ROOM, EndingFault, RoomTask, classify
from celeste_rl.observation import ObservationBuilder, SchemaViolation, observation_space
from celeste_rl.potential import RoomPotential
from celeste_rl.reward import RewardConfig, reward_components
from celeste_rl.schema import ACT_VERSION, ACTION_INPUTS, FINGERPRINT, MENU_INPUTS, OBS_VERSION
from celeste_rl.starts import Start, StartArchive, cell_of

# The canonical start: TAS frame 300 of the episode prefix, room 1, standing at (19, 144) with one dash.
CANONICAL_START = {"room": "1", "position": (19, 144), "dashes": 1}


class BridgeFault(RuntimeError):
    """The step or reset could not be completed or trusted. Discard the interrupted episode and call reset()."""


class CelesteRoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bridge, disabled_inputs: Iterable[str] = MENU_INPUTS, task: RoomTask = RoomTask(),
                 reward_config: RewardConfig = RewardConfig(), start_sampler=None,
                 archive: StartArchive | None = None, task_start: Start | None = None):
        """`bridge` is a LockstepBridge (or anything with the same reset/step/close). `disabled_inputs` defaults to
        pause, quick restart and journal (decision D1); pass () to enable all 24 inputs.

        `task_start` is the canonical entry recipe for a task after Room 1. `start_sampler` is called with no
        arguments at each reset and returns a `Start` to begin from, or None for the task start. `archive`
        collects the states this environment reaches. They are separate so an evaluation can run held-out
        starts without writing to the archive it is measuring.
        """
        self.bridge = bridge
        self.task = task
        self.reward_config = reward_config
        self.start_sampler = start_sampler
        self.archive = archive
        self.task_start = task_start
        self.disabled_inputs = tuple(disabled_inputs)
        self._mask = disabled_mask(self.disabled_inputs)
        self.action_space = spaces.MultiBinary(len(ACTION_INPUTS))
        self.observation_space = observation_space()
        self._builder = ObservationBuilder()
        self._elapsed = 0
        self._ready = False
        # The room's progress potential, rebuilt only when the room changes, and its value at the previous
        # step. Both stay out of the observation. It is recorded under every reward version, so an unshaped
        # run can be compared with a shaped one, but only rew-v2 pays anything for it.
        self._potential: RoomPotential | None = None
        self._potential_value = 0.0
        # The input lines of the episode so far, including any replayed start prefix. This is what an archive
        # entry is made of, so it is kept whenever an archive is attached and left empty otherwise.
        self._lines: list[str] = []
        self._start_kind, self._start_frames = "canonical", 0
        self._start: Start | None = None
        self._start_problem: str | None = None

    def _info(self, **extra) -> dict:
        return {
            "obs_version": OBS_VERSION,
            "act_version": ACT_VERSION,
            "reward_version": self.reward_config.version,
            "schema_fingerprint": FINGERPRINT,
            "disabled_inputs": self.disabled_inputs,
            "elapsed": self._elapsed,
            # On every step, not only at reset, so a run's records can say which start each episode came from.
            "start": self._start_kind,
            "start_frames": self._start_frames,
            "start_cell": None if self._start is None else cell_of(self._start.position),
            "start_problem": self._start_problem,
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
                "room": (observation.room, CANONICAL_START["room"]),
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

    @staticmethod
    def _player_facts(state: dict | None, extras: dict | None) -> dict | None:
        """Raw position, speed and dashes for the run's records (Codex J9, K8).

        These go in `info` and never in the observation: the encoder already sees position and speed as scaled
        features, and a run needs the unscaled values to say how far an episode actually got. None when there
        is no player, which is every ending except success and timeout.
        """
        if state is None:
            return None
        position, speed = state["Player"]["Position"], state["Player"]["Speed"]
        bounds = state.get("Level", {}).get("Bounds", {})
        room_x = position["X"] - bounds.get("X", 0)
        room_y = position["Y"] - bounds.get("Y", 0)
        player = extras.get("player", {}) if isinstance(extras, dict) else {}
        return {"x": position["X"], "y": position["Y"], "room_x": room_x, "room_y": room_y,
                "speed_x": speed["X"], "speed_y": speed["Y"], "dashes": player.get("Dashes")}

    def _replay(self, start: Start, task_start_frames: int = 0):
        """Play a start's inputs from the canonical start. Returns (observation, obs, problem); problem is None
        on success and a short reason otherwise.

        These frames are not transitions: no reward is computed, nothing is returned to the learner, and the
        episode's step count begins at the end of them. They do go through the observation builder, so the
        history the policy sees at its first real step is the history it would have had if it had played here
        itself (Codex K5).
        """
        observation, obs = None, None
        for index, line in enumerate(start.lines, start=1):
            applied = apply_disabled(parse_line(line), self._mask)
            elapsed = max(0, index - task_start_frames)
            try:
                observation = self.bridge.step(*to_parts(applied))
                ending = classify(observation.events, observation.state, elapsed, self.task)
                obs = self._builder.step(observation.state, observation.extras, applied, elapsed)
            except (EndingFault, SchemaViolation) as error:
                return observation, obs, f"frame {index} of the prefix could not be interpreted: {error}"
            # A later task's canonical recipe crosses earlier room boundaries. Transitions are setup, not an
            # ending, until the replay reaches the declared task start. Death, restart, leaving the level and
            # exhausting the deadline still invalidate the recipe.
            setup_transition = index <= task_start_frames and ending in (SUCCESS, WRONG_ROOM)
            if ending is not None and not setup_transition:
                return observation, obs, f"the prefix ended the episode at frame {index} ({ending})"
        facts = self._player_facts(observation.state, observation.extras)
        if facts is None:
            return observation, obs, "the prefix ended with no player"
        arrived = (facts["x"], facts["y"])
        if arrived != tuple(start.position):
            return observation, obs, f"the prefix arrived at {arrived}, not {tuple(start.position)}"
        if observation.room != start.room:
            return observation, obs, f"the prefix arrived in room {observation.room!r}, not {start.room!r}"
        if start.dashes is not None and facts["dashes"] != start.dashes:
            return observation, obs, f"the prefix arrived with {facts['dashes']} dashes, not {start.dashes}"
        return observation, obs, None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ready = False
        options = options or {}
        sampled = options.get("start")
        if sampled is None and self.start_sampler is not None and not options.get("canonical"):
            sampled = self.start_sampler()
        start = sampled if sampled is not None else self.task_start
        base_frames = self.task_start.frames if self.task_start is not None else 0
        problem = None
        try:
            observation = self.bridge.reset()
            self._check_start(observation)
            obs = self._builder.reset(observation.state, observation.extras)
            if start is not None:
                replayed, replayed_obs, problem = self._replay(start, base_frames)
                if problem is None:
                    observation, obs = replayed, replayed_obs
                else:
                    if sampled is None:
                        raise BridgeFault(f"task start no longer replays: {problem}")
                    # A stale sampled entry is not a bridge fault: the bridge is fine, the recipe is not.
                    # Return to the task start and say so, so the caller can drop the sampled entry.
                    observation = self.bridge.reset()
                    self._check_start(observation)
                    obs = self._builder.reset(observation.state, observation.extras)
                    start = self.task_start
                    if start is not None:
                        replayed, replayed_obs, fallback_problem = self._replay(start, base_frames)
                        if fallback_problem is not None:
                            raise BridgeFault(f"task start no longer replays: {fallback_problem}")
                        observation, obs = replayed, replayed_obs
        except (BridgeError, SchemaViolation) as error:
            raise BridgeFault(f"reset failed: {error}") from error
        used_sample = sampled if problem is None else None
        self._elapsed = max(0, used_sample.frames - base_frames) if used_sample is not None else 0
        self._lines = list(start.lines) if (start is not None and self.archive is not None) else []
        self._start, self._start_problem = used_sample, problem
        self._start_kind = "archive" if used_sample is not None else "canonical"
        self._start_frames = self._elapsed
        # Counted here rather than when the start was sampled, so an entry that no longer replays is not
        # recorded as having been tried. This assumes the sampler draws from `archive`, which is how a training
        # run is wired; an evaluation sampling held-out starts passes no archive and records nothing.
        if used_sample is not None and self.archive is not None:
            self.archive.record_use(used_sample)
        self._potential_value = self._start_potential(observation.state)
        self._ready = True
        return obs, self._info(frame=observation.tas_frame, reset_events=observation.events,
                               potential=self._potential_value,
                               player=self._player_facts(observation.state, observation.extras))

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
        player = self._player_facts(observation.state, observation.extras)
        if terminated and self._start is not None and self.archive is not None:
            self.archive.record_outcome(self._start, ending == SUCCESS)
        if self.archive is not None:
            self._lines.append(to_line(applied))
            if (not terminated and observation.room == self.task.start_room and player is not None
                    and self.archive.would_keep((player["x"], player["y"]), len(self._lines))):
                self.archive.offer(Start(tuple(self._lines), (player["x"], player["y"]), observation.room,
                                         player["dashes"]))
        info = self._info(ending=ending, events=observation.events, reward_components=components,
                          applied_action=applied, frame=observation.tas_frame, potential=value, player=player)
        return obs, float(sum(components.values())), terminated, False, info

    def _start_potential(self, state: dict | None) -> float:
        """Build the room's progress potential, reusing it while the room is unchanged.

        Built under every reward version, because how far an episode got is a record every run wants (Codex K8)
        and the two versions are only comparable if both report it. Under rew-v1 nothing is paid for it, so a
        potential that cannot be built there is recorded as absent rather than failing the episode; under
        rew-v2 the reward depends on it, so the same failure is a fault.
        """
        try:
            if self._potential is None or not self._potential.matches(state):
                self._potential = RoomPotential(state)
            return self._potential.value(state)
        except (KeyError, TypeError, ValueError) as error:
            if self.reward_config.shaped:
                raise BridgeFault(f"the room's progress potential could not be built: {error}") from error
            self._potential = None
            return 0.0

    def close(self):
        self._ready = False
        self.bridge.close()
