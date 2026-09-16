"""Python side of the Celeste game bridge.

The bridge talks to CelesteTAS through Everest's DebugRC HTTP server. CelesteTAS has no
endpoint that accepts game inputs, so inputs are delivered by rewriting the TAS file it is
playing. The file always ends exactly at the frame the game is paused on:

- To act, the bridge appends one input line and asks CelesteTAS to advance one frame.
- If CelesteTAS has not re-read the file yet, it refuses to advance past the end of the file
  ("Cannot advance further: Reached end-of-file"), so the bridge simply asks again.

There is never a spare line in the file, so an input cannot be played on the wrong frame or
played twice. Every observation is read while the game is paused, between two info reads that
agree on the frame number.

This module only moves inputs and observations. Rewards, termination rules and observation
features belong to the environment layer built on top of it.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = 32279

# One letter per game button, as written in TAS files. These are the inputs a player can bind
# in the game's controls menu. Syntax that needs extra arguments (feather angle 'F', dash-only
# 'A' and move-only 'M' directions) or that only exists in TAS tooling ('P') is not accepted,
# so the bridge can never write a TAS command such as Set, Invoke or console into the file.
BUTTON_LETTERS = {
    "L": "left",
    "R": "right",
    "U": "up",
    "D": "down",
    "J": "jump",
    "K": "jump (second binding)",
    "X": "dash",
    "C": "dash (second binding)",
    "Z": "crouch dash",
    "V": "crouch dash (second binding)",
    "G": "grab",
    "H": "grab (second binding)",
    "S": "pause",
    "Q": "quick restart",
    "N": "journal",
    "O": "confirm",
}
_LETTER_ORDER = {letter: i for i, letter in enumerate(BUTTON_LETTERS)}


class BridgeError(RuntimeError):
    """The game did not behave the way the bridge requires."""


def format_input_line(buttons: str | set[str] | frozenset[str]) -> str:
    """Turn a set of held buttons into one TAS input line lasting exactly one frame.

    >>> format_input_line({"R", "J"})
    '1,R,J'
    >>> format_input_line("")
    '1'
    """
    letters = set(buttons)
    unknown = letters - BUTTON_LETTERS.keys()
    if unknown:
        raise ValueError(f"Not a game button: {sorted(unknown)}. Allowed: {''.join(BUTTON_LETTERS)}")
    return ",".join(["1", *sorted(letters, key=_LETTER_ORDER.__getitem__)])


@dataclass(frozen=True)
class TasInfo:
    """The fields of /tas/info that the bridge relies on."""

    running: bool
    state: str
    current_frame: int
    total_frames: int
    room: str

    @property
    def paused_at_end(self) -> bool:
        """Paused with no unplayed inputs left, which is the only state the bridge acts from."""
        return self.running and self.state == "Paused" and self.current_frame == self.total_frames


def parse_info(html: str) -> TasInfo:
    """Parse the HTML page returned by /tas/info. Each field is written as 'Key: value<br />'."""

    def field(name: str) -> str:
        match = re.search(rf"{name}: (.*?)<br />", html)
        if match is None:
            raise BridgeError(f"/tas/info did not contain {name!r}")
        return match[1]

    return TasInfo(
        running=field("Running") == "True",
        state=field("State"),
        current_frame=int(field("CurrentFrame")),
        total_frames=int(field("TotalFrames")),
        room=field("RoomName"),
    )


class DebugRcClient:
    """Minimal HTTP client for Everest's DebugRC server, reusing one connection."""

    def __init__(self, port: int = DEFAULT_PORT, timeout: float = 10.0):
        self.port = port
        self.timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def get(self, path: str, retry: bool = True) -> str:
        """GET `path`. Only requests that merely read may retry: if a hotkey request fails after being
        sent, the game may already have acted on it, so resending could press it twice."""
        for attempt in range(2 if retry else 1):
            if self._connection is None:
                self._connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
            try:
                self._connection.request("GET", path, headers={"Host": f"localhost:{self.port}"})
                response = self._connection.getresponse()
                body = response.read().decode("utf-8")
            except (ConnectionError, http.client.HTTPException, OSError):
                self.close()
                if not retry or attempt == 1:
                    raise
                continue
            if response.status != 200:
                raise BridgeError(f"GET {path} returned {response.status}: {body[:200]}")
            return body
        raise AssertionError("unreachable")

    def info(self, attempts: int = 20) -> TasInfo:
        # CelesteTAS builds this page on the HTTP thread while the game thread may be changing entity
        # lists (seen during level loading), which makes the request fail with an error page. Such
        # failures are transient, so retry briefly.
        for attempt in range(attempts):
            try:
                return parse_info(self.get("/tas/info"))
            except BridgeError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.005)
        raise AssertionError("unreachable")

    def game_state(self) -> dict | None:
        text = self.get("/tas/game_state")
        return json.loads(text) if text.strip() else None

    def send_hotkey(self, hotkey_id: str) -> None:
        self.get(f"/tas/sendhotkey?id={urllib.parse.quote(hotkey_id)}", retry=False)

    def play_tas(self, path: Path) -> None:
        self.get("/tas/playtas?filePath=" + urllib.parse.quote(str(path)), retry=False)

    def is_available(self) -> bool:
        try:
            self.info()
            return True
        except (OSError, http.client.HTTPException, BridgeError):
            return False

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


@dataclass(frozen=True)
class Observation:
    """Raw game state read at one paused frame, stamped so stale replies can be rejected."""

    episode_id: int
    step_id: int
    tas_frame: int
    room: str
    state: dict | None


def check_episode_start(observation: Observation, reference: dict | None) -> dict:
    """Validate an episode start and return the reference to compare later starts against.

    Every start must have a player in a named room, and must match the first episode's start exactly.
    Without the first two checks, a failed reset that returns no state would become the reference,
    and every later failed reset would then "match" it.
    """
    state = observation.state
    if state is None or state.get("Player") is None:
        raise BridgeError(f"No player at the episode start (frame {observation.tas_frame})")
    if not observation.room:
        raise BridgeError(f"No room name at the episode start (frame {observation.tas_frame})")
    start = {"room": observation.room, "player": state["Player"]}
    if reference is not None and start != reference:
        raise BridgeError(f"Episode start differs from the first episode: {start} != {reference}")
    return start


@dataclass
class StepTiming:
    """How long one step took and how many FrameAdvance requests it sent. More than one means no
    frame appeared within the resend interval, typically because CelesteTAS had not re-read the file yet."""

    total_ms: float
    advance_requests: int


class CelesteBridge:
    """Deliver one frame of inputs at a time to Celeste and read back the resulting state.

    The TAS file starts with a fixed prefix: `console load <level>`, neutral frames until the
    intro has finished, a savestate breakpoint (***S) one frame before the episode start, and
    one more neutral frame. A reset restores the savestate, which pauses on the breakpoint frame,
    then advances that last prefix frame. Pausing on the breakpoint frame is the signal that the
    restore really happened, because the bridge is never otherwise paused there.
    """

    def __init__(
        self,
        tas_path: Path,
        level: str = "1",
        warmup_frames: int = 300,
        client: DebugRcClient | None = None,
        step_timeout: float = 5.0,
        resend_interval: float = 0.1,
    ):
        self.tas_path = Path(tas_path).resolve()
        self.level = level
        self.warmup_frames = warmup_frames
        self.client = client or DebugRcClient()
        self.step_timeout = step_timeout
        self.resend_interval = resend_interval

        self.episode_id = 0
        self.step_id = 0
        self.last_timing: StepTiming | None = None
        self.reference_start: dict | None = None
        self._actions: list[str] = []
        self._started = False

    # File handling

    def _prefix_lines(self) -> list[str]:
        # Load the level, let the intro finish with no inputs, savestate one frame before the
        # episode start, then play one last neutral frame.
        return [f"console load {self.level}", str(self.breakpoint_frame), "***S", "1"]

    def _write_tas(self) -> None:
        """Write the whole file atomically so CelesteTAS never reads a half-written line."""
        text = "\n".join(self._prefix_lines() + self._actions) + "\n"
        self.tas_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.tas_path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        # CelesteTAS may have the file open while re-reading it; Windows then refuses the swap.
        deadline = time.perf_counter() + 1.0
        while True:
            try:
                os.replace(temporary, self.tas_path)
                return
            except PermissionError:
                if time.perf_counter() > deadline:
                    raise
                time.sleep(0.002)

    # Waiting

    def _wait_for(self, predicate, description: str, timeout: float) -> TasInfo:
        deadline = time.perf_counter() + timeout
        while True:
            info = self.client.info()
            if predicate(info):
                return info
            if time.perf_counter() > deadline:
                raise BridgeError(f"Timed out waiting for {description}; last info: {info}")
            time.sleep(0.0005)

    def _read_observation(self, expected_frame: int) -> Observation:
        """Read state between two info reads that agree the game is still paused on this frame."""
        before = self.client.info()
        state = self.client.game_state()
        after = self.client.info()
        for info in (before, after):
            if not info.paused_at_end or info.current_frame != expected_frame:
                raise BridgeError(f"Game moved while reading frame {expected_frame}: {before} / {after}")
        return Observation(self.episode_id, self.step_id, expected_frame, after.room, state)

    # Episodes

    @property
    def start_frame(self) -> int:
        return self.warmup_frames

    @property
    def breakpoint_frame(self) -> int:
        return self.warmup_frames - 1

    @property
    def expected_frame(self) -> int:
        return self.start_frame + len(self._actions)

    def _paused_on_breakpoint(self, info: TasInfo) -> bool:
        return (
            info.running
            and info.state == "Paused"
            and info.current_frame == self.breakpoint_frame
            and info.total_frames == self.start_frame
        )

    def reset(self) -> Observation:
        """Return to the savestated episode start and read the first observation."""
        self._actions = []
        self._write_tas()

        if not self.client.info().running:
            # First episode, or the TAS was stopped: play the file from the level load. This
            # creates the savestate, or loads it if this game session already has one.
            self.client.play_tas(self.tas_path)
            self._wait_for(self._paused_on_breakpoint, "playback to reach the savestate breakpoint", timeout=60.0)
        else:
            # The previous episode's inputs must be gone from CelesteTAS's copy of the file before
            # restarting, or they would be replayed. Advancing at the end of the file forces a
            # re-read and never plays anything, so repeat it until the shorter file is visible.
            self._request_until(
                lambda i: i.total_frames == self.start_frame,
                "CelesteTAS to re-read the truncated file",
            )
            self.client.send_hotkey("Restart")
            self._wait_for(self._paused_on_breakpoint, "the savestate to load", timeout=self.step_timeout)

        # Play the fixed neutral frame from the breakpoint to the episode start.
        self._request_until(
            lambda i: i.paused_at_end and i.current_frame == self.start_frame,
            "the episode start frame",
        )
        self._started = True
        self.episode_id += 1
        self.step_id = 0
        observation = self._read_observation(self.start_frame)
        self.reference_start = check_episode_start(observation, self.reference_start)
        return observation

    def _request_until(self, done, description: str) -> tuple[TasInfo, int]:
        """Send FrameAdvance until `done` holds, resending if CelesteTAS refused an earlier one."""
        requests = 0
        deadline = time.perf_counter() + self.step_timeout
        while True:
            self.client.send_hotkey("FrameAdvance")
            requests += 1
            resend_at = time.perf_counter() + self.resend_interval
            while time.perf_counter() < resend_at:
                info = self.client.info()
                if done(info):
                    return info, requests
                time.sleep(0.0005)
            if time.perf_counter() > deadline:
                raise BridgeError(f"Timed out waiting for {description} after {requests} requests; last info: {info}")

    def step(self, buttons: str | set[str] | frozenset[str]) -> Observation:
        """Hold `buttons` for exactly one frame and return the state after that frame."""
        if not self._started:
            raise BridgeError("Call reset() before step()")
        line = format_input_line(buttons)

        start = time.perf_counter()
        frame = self.expected_frame
        current = self.client.info()
        if not current.paused_at_end or current.current_frame != frame:
            raise BridgeError(f"Expected to be paused at the end of the file on frame {frame}, got {current}")

        self._actions.append(line)
        self._write_tas()
        _, requests = self._request_until(
            lambda i: i.current_frame == frame + 1 and i.total_frames == frame + 1 and i.state == "Paused",
            f"frame {frame + 1}",
        )

        self.step_id += 1
        observation = self._read_observation(frame + 1)
        self.last_timing = StepTiming((time.perf_counter() - start) * 1000, requests)
        return observation

    def close(self) -> None:
        self.client.close()
