"""Fast bridge: the same reset()/step() as CelesteBridge, over the CelesteRLLockstep mod's socket.

The mod (mod/CelesteRLLockstep) runs the game in lockstep with this client: it plays one input
frame through CelesteTAS, sends back the resulting state, and waits for the next input, instead of
waiting for the 60 Hz game clock. Physics go through the same code path as the HTTP bridge, which
is what scripts/lockstep_check.py verifies frame by frame.

The first reset uses the HTTP bridge, which writes the TAS file, creates the savestate and leaves
the game paused at the episode start. Everything after that goes over the socket.
"""
from __future__ import annotations

import json
import socket
import time

from celeste_rl.bridge import BridgeError, CelesteBridge, Observation, StepTiming, format_input_line

DEFAULT_LOCKSTEP_PORT = 32280


class LockstepBridge:
    def __init__(self, http_bridge: CelesteBridge, port: int = DEFAULT_LOCKSTEP_PORT, timeout: float = 30.0):
        self.http = http_bridge
        self.port = port
        self.timeout = timeout

        self.episode_id = 0
        self.step_id = 0
        self.last_timing: StepTiming | None = None
        self.reference_start: dict | None = None

        self._socket: socket.socket | None = None
        self._buffer = b""
        self._next_request_id = 0
        self._frame = 0

    # Transport

    def _connect(self) -> None:
        self._socket = socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout)
        # Without this, small messages can wait tens of milliseconds to be batched.
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _request(self, message: dict) -> dict:
        self._next_request_id += 1
        request_id = self._next_request_id
        payload = json.dumps({"id": request_id, **message}, separators=(",", ":")) + "\n"
        self._socket.sendall(payload.encode("utf-8"))

        while b"\n" not in self._buffer:
            chunk = self._socket.recv(1 << 16)
            if not chunk:
                raise BridgeError("The game closed the lockstep connection")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        reply = json.loads(line)

        # One request is in flight at a time, so any other id means a stale or duplicated reply.
        if reply.get("id") != request_id:
            raise BridgeError(f"Expected reply to request {request_id}, got {reply.get('id')}")
        if "error" in reply:
            raise BridgeError(f"Game rejected {message}: {reply['error']}")
        return reply

    def _observation(self, reply: dict, expected_frame: int) -> Observation:
        if reply["frame"] != expected_frame:
            raise BridgeError(f"Expected frame {expected_frame}, game reported {reply['frame']}")
        state = reply["state"]
        self._frame = expected_frame
        return Observation(self.episode_id, self.step_id, expected_frame, (state or {}).get("RoomName", ""), state)

    # Episodes

    def reset(self) -> Observation:
        if self._socket is None:
            self.http.reset()
            self._connect()
            reply = self._request({"cmd": "observe"})
        else:
            reply = self._request({"cmd": "reset"})

        self.episode_id += 1
        self.step_id = 0
        observation = self._observation(reply, self.http.start_frame)

        start = {"room": observation.room, "player": (observation.state or {}).get("Player")}
        if self.reference_start is None:
            self.reference_start = start
        elif start != self.reference_start:
            raise BridgeError(f"Episode start differs from the first episode: {start} != {self.reference_start}")
        return observation

    def step(self, buttons: str | set[str] | frozenset[str]) -> Observation:
        if self._socket is None:
            raise BridgeError("Call reset() before step()")
        line = format_input_line(buttons)
        start = time.perf_counter()
        reply = self._request({"cmd": "step", "line": line})
        self.step_id += 1
        observation = self._observation(reply, self._frame + 1)
        self.last_timing = StepTiming((time.perf_counter() - start) * 1000, 1)
        return observation

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.http.close()
