"""Fast bridge: the same reset()/step() as CelesteBridge, over the CelesteRLLockstep mod's socket.

The mod (mod/CelesteRLLockstep) runs the game in lockstep with this client: it plays one input
frame through CelesteTAS, sends back the resulting state, and waits for the next input, instead of
waiting for the 60 Hz game clock. Physics go through the same CelesteTAS playback path as the HTTP
bridge, which scripts/lockstep_check.py compares frame by frame.

Sessions: a session is one connection that has completed a reset. Any ambiguous failure (timeout,
dropped connection, malformed or mismatched reply, an error from the game, or an invalid episode
start) ends the session, because an input may or may not have been applied and the game state is
no longer known. The only way back is reset(), which first resets over HTTP (this works even if
the TAS stopped), then opens a fresh connection and resets again over the socket. Interrupted
episodes must be discarded; a failed step is never retried.
"""
from __future__ import annotations

import json
import socket
import time

from celeste_rl.bridge import (
    BridgeError,
    CelesteBridge,
    Observation,
    StepTiming,
    check_episode_start,
    format_input_line,
)

DEFAULT_LOCKSTEP_PORT = 32280


class LockstepBridge:
    def __init__(self, http_bridge: CelesteBridge, port: int = DEFAULT_LOCKSTEP_PORT, timeout: float = 10.0):
        self.http = http_bridge
        self.port = port
        self.timeout = timeout

        self.episode_id = 0
        self.step_id = 0
        self.last_timing: StepTiming | None = None
        self.reference_start: dict | None = None
        self.failure: str | None = None

        self._socket: socket.socket | None = None
        self._buffer = b""
        # Never reset across sessions, so a late reply from an old session cannot match a new request.
        self._next_request_id = 0
        self._frame = 0

    # Transport

    def _connect(self) -> None:
        self._socket = socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout)
        # Without this, small messages can wait tens of milliseconds to be batched.
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buffer = b""

    def _end_session(self, reason: str) -> BridgeError:
        """Close the transport, forget any partial reply, and record why. Returns the error to raise."""
        self.failure = reason
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer = b""
        return BridgeError(f"Lockstep session ended: {reason}. Discard this episode and call reset().")

    def _request(self, message: dict) -> dict:
        if self._socket is None:
            raise BridgeError(f"No lockstep session ({self.failure or 'not started'}); call reset()")

        self._next_request_id += 1
        request_id = self._next_request_id
        payload = json.dumps({"id": request_id, **message}, separators=(",", ":")) + "\n"
        try:
            self._socket.sendall(payload.encode("utf-8"))
            while b"\n" not in self._buffer:
                chunk = self._socket.recv(1 << 16)
                if not chunk:
                    raise ConnectionError("the game closed the connection")
                self._buffer += chunk
            line, self._buffer = self._buffer.split(b"\n", 1)
            reply = json.loads(line)
        except (OSError, ValueError) as error:  # timeouts and dropped connections are OSError; bad JSON is ValueError
            raise self._end_session(f"{type(error).__name__} during {message['cmd']}: {error}") from error

        # One request is in flight at a time, so any other id means a stale or duplicated reply.
        if not isinstance(reply, dict) or reply.get("id") != request_id:
            raise self._end_session(f"expected reply to request {request_id}, got {str(reply)[:200]}")
        if "error" in reply:
            raise self._end_session(f"game rejected {message['cmd']}: {reply['error']}")
        if "frame" not in reply or "state" not in reply:
            raise self._end_session(f"reply to {message['cmd']} is missing frame or state")
        return reply

    def _observation(self, reply: dict, expected_frame: int) -> Observation:
        if reply["frame"] != expected_frame:
            raise self._end_session(f"expected frame {expected_frame}, game reported {reply['frame']}")
        state = reply["state"]
        self._frame = expected_frame
        return Observation(self.episode_id, self.step_id, expected_frame, (state or {}).get("RoomName", ""), state)

    # Episodes

    def reset(self) -> Observation:
        if self._socket is None:
            # New session: get the game to a known state over HTTP, then connect and reset over the socket.
            self.http.reset()
            try:
                self._connect()
            except OSError as error:
                raise self._end_session(f"could not connect to port {self.port}: {error}") from error

        reply = self._request({"cmd": "reset"})
        self.step_id = 0
        observation = self._observation(reply, self.http.start_frame)
        try:
            self.reference_start = check_episode_start(observation, self.reference_start)
        except BridgeError as error:
            raise self._end_session(str(error)) from error

        self.episode_id += 1
        self.failure = None
        return Observation(self.episode_id, 0, observation.tas_frame, observation.room, observation.state)

    def step(self, buttons: str | set[str] | frozenset[str]) -> Observation:
        # Invalid buttons are rejected before anything is sent, so they do not end the session.
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
        self._buffer = b""
        self.http.close()
