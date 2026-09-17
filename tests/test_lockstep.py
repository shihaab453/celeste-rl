"""Failure handling of the lockstep client, tested against a fake game server (no game needed).

Run from the repo root:
    .venv-rl/Scripts/python.exe -m unittest
"""
import json
import socket
import threading
import unittest

from celeste_rl.bridge import BridgeError
from celeste_rl.lockstep import LockstepBridge

START_STATE = {"RoomName": "1", "Player": {"Position": {"X": 19, "Y": 144}}}


class StubHttpBridge:
    """Stands in for the HTTP bridge, which the client uses to reach a known state before connecting."""

    start_frame = 300

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def close(self):
        pass


class FakeGame:
    """Loopback TCP server that answers each request with whatever `respond` returns.

    `respond(request, connection_number)` returns bytes to send, or None to send nothing. After
    sending, the server closes the connection if `respond` set `self.close_after_reply`.
    """

    def __init__(self, respond):
        self.respond = respond
        self.requests = []  # (connection_number, request)
        self.close_after_reply = False
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._connections = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            self._connections += 1
            threading.Thread(target=self._handle, args=(connection, self._connections), daemon=True).start()

    def _handle(self, connection, number):
        with connection, connection.makefile("rb") as reader:
            for line in reader:
                request = json.loads(line)
                self.requests.append((number, request))
                reply = self.respond(request, number)
                if reply is not None:
                    connection.sendall(reply)
                if self.close_after_reply:
                    self.close_after_reply = False
                    return

    def close(self):
        self._listener.close()


EXTRAS = {"player": {"Dashes": 1}, "input_buffers": {"Jump": 0.0}, "level": {"Paused": False}}


def reply(request, frame, state=START_STATE, **extra):
    return (json.dumps({"id": request["id"], "frame": frame, "state": state, **extra}) + "\n").encode()


def normal_game(request, _connection):
    """Resets to frame 300; each step adds one frame. Frame tracking is per request id for simplicity."""
    if request["cmd"] == "reset":
        normal_game.frame = 300
    else:
        normal_game.frame += 1
    return reply(request, normal_game.frame)


class LockstepFailureTests(unittest.TestCase):
    def make(self, respond):
        game = FakeGame(respond)
        self.addCleanup(game.close)
        http = StubHttpBridge()
        bridge = LockstepBridge(http, port=game.port, timeout=0.3)
        self.addCleanup(bridge.close)
        return game, http, bridge

    def test_normal_reset_and_steps(self):
        game, http, bridge = self.make(normal_game)
        self.assertEqual(bridge.reset().tas_frame, 300)
        self.assertEqual(bridge.step("R").tas_frame, 301)
        self.assertEqual(bridge.step("RJ").tas_frame, 302)
        self.assertEqual(http.resets, 1)
        self.assertEqual([r["cmd"] for _, r in game.requests], ["reset", "step", "step"])
        self.assertEqual(game.requests[0][1]["start_frame"], 300)

    def test_reset_with_no_player_is_rejected_and_not_used_as_reference(self):
        game, _, bridge = self.make(lambda request, _: reply(request, 300, state=None))
        with self.assertRaises(BridgeError):
            bridge.reset()
        self.assertIsNone(bridge.reference_start)
        # A second no-player reset must fail too, not "match" the first.
        with self.assertRaises(BridgeError):
            bridge.reset()

    def test_step_timeout_ends_session_and_nothing_more_is_sent(self):
        def respond(request, _):
            return None if request["cmd"] == "step" else reply(request, 300)

        game, _, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaises(BridgeError):
            bridge.step("R")
        sent = len(game.requests)
        with self.assertRaises(BridgeError):
            bridge.step("R")
        self.assertEqual(len(game.requests), sent, "a step was sent on an ended session")

    def test_reset_after_failure_uses_http_and_a_fresh_connection(self):
        def respond(request, connection):
            if connection == 1 and request["cmd"] == "step":
                return None  # time out on the first connection
            return normal_game(request, connection)

        game, http, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaises(BridgeError):
            bridge.step("R")
        observation = bridge.reset()
        self.assertEqual(observation.tas_frame, 300)
        self.assertEqual(http.resets, 2)
        self.assertEqual(game.requests[-1][0], 2, "the reset after a failure must use a new connection")
        self.assertEqual(bridge.step("L").tas_frame, 301)

    def test_reply_to_another_request_ends_session(self):
        def respond(request, _):
            if request["cmd"] == "step":
                return reply({"id": request["id"] - 1}, 301)
            return reply(request, 300)

        _, _, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaises(BridgeError):
            bridge.step("R")
        self.assertIsNotNone(bridge.failure)

    def test_error_reply_ends_session(self):
        def respond(request, _):
            if request["cmd"] == "step":
                return (json.dumps({"id": request["id"], "error": "boom"}) + "\n").encode()
            return reply(request, 300)

        _, _, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaisesRegex(BridgeError, "boom"):
            bridge.step("R")
        with self.assertRaisesRegex(BridgeError, "call reset"):
            bridge.step("R")

    def test_malformed_reply_shapes_end_session_without_advancing(self):
        malformed = [
            {"frame": 301, "state": ["bad-shape"]},
            {"frame": "301", "state": START_STATE},
            {"frame": True, "state": START_STATE},
            {"frame": 301, "state": {"RoomName": 1, "Player": {}}},
            {"frame": 301, "state": {"RoomName": "1", "Player": [1, 2]}},
            {"frame": 301, "state": START_STATE, "diagnostics": "loading"},
            {"frame": 301},
            {"frame": 301, "state": START_STATE, "extras": [], "events": []},
            {"frame": 301, "state": START_STATE, "extras": {"player": {}}, "events": []},
            {"frame": 301, "state": START_STATE, "extras": EXTRAS},
            {"frame": 301, "state": START_STATE, "events": []},
            {"frame": 301, "state": START_STATE, "extras": EXTRAS, "events": {"type": "death"}},
            {"frame": 301, "state": START_STATE, "extras": EXTRAS, "events": [{"room": "1"}]},
        ]
        for body in malformed:
            with self.subTest(body=body):
                def respond(request, _, body=body):
                    if request["cmd"] == "reset":
                        return reply(request, 300)
                    return (json.dumps({"id": request["id"], **body}) + "\n").encode()

                _, _, bridge = self.make(respond)
                bridge.reset()
                with self.assertRaisesRegex(BridgeError, "malformed reply"):
                    bridge.step("R")
                self.assertIsNotNone(bridge.failure)
                self.assertEqual(bridge._frame, 300, "local frame advanced on a malformed reply")

    def test_wrong_frame_ends_session(self):
        def respond(request, _):
            return reply(request, 300 if request["cmd"] == "reset" else 305)

        _, _, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaisesRegex(BridgeError, "expected frame 301"):
            bridge.step("R")

    def test_partial_reply_is_not_carried_into_the_next_session(self):
        game = None

        def respond(request, connection):
            if connection == 1 and request["cmd"] == "step":
                game.close_after_reply = True
                return b'{"id": 999, "frame": 3'  # cut off mid-message, then the connection closes
            return normal_game(request, connection)

        game, _, bridge = self.make(respond)
        bridge.reset()
        with self.assertRaises(BridgeError):
            bridge.step("R")
        self.assertEqual(bridge.reset().tas_frame, 300)
        self.assertEqual(bridge.step("R").tas_frame, 301)

    def test_invalid_buttons_are_rejected_locally_without_ending_session(self):
        game, _, bridge = self.make(normal_game)
        bridge.reset()
        with self.assertRaises(ValueError):
            bridge.step("F")
        self.assertIsNone(bridge.failure)
        self.assertEqual(bridge.step("R").tas_frame, 301)
        self.assertEqual(len(game.requests), 2)


    def test_extras_and_events_reach_the_observation(self):
        events = [{"type": "transition", "from": "1", "to": "2"}, {"type": "load_level", "room": "2"}]

        def respond(request, _):
            if request["cmd"] == "reset":
                return reply(request, 300, extras=EXTRAS, events=[{"type": "load_level", "room": "1"}])
            return reply(request, 301, extras=None, events=events)

        _, _, bridge = self.make(respond)
        start = bridge.reset()
        self.assertEqual(start.extras, EXTRAS)
        self.assertEqual(start.events, [{"type": "load_level", "room": "1"}])
        step = bridge.step("R")
        self.assertIsNone(step.extras)
        self.assertEqual(step.events, events)

    def test_query_solids(self):
        def respond(request, _):
            if request["cmd"] == "query_solids":
                return (json.dumps({"id": request["id"], "solids": [True, False][:len(request["rects"])]}) + "\n").encode()
            return reply(request, 300)

        _, _, bridge = self.make(respond)
        bridge.reset()
        self.assertEqual(bridge.query_solids([(0, 0, 8, 8), (40, 40, 8, 8)]), [True, False])
        # A reply of the wrong length ends the session.
        with self.assertRaisesRegex(BridgeError, "malformed query_solids"):
            bridge.query_solids([(0, 0, 8, 8), (1, 1, 1, 1), (2, 2, 2, 2)])
        with self.assertRaisesRegex(BridgeError, "call reset"):
            bridge.step("R")


if __name__ == "__main__":
    unittest.main()
