"""Tests for the parts of the bridge that do not need the game running.

Run from the repo root:
    .venv-rl/Scripts/python.exe -m unittest
"""
import random
import re
import tempfile
import unittest
from pathlib import Path

from celeste_rl.bridge import (
    BUTTON_LETTERS,
    DIRECTIONS,
    BridgeError,
    CelesteBridge,
    Observation,
    TasInfo,
    format_input_line,
    parse_info,
)

# Body of a real /tas/info response captured at the title screen (styles and header trimmed).
TITLE_SCREEN_INFO_HTML = (
    '<div id="main">\r\nRunning: False<br />State: Disabled<br />SaveState Lines: <br />'
    "CurrentFrame: 0<br />TotalFrames: 0<br />RoomName: <br />ChapterTime: <br />Game Info: <br />"
    "<pre>OuiTitleScreen \n\nWind: &lt;Not found&gt;</pre>"
)
# The same format with the values the bridge sees while paused at an episode start.
INFO_HTML = (
    '<div id="main">\r\nRunning: True<br />State: Paused<br />SaveState Lines: 3<br />'
    "CurrentFrame: 300<br />TotalFrames: 300<br />RoomName: 1<br />ChapterTime: 0:03.587(211)<br />Game Info: <br />"
    "<pre>Pos: 19.00, 144.00</pre>"
)


class FormatInputLineTests(unittest.TestCase):
    def test_no_buttons_is_a_neutral_frame(self):
        self.assertEqual(format_input_line(""), "1")

    def test_buttons_are_written_in_a_fixed_order(self):
        self.assertEqual(format_input_line({"J", "R"}), "1,R,J")
        self.assertEqual(format_input_line("XRU"), "1,R,U,X")

    def test_crouch_dash_is_allowed(self):
        self.assertEqual(format_input_line({"Z", "L"}), "1,L,Z")

    def test_dash_only_and_move_only_directions(self):
        self.assertEqual(format_input_line("RX", dash_only="U"), "1,R,X,AU")
        self.assertEqual(format_input_line("", move_only="UR"), "1,MRU")
        self.assertEqual(format_input_line("X", dash_only="RU", move_only="L"), "1,X,ARU,ML")

    def test_rejects_invalid_directions(self):
        for bad in ["X", "A", "u", ",", "RR,Set"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    format_input_line("", dash_only=bad)
                with self.assertRaises(ValueError):
                    format_input_line("", move_only=bad)

    def test_rejects_letters_that_are_not_buttons(self):
        for bad in ["F", "P", "A", "M", "r", ",", "\n", "Set"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                format_input_line(bad)


class ModInputPatternTests(unittest.TestCase):
    """The lockstep mod validates input lines with its own regex; it must accept exactly what Python writes."""

    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).resolve().parents[1] / "mod" / "CelesteRLLockstep" / "Source" / "LockstepDriver.cs").read_text()
        match = re.search(r'InputLine = new\(@"(.+?)", RegexOptions', source)
        # The mod calls Regex.IsMatch, which searches; the pattern must anchor itself. .NET's \z (absolute
        # end) is \Z in Python before 3.14.
        cls.pattern = re.compile(match[1].replace(r"\z", r"\Z"))

    def test_accepts_every_line_python_writes(self):
        rng = random.Random(0)
        for _ in range(5000):
            buttons = "".join(b for b in BUTTON_LETTERS if rng.random() < 0.25)
            dash = "".join(d for d in DIRECTIONS if rng.random() < 0.3)
            move = "".join(d for d in DIRECTIONS if rng.random() < 0.3)
            line = format_input_line(buttons, dash, move)
            self.assertRegex(line, self.pattern)

    def test_rejects_anything_else(self):
        for bad in ["", "2,R", "1,R,", "1,A", "1,M", "1,P", "1,F,90", "1,AU,R", "1,R,MU,AU", "1,R\nSet Player.X 5",
                    "1,Set", "1,AX", "1,r", " 1,R", "1,R,J;", "1,R" + chr(10)]:
            with self.subTest(bad=bad):
                self.assertIsNone(self.pattern.search(bad))


class ParseInfoTests(unittest.TestCase):
    def test_parses_real_title_screen_response(self):
        self.assertEqual(parse_info(TITLE_SCREEN_INFO_HTML), TasInfo(False, "Disabled", 0, 0, ""))

    def test_parses_paused_response(self):
        self.assertEqual(parse_info(INFO_HTML), TasInfo(True, "Paused", 300, 300, "1"))

    def test_missing_field_is_an_error(self):
        with self.assertRaises(BridgeError):
            parse_info("<html>Running: True<br /></html>")

    def test_paused_at_end_requires_no_unplayed_inputs(self):
        self.assertTrue(TasInfo(True, "Paused", 300, 300, "1").paused_at_end)
        self.assertFalse(TasInfo(True, "Paused", 299, 300, "1").paused_at_end)
        self.assertFalse(TasInfo(True, "Running", 300, 300, "1").paused_at_end)
        self.assertFalse(TasInfo(False, "Paused", 300, 300, "1").paused_at_end)


class TasFileTests(unittest.TestCase):
    def test_file_is_prefix_plus_one_line_per_action(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = CelesteBridge(Path(directory) / "episode.tas", level="1", warmup_frames=300)
            bridge._actions = [format_input_line("R"), format_input_line("RJ")]
            bridge._write_tas()
            lines = bridge.tas_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["console load 1", "299", "***S", "1", "1,R", "1,R,J"])
        self.assertEqual(bridge.expected_frame, 302)

    def test_step_requires_reset_first(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = CelesteBridge(Path(directory) / "episode.tas")
            with self.assertRaises(BridgeError):
                bridge.step("R")


class ResetPathTests(unittest.TestCase):
    """Which way reset() restores the episode start, depending on where the game is."""

    class Client:
        def __init__(self, info):
            self._info, self.calls = info, []

        def info(self):
            return self._info

        def play_tas(self, path):
            self.calls.append("play_tas")

        def send_hotkey(self, name):
            self.calls.append(name)

    def reset_path(self, info: TasInfo) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            client = self.Client(info)
            bridge = CelesteBridge(Path(directory) / "episode.tas", client=client)
            start = Observation(1, 0, 300, "1", {"Player": {"Position": {"X": 19, "Y": 144}}, "RoomName": "1"})
            bridge._wait_for = lambda *args, **kwargs: info
            bridge._play_from_level_load = lambda: client.calls.append("play_tas")
            bridge._request_until = lambda *args, **kwargs: (info, 1)
            bridge._read_observation = lambda frame: start
            bridge.reset()
            return client.calls

    def test_running_in_a_level_uses_the_restart_hotkey(self):
        self.assertEqual(self.reset_path(TasInfo(True, "Paused", 400, 400, "1")), ["Restart"])

    def test_stopped_tas_plays_the_file_from_the_level_load(self):
        self.assertEqual(self.reset_path(TasInfo(False, "Disabled", 0, 0, "")), ["play_tas"])

    def test_running_outside_a_level_plays_the_file_from_the_level_load(self):
        # After returning to the map the TAS can still report running, but the Restart hotkey cannot load the
        # level savestate there.
        self.assertEqual(self.reset_path(TasInfo(True, "Paused", 400, 400, "")), ["play_tas"])


class PlayFromLevelLoadTests(unittest.TestCase):
    """Replaying the file once more when CelesteTAS disables a playback that has just started (seen after
    returning to the map), and never more than the attempt limit."""

    class Client:
        def __init__(self, sequences):
            self.sequences, self.plays, self.current = sequences, 0, iter(())
            self.last = TasInfo(True, "Paused", 356, 356, "")

        def play_tas(self, path):
            self.current = iter(self.sequences[self.plays])
            self.plays += 1

        def info(self):
            self.last = next(self.current, self.last)
            return self.last

    STARTED = TasInfo(True, "Running", 0, 300, "")
    STOPPED = TasInfo(False, "Disabled", 0, 300, "")
    BREAKPOINT = TasInfo(True, "Paused", 299, 300, "1")

    def bridge(self, sequences):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        client = self.Client(sequences)
        return CelesteBridge(Path(directory.name) / "episode.tas", client=client), client

    def test_playback_disabled_after_starting_is_played_again(self):
        bridge, client = self.bridge([[self.STARTED, self.STOPPED], [self.STOPPED, self.STARTED, self.BREAKPOINT]])
        bridge._play_from_level_load(timeout=1)
        self.assertEqual(client.plays, 2)

    def test_first_playback_reaching_the_breakpoint_is_not_repeated(self):
        bridge, client = self.bridge([[self.STARTED, self.BREAKPOINT]])
        bridge._play_from_level_load(timeout=1)
        self.assertEqual(client.plays, 1)

    def test_stopping_every_time_raises_after_the_attempt_limit(self):
        bridge, client = self.bridge([[self.STARTED, self.STOPPED]] * 3)
        with self.assertRaisesRegex(BridgeError, "stopped before the savestate breakpoint 2 times"):
            bridge._play_from_level_load(timeout=1)
        self.assertEqual(client.plays, 2)

    def test_a_disabled_state_before_playback_starts_is_not_mistaken_for_a_stop(self):
        bridge, client = self.bridge([[self.STOPPED, self.STOPPED, self.STARTED, self.BREAKPOINT]])
        bridge._play_from_level_load(timeout=1)
        self.assertEqual(client.plays, 1)


class ObservationTests(unittest.TestCase):
    def test_transitional_flag(self):
        level = {"loading": False, "freeze_timer": 0, "scene": "Celeste.Level"}
        self.assertFalse(Observation(1, 0, 300, "1", {}, level).transitional)
        self.assertTrue(Observation(1, 0, 300, "1", None, {**level, "loading": True}).transitional)
        self.assertTrue(Observation(1, 0, 300, "", None, {**level, "scene": "Celeste.LevelExit"}).transitional)
        self.assertIsNone(Observation(1, 0, 300, "1", {}).transitional)


if __name__ == "__main__":
    unittest.main()
