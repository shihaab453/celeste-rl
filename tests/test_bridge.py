"""Tests for the parts of the bridge that do not need the game running.

Run from the repo root:
    .venv-rl/Scripts/python.exe -m unittest
"""
import tempfile
import unittest
from pathlib import Path

from celeste_rl.bridge import BridgeError, CelesteBridge, Observation, TasInfo, format_input_line, parse_info

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

    def test_rejects_letters_that_are_not_buttons(self):
        for bad in ["F", "P", "A", "M", "r", ",", "\n", "Set"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                format_input_line(bad)


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


class ObservationTests(unittest.TestCase):
    def test_transitional_flag(self):
        level = {"loading": False, "freeze_timer": 0, "scene": "Celeste.Level"}
        self.assertFalse(Observation(1, 0, 300, "1", {}, level).transitional)
        self.assertTrue(Observation(1, 0, 300, "1", None, {**level, "loading": True}).transitional)
        self.assertTrue(Observation(1, 0, 300, "", None, {**level, "scene": "Celeste.LevelExit"}).transitional)
        self.assertIsNone(Observation(1, 0, 300, "1", {}).transitional)


if __name__ == "__main__":
    unittest.main()
