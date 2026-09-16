"""Tests for comparing bridge traces with CelesteTAS ExportGameInfo output. No game needed.

Run from the repo root:
    .venv-rl/Scripts/python.exe -m unittest
"""
import struct
import unittest

from celeste_rl.export_compare import ExportFormatError, compare_trace_with_export, f32, parse_export

HEADER = "Line\tInputs\tFrames\tTime\tPosition\tSpeed\tState\tStatuses\tRoom\tEntities"


def player_state(x, remainder_x, speed_x, time, room="1", state_name="StNormal"):
    return {
        "ChapterTime": time, "PlayerStateName": state_name, "RoomName": room,
        "Player": {"Position": {"X": x, "Y": 144}, "PositionRemainder": {"X": remainder_x, "Y": 0},
                   "Speed": {"X": speed_x, "Y": 0}},
    }


# Start frame 300, three action frames with a player, then one death frame without.
TRACE = [
    {"frame": 300, "state": player_state(19, 0.0, 0.0, "0:03.587(211)")},
    {"frame": 301, "state": player_state(19, 0.27777863, 16.666698, "0:03.604(212)")},
    {"frame": 302, "state": player_state(19, 0.8333359, 33.333397, "0:03.621(213)")},
    {"frame": 303, "state": player_state(20, 0.6666718, 50.000095, "0:03.638(214)")},
    {"frame": 304, "state": None},
]


def row(frame, x, remainder, speed, time, room="1"):
    position = x + f32(remainder)
    return f"{frame - 299}\t1/1,R\t{frame}\t{time}\t{position:.12f}, 144.000000000000\t{f32(speed):.12f}, 0.000000000000\tStNormal\t\t[{room}]"


ROWS = [
    row(301, 19, 0.27777863, 16.666698, "0:03.604(212)"),
    row(302, 19, 0.8333359, 33.333397, "0:03.621(213)"),
    row(303, 20, 0.6666718, 50.000095, "0:03.638(214)"),
]


def export(rows):
    return parse_export("\n".join([HEADER, *rows]) + "\n")


class ExportComparisonTests(unittest.TestCase):
    def test_matching_export_passes_with_full_coverage(self):
        result = compare_trace_with_export(TRACE, export(ROWS))
        self.assertTrue(result.passed, result.summary())
        self.assertEqual((result.expected_frames, result.matched), (3, 3))
        self.assertEqual(result.excluded_no_player_frames, [304])
        self.assertEqual(result.excluded_start_frame, 300)

    def test_header_only_export_fails(self):
        result = compare_trace_with_export(TRACE, export([]))
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_frames, [301, 302, 303])

    def test_any_missing_row_fails(self):
        for index in range(len(ROWS)):
            with self.subTest(removed=index):
                result = compare_trace_with_export(TRACE, export(ROWS[:index] + ROWS[index + 1:]))
                self.assertFalse(result.passed)
                self.assertEqual(len(result.missing_frames), 1)

    def test_duplicate_row_is_a_format_error(self):
        with self.assertRaises(ExportFormatError):
            export(ROWS + [ROWS[1]])

    def test_corrupted_row_is_a_format_error(self):
        for corrupt in ["301\t1/1,R\tnot-a-frame\t0:03.604(212)\t19.3, 144\t16.7, 0\tStNormal",
                        "301\t1/1,R\t301\t0:03.604(212)\t19.3\t16.7, 0\tStNormal",
                        "301\t1/1,R"]:
            with self.subTest(corrupt=corrupt), self.assertRaises(ExportFormatError):
                export([corrupt] + ROWS[1:])

    def test_wrong_header_is_a_format_error(self):
        with self.assertRaises(ExportFormatError):
            parse_export("Frames\tTime\n")

    def test_rows_shifted_by_one_frame_fail(self):
        shifted = [r.replace(f"\t{frame}\t", f"\t{frame + 1}\t", 1) for r, frame in zip(ROWS, (301, 302, 303))]
        result = compare_trace_with_export(TRACE, export(shifted))
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_frames, [301])
        self.assertEqual(result.unexpected_frames, [304])

    def test_row_for_a_no_player_frame_fails(self):
        result = compare_trace_with_export(TRACE, export(ROWS + [row(304, 20, 0.5, 60.0, "0:03.655(215)")]))
        self.assertFalse(result.passed)
        self.assertEqual(result.unexpected_frames, [304])

    def test_scene_row_allowed_only_on_a_no_player_frame(self):
        result = compare_trace_with_export(TRACE, export(ROWS + ["5\t1/1\t304\tLevelExit"]))
        self.assertTrue(result.passed, result.summary())
        self.assertEqual(result.scene_rows, {304: "LevelExit"})

        result = compare_trace_with_export(TRACE, export([ROWS[0], "3\t1/1\t302\tLevelExit", ROWS[2]]))
        self.assertFalse(result.passed)
        self.assertEqual([m["frame"] for m in result.mismatched], [302])

    def test_one_float32_step_in_speed_fails(self):
        speed = f32(33.333397)
        bumped = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", speed))[0] + 1))[0]
        rows = [ROWS[0], row(302, 19, 0.8333359, bumped, "0:03.621(213)"), ROWS[2]]
        result = compare_trace_with_export(TRACE, export(rows))
        self.assertFalse(result.passed)
        self.assertEqual([m["frame"] for m in result.mismatched], [302])

    def test_different_time_state_or_room_fails(self):
        for rows in ([ROWS[0], row(302, 19, 0.8333359, 33.333397, "0:03.622(213)"), ROWS[2]],
                     [ROWS[0], ROWS[1].replace("StNormal", "StDash"), ROWS[2]],
                     [ROWS[0], row(302, 19, 0.8333359, 33.333397, "0:03.621(213)", room="2"), ROWS[2]]):
            with self.subTest(rows=rows[1]):
                self.assertFalse(compare_trace_with_export(TRACE, export(rows)).passed)


if __name__ == "__main__":
    unittest.main()
