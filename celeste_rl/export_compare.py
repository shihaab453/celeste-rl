"""Compare a bridge trace with CelesteTAS ExportGameInfo output from plain TAS playback.

The export writes one row per consumed input, after that frame's engine update, labelled with the TAS
frame counter after it advanced, which is the same counter the bridge reports. The frame offset is
therefore fixed at zero rather than searched for. Rows are omitted when the level has no Player, so
the expected rows are exactly the trace's action frames that have a player.

Numeric precision: the export prints the double sum Position + PositionRemainder with its own decimal
formatting; speeds are float32 values printed with many decimals. Positions are compared as that double
sum with a 1e-9 tolerance and speeds as float32. This rejects the one-float32-step changes seen at these
magnitudes, but it is a comparison of exported values, not proof of bitwise equality of every internal
field (for example, position and remainder are not compared separately).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

HEADER = ["Line", "Inputs", "Frames", "Time", "Position", "Speed", "State", "Statuses", "Room", "Entities"]


class ExportFormatError(ValueError):
    """The export file is not in the expected format."""


def f32(value: float) -> float:
    """The float32 value Celeste stores."""
    return struct.unpack("f", struct.pack("f", value))[0]


def parse_export(text: str) -> dict[int, dict]:
    """Rows keyed by TAS frame. Raises ExportFormatError on a bad header, malformed row or duplicate frame."""
    lines = text.splitlines()
    if not lines or lines[0].split("\t") != HEADER:
        raise ExportFormatError(f"unexpected header: {lines[0] if lines else '(empty file)'!r}")
    rows = {}
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) == 4:
            # Outside a Level the export writes only Line, Inputs, Frames and the scene name.
            try:
                frame = int(cells[2])
            except ValueError as error:
                raise ExportFormatError(f"line {number}: {error}: {line[:120]!r}") from error
            if frame in rows:
                raise ExportFormatError(f"line {number}: duplicate row for frame {frame}")
            rows[frame] = {"scene": cells[3]}
            continue
        try:
            if len(cells) < 7:
                raise ValueError(f"only {len(cells)} columns")
            frame = int(cells[2])
            position = [float(v) for v in cells[4].split(",")]
            speed = [float(v) for v in cells[5].split(",")]
            if len(position) != 2 or len(speed) != 2:
                raise ValueError("position and speed need two components")
        except ValueError as error:
            raise ExportFormatError(f"line {number}: {error}: {line[:120]!r}") from error
        if frame in rows:
            raise ExportFormatError(f"line {number}: duplicate row for frame {frame}")
        room = re.search(r"\[([^\]]+)\]", "\t".join(cells[7:]))
        rows[frame] = {"time": cells[3], "position": position, "speed": speed, "state": cells[6],
                       "room": room[1] if room else None}
    return rows


@dataclass
class ExportComparison:
    expected_frames: int = 0
    matched: int = 0
    mismatched: list[dict] = field(default_factory=list)
    missing_frames: list[int] = field(default_factory=list)
    unexpected_frames: list[int] = field(default_factory=list)
    excluded_no_player_frames: list[int] = field(default_factory=list)
    scene_rows: dict[int, str] = field(default_factory=dict)  # no-player frames exported as a non-Level scene
    excluded_start_frame: int | None = None

    @property
    def passed(self) -> bool:
        return (self.expected_frames > 0 and self.matched == self.expected_frames and not self.mismatched
                and not self.missing_frames and not self.unexpected_frames)

    def summary(self) -> dict:
        return {
            "passed": self.passed, "expected_frames": self.expected_frames, "matched": self.matched,
            "mismatched": len(self.mismatched), "first_mismatch": self.mismatched[0] if self.mismatched else None,
            "missing_frames": self.missing_frames, "unexpected_frames": self.unexpected_frames,
            "excluded_no_player_frames": self.excluded_no_player_frames, "scene_rows": self.scene_rows,
            "excluded_start_frame": self.excluded_start_frame,
        }


def compare_trace_with_export(trace: list[dict], export: dict[int, dict]) -> ExportComparison:
    """`trace` is [{"frame", "state"}, ...] starting with the episode start (reset) observation."""
    result = ExportComparison(excluded_start_frame=trace[0]["frame"] if trace else None)
    expected = set()
    allowed_scene_frames = set()
    for entry in trace[1:]:
        if entry["state"] is None or entry["state"].get("Player") is None:
            # No player: the export has no row (Level without a player) or a scene-name row (not a Level).
            result.excluded_no_player_frames.append(entry["frame"])
            row = export.get(entry["frame"])
            if row is not None and "scene" in row:
                result.scene_rows[entry["frame"]] = row["scene"]
                allowed_scene_frames.add(entry["frame"])
            continue
        expected.add(entry["frame"])
        row = export.get(entry["frame"])
        if row is None:
            result.missing_frames.append(entry["frame"])
            continue
        if "scene" in row:
            result.mismatched.append({"frame": entry["frame"], "export": row, "trace": "player present"})
            continue

        state, player = entry["state"], entry["state"]["Player"]
        exact = [player["Position"][axis] + f32(player["PositionRemainder"][axis]) for axis in "XY"]
        speed = [f32(player["Speed"][axis]) for axis in "XY"]
        same = (
            row["time"] == state["ChapterTime"]
            and row["state"] == state["PlayerStateName"]
            and row["room"] == state["RoomName"]
            and all(abs(a - b) < 1e-9 for a, b in zip(row["position"], exact))
            and [f32(v) for v in row["speed"]] == speed
        )
        if same:
            result.matched += 1
        else:
            result.mismatched.append({"frame": entry["frame"], "export": row, "trace": {
                "time": state["ChapterTime"], "position": exact, "speed": speed,
                "state": state["PlayerStateName"], "room": state["RoomName"]}})
    result.expected_frames = len(expected)
    result.unexpected_frames = sorted(set(export) - expected - allowed_scene_frames)
    return result


def compare_files(trace: list[dict], export_path: Path) -> ExportComparison:
    return compare_trace_with_export(trace, parse_export(export_path.read_text(encoding="utf-8")))
