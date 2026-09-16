"""Compare a fixed room-transition + dash trace across HTTP, lockstep and plain TAS playback.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/transition_check.py --route runs/routes/<timestamp>/route.json

1. HTTP bridge trace (reference for full game state).
2. Lockstep trace, then lockstep again with a client delay of 5, 50 and 500 ms right after the
   transition step. If the game keeps changing while waiting (for example during loading), delayed
   traces differ from the undelayed one.
3. Plain TAS playback at normal speed from the level load, with no savestate and no bridge, exported
   frame by frame with CelesteTAS's ExportGameInfo, compared on time, position, speed, state and room.
4. Per-frame lockstep diagnostics: whether CelesteTAS ever reported loading, and freeze frames.

Writes runs/transition-check/<timestamp>/results.json and traces.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.export_compare import compare_files  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402


def http_trace(bridge: CelesteBridge, actions: list[str]) -> list[dict]:
    start = bridge.reset()
    frames = [{"frame": start.tas_frame, "state": start.state}]
    for buttons in actions:
        observation = bridge.step(buttons)
        frames.append({"frame": observation.tas_frame, "state": observation.state})
    return frames


def lockstep_trace(bridge: LockstepBridge, actions: list[str], delay_after: int | None = None, delay: float = 0) -> list[dict]:
    start = bridge.reset()
    frames = [{"frame": start.tas_frame, "state": start.state, "diagnostics": start.diagnostics}]
    for index, buttons in enumerate(actions):
        observation = bridge.step(buttons)
        frames.append({"frame": observation.tas_frame, "state": observation.state, "diagnostics": observation.diagnostics})
        if index == delay_after:
            time.sleep(delay)
    return frames


def strip_diagnostics(trace: list[dict]) -> list[dict]:
    return [{"frame": entry["frame"], "state": entry["state"]} for entry in trace]


def first_difference(expected: list[dict], actual: list[dict], skip: set[int] = frozenset()) -> dict | None:
    for index, (a, b) in enumerate(zip(expected, actual)):
        if index in skip:
            continue
        if a != b:
            fields = sorted(k for k in set(a["state"] or {}) | set(b["state"] or {})
                            if (a["state"] or {}).get(k) != (b["state"] or {}).get(k))
            return {"step": index, "frame": a["frame"], "differing_fields": fields}
    return None if len(expected) == len(actual) else {"reason": "different lengths"}


def play_plain_tas(client, tas_path: Path, export_path: Path, actions: list[str], warmup_frames: int) -> None:
    """Play the route from the level load at normal speed with no savestate, exporting every frame."""
    lines = ["console load 1", str(warmup_frames), f"StartExportGameInfo {export_path.as_posix()}"]
    lines += [format_input_line(buttons) for buttons in actions]
    lines += ["FinishExportGameInfo"]
    tas_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = warmup_frames + len(actions)

    client.play_tas(tas_path)
    deadline = time.perf_counter() + 120
    # The previous run may already be paused on the same final frame number, so first wait for this
    # playback to start from the level load; otherwise the end condition could match stale state.
    started = False
    while True:
        info = client.info()
        if not started:
            started = info.running and info.current_frame < warmup_frames
        elif info.paused_at_end and info.current_frame == total:
            if not export_path.exists():
                raise RuntimeError("Plain TAS playback finished but wrote no export")
            return
        elif not info.running:
            raise RuntimeError(f"Plain TAS playback stopped early: {info}")
        if time.perf_counter() > deadline:
            raise RuntimeError(f"Plain TAS playback did not finish (started={started}): {info}")
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--label", default="", help="added to the output folder name, e.g. hide-freeze-on")
    args = parser.parse_args()

    route = json.loads(args.route.read_text(encoding="utf-8"))
    actions, transition = route["actions"], route["transition_step"]
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else "")
    output_dir = REPO / "runs" / "transition-check" / name
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    tas_settings = (args.game_dir / "probe-profile" / "Saves" / "modsettings-CelesteTAS.celeste").read_text(encoding="utf-8-sig")
    results = {
        "commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
        "route": str(args.route), "frames": len(actions), "transition_step": transition,
        "settings": {key: re.search(rf"{key}: (\S+)", tas_settings)[1] for key in ("HideFreezeFrames", "AutoPauseDraft")},
    }
    print(f"Route {route['start_room']} -> {route['next_room']}, {len(actions)} frames, transition at step {transition}, "
          f"settings {results['settings']}")

    process = game_process.launch(args.game_dir, focus=False)
    http = CelesteBridge(output_dir / "episode.tas")
    lockstep = LockstepBridge(http)
    traces = {}
    try:
        print("1. HTTP trace")
        traces["http"] = http_trace(http, actions)

        print("2. Lockstep traces")
        traces["lockstep"] = lockstep_trace(lockstep, actions)
        for delay in (0.005, 0.05, 0.5):
            traces[f"lockstep_delay_{int(delay * 1000)}ms"] = lockstep_trace(lockstep, actions, transition, delay)
        lockstep.close()

        # Where lockstep waited for loading to finish before replying, its frame shows the post-loading state
        # while the HTTP bridge shows the mid-loading frame. Those frames are reported, not compared.
        waited = {i for i, entry in enumerate(traces["lockstep"]) if ((entry["diagnostics"] or {}).get("loading_updates") or 0) > 0}
        results["lockstep_waited_for_loading_at"] = sorted(waited)
        comparisons = {"lockstep_vs_http": first_difference(traces["http"], strip_diagnostics(traces["lockstep"]), waited)}
        for key in traces:
            if key.startswith("lockstep_delay"):
                comparisons[f"{key}_vs_lockstep"] = first_difference(strip_diagnostics(traces["lockstep"]), strip_diagnostics(traces[key]))
        results["comparisons"] = comparisons
        for key, difference in comparisons.items():
            print(f"   {key}: {'identical' if difference is None else difference}")

        diagnostics = [entry["diagnostics"] or {} for entry in traces["lockstep"]]
        rooms = [(entry["state"] or {}).get("RoomName") for entry in traces["http"]]
        results["diagnostics"] = {
            "loading_frames": [i for i, d in enumerate(diagnostics) if d.get("loading")],
            "loading_updates": {i: d["loading_updates"] for i, d in enumerate(diagnostics) if d.get("loading_updates")},
            "freeze_frames": [i for i, d in enumerate(diagnostics) if (d.get("freeze_timer") or 0) > 0],
            "scenes": sorted({d.get("scene") for d in diagnostics if d.get("scene")}),
            "no_player_frames": [i for i, e in enumerate(traces["http"]) if e["state"] is None],
            "room_change_at": next((i for i, r in enumerate(rooms) if r and r != rooms[0]), None),
        }
        print(f"   diagnostics: {results['diagnostics']}")

        print("3. Plain TAS playback at normal speed, no savestate")
        export_path = output_dir / "plain-export.txt"
        play_plain_tas(http.client, output_dir / "plain.tas", export_path, actions, http.warmup_frames)
        comparison = compare_files(traces["http"], export_path)
        results["plain_vs_http"] = comparison.summary()
        print(f"   {comparison.matched}/{comparison.expected_frames} expected player frames match; "
              f"{len(comparison.mismatched)} differ, {len(comparison.missing_frames)} missing, "
              f"{len(comparison.unexpected_frames)} unexpected; excluded {len(comparison.excluded_no_player_frames)} "
              f"no-player frames and start frame {comparison.excluded_start_frame}")
        if comparison.mismatched:
            print(f"   first difference: {comparison.mismatched[0]}")
    finally:
        http.close()
        (output_dir / "traces.json").write_text(json.dumps(traces), encoding="utf-8")
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results: {output_dir / 'results.json'}")
        game_process.stop(process)

    passed = all(v is None for v in results.get("comparisons", {"x": 1}).values()) and \
        results.get("plain_vs_http", {}).get("passed", False)
    print("All traces agree." if passed else "TRACES DISAGREE.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
