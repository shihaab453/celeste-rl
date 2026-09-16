"""Check that every accepted input actually does what its game binding should, in the real game.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/binding_audit.py

Each check resets to the episode start (standing on the ground, facing right, dash available, a wall to
the left), plays a short input sequence through the lockstep bridge, and looks for the binding's effect
in the game state. Checks marked "observe" have no confident expectation; their results are recorded
for review rather than asserted. The mod's own input validation is also checked with raw lines Python
would never send.

Writes runs/binding-audit/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import DEFAULT_LOCKSTEP_PORT, LockstepBridge  # noqa: E402

Step = tuple[str, str, str]  # buttons, dash_only, move_only


def hold(buttons: str = "", frames: int = 1, dash_only: str = "", move_only: str = "") -> list[Step]:
    return [(buttons, dash_only, move_only)] * frames


def play(bridge: LockstepBridge, steps: list[Step]) -> list[dict]:
    bridge.reset()
    frames = []
    for buttons, dash_only, move_only in steps:
        observation = bridge.step(buttons, dash_only, move_only)
        state = observation.state or {}
        player = state.get("Player") or {}
        frames.append({
            "input": ",".join(filter(None, [buttons, dash_only and "A" + dash_only, move_only and "M" + move_only])),
            "state": state.get("PlayerStateName"),
            "speed": (player.get("Speed") or {}),
            "on_ground": player.get("OnGround"),
            "paused": (observation.diagnostics or {}).get("level_paused"),
            "freeze": (observation.diagnostics or {}).get("freeze_timer"),
            "player": bool(player),
        })
    return frames


def first(frames: list[dict], predicate) -> int | None:
    return next((i for i, f in enumerate(frames) if predicate(f)), None)


def dash_vector(frames: list[dict]) -> tuple[float, float] | None:
    """Speed on the first dash frame after the freeze, rounded, or None if no dash happened."""
    index = first(frames, lambda f: f["state"] == "StDash" and not f["freeze"] and (f["speed"].get("X") or f["speed"].get("Y")))
    if index is None:
        return None
    speed = frames[index]["speed"]
    return round(speed.get("X", 0)), round(speed.get("Y", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    args = parser.parse_args()

    output_dir = REPO / "runs" / "binding-audit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    results = {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")), "checks": []}

    def record(name: str, passed: bool | None, detail, frames=None):
        results["checks"].append({"name": name, "passed": passed, "detail": detail, "frames": frames})
        label = {True: "OK     ", False: "FAIL   ", None: "observe"}[passed]
        print(f"  {label} {name}: {detail}")

    process = game_process.launch(args.game_dir, focus=False)
    http = CelesteBridge(output_dir / "episode.tas")
    bridge = LockstepBridge(http)
    try:
        print("Movement and jump")
        for letter, sign in (("R", 1), ("L", -1)):
            frames = play(bridge, hold(letter, 3))
            speed = frames[-1]["speed"].get("X", 0)
            record(f"{letter} moves", speed * sign > 0, f"speed X after 3 frames {speed}", frames)
        for letter in "JK":
            frames = play(bridge, hold(letter, 3))
            record(f"{letter} jumps", frames[0]["speed"].get("Y", 0) < 0 and frames[-1]["on_ground"] is False,
                   f"speed Y {frames[0]['speed'].get('Y')}, on ground after 3 frames {frames[-1]['on_ground']}", frames)
        frames = play(bridge, hold("D", 3))
        record("D (crouch) does not move", frames[-1]["speed"].get("X", 0) == 0, f"state {frames[-1]['state']}", frames)

        print("Dash bindings")
        for letter in "XC":
            frames = play(bridge, hold(letter, 1) + hold("", 10))
            record(f"{letter} dashes (neutral, facing right)", dash_vector(frames) is not None and dash_vector(frames)[0] > 0,
                   f"dash speed {dash_vector(frames)}", frames)
        for letter in "ZV":
            frames = play(bridge, hold(letter, 1) + hold("", 10))
            record(f"{letter} crouch-dashes", dash_vector(frames) is not None, f"dash speed {dash_vector(frames)}", frames)
        frames = play(bridge, hold("UX", 1) + hold("U", 10))
        up = dash_vector(frames)
        record("U+X dashes straight up", up is not None and up[0] == 0 and up[1] < 0, f"dash speed {up}", frames)

        print("Dash-only and move-only directions")
        frames = play(bridge, hold("X", 1, dash_only="U") + hold("", 10, dash_only="U"))
        vector = dash_vector(frames)
        record("dash-only up aims the dash up", vector is not None and vector[0] == 0 and vector[1] < 0, f"dash speed {vector}", frames)
        frames = play(bridge, hold("RX", 1, dash_only="U") + hold("R", 10, dash_only="U"))
        record("right held + dash-only up", None, f"dash speed {dash_vector(frames)} (does dash-only override movement aim?)", frames)
        frames = play(bridge, hold("", 3, move_only="R"))
        speed = frames[-1]["speed"].get("X", 0)
        record("move-only right moves", speed > 0, f"speed X after 3 frames {speed}", frames)
        frames = play(bridge, hold("UX", 1, move_only="R") + hold("U", 10, move_only="R"))
        with_move_only = dash_vector(frames)
        frames_both = play(bridge, hold("RUX", 1) + hold("RU", 10))
        with_right = dash_vector(frames_both)
        record("move-only right does not aim the dash", with_move_only is not None and with_right is not None
               and with_move_only[0] == 0 and with_right[0] > 0,
               f"U+X with move-only right {with_move_only}; U+R+X {with_right}", frames)

        print("Grab")
        for letter in "GH":
            frames = play(bridge, hold("L", 12) + hold("L" + letter, 6))
            record(f"{letter} grabs the left wall", frames[-1]["state"] == "StClimb", f"state {frames[-1]['state']}", frames)

        print("Pause and menu bindings")
        frames = play(bridge, hold("", 2) + hold("S", 1) + hold("", 20))
        paused_at = first(frames, lambda f: f["paused"])
        record("S pauses", paused_at is not None, f"paused from step {paused_at}", frames)
        for letter, role in (("J", "confirm"), ("O", "confirm (second binding)"), ("X", "cancel"), ("C", "cancel (second binding)")):
            frames = play(bridge, hold("", 2) + hold("S", 1) + hold("", 40) + hold(letter, 1) + hold("", 40))
            record(f"{letter} as {role} closes the pause menu", frames[42]["paused"] is True and frames[-1]["paused"] is False,
                   f"paused before {frames[42]['paused']}, after {frames[-1]['paused']}", frames)
        frames = play(bridge, hold("", 2) + hold("Q", 1) + hold("", 30))
        record("Q (quick restart)", None,
               f"no-player frames {sum(not f['player'] for f in frames)}, paused frames {sum(bool(f['paused']) for f in frames)}", frames)
        frames = play(bridge, hold("", 2) + hold("N", 1) + hold("", 30))
        record("N (journal and talk) in a level", None,
               f"paused frames {sum(bool(f['paused']) for f in frames)}, final state {frames[-1]['state']}, speed {frames[-1]['speed']}", frames)
        bridge.close()

        print("Mod-side input validation")
        with socket.create_connection(("127.0.0.1", DEFAULT_LOCKSTEP_PORT), timeout=10) as raw:
            reader = raw.makefile("rb")

            def request(message):
                raw.sendall((json.dumps(message) + "\n").encode())
                return json.loads(reader.readline())

            reset = request({"id": 1, "cmd": "reset", "start_frame": http.start_frame})
            record("raw client reset", reset.get("frame") == http.start_frame, str(reset)[:60])
            for i, line in enumerate(["1,A", "1,M", "1,P", "1,F,90", "1,AU,R", "1,R\n", "1,R,MU,AU", "1\nconsole load 2"], start=2):
                reply = request({"id": i, "cmd": "step", "line": line})
                record(f"mod rejects {line!r}", "Rejected input line" in reply.get("error", ""), str(reply)[:90])
            reply = request({"id": 99, "cmd": "step", "line": "1,R,X,AU,MR"})
            record("mod accepts '1,R,X,AU,MR'", reply.get("frame") == http.start_frame + 1, str(reply)[:60])
    finally:
        bridge.close()
        (output_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"Results: {output_dir / 'results.json'}")
        game_process.stop(process)

    asserted = [c for c in results["checks"] if c["passed"] is not None]
    failed = [c["name"] for c in asserted if not c["passed"]]
    print(f"{len(asserted) - len(failed)}/{len(asserted)} asserted checks passed; "
          f"{len(results['checks']) - len(asserted)} observations recorded.")
    if failed:
        print("Failed: " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
