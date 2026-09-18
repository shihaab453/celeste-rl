"""Phase 3B step 1: turn the archives' deepest reached states into verified clears of room 1.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/extract_demonstrations.py

Phase 3 without demonstrations is finished and did not clear the room from the canonical start. Phase 3B is the
demonstration-assisted comparison, and it needs a dataset. This builds one without a human playing and without
authoring a TAS.

**Where the demonstrations come from, which must be stated wherever they are used.** The varied-start runs kept
an archive of states the agent itself reached, each stored as the input lines that reach it from the canonical
start. Several of those sit inside the exit gap at the top of the room. This script replays such a prefix and
then searches a few frames of continuation for one that crosses into room 2. So a demonstration here is
**agent-produced, completed by a short search**: it is not human play, not a TAS, and not derived from the
recorded route fixture. It is the same kind of artefact as `scripts/find_room_exit.py` produces, at scale.

Every candidate is verified by playing it in the game from the canonical start: a sequence is only written out
if the environment reports `success`. Nothing that was not seen to work is kept.

Results go to runs/demonstrations/<timestamp>/: `demonstrations.json` holds each verified clear with its input
lines and provenance, and `results.json` the summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import parse_line, to_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402

# The exit is the open top row of room 1, columns 31 to 35, so x 248 to 288 with y near 0.
EXIT_X = (240, 296)
# Continuations tried after a prefix, in order. The exit is upwards, so these are the ways to keep going up.
CONTINUATIONS = ("U", "", "RU", "LU", "XU", "R", "L", "JU")
CONTINUATION_FRAMES = 12


def candidates(archives: list[Path], limit: int, max_frames: int) -> list[dict]:
    """Archived cells worth trying, nearest the top of the room first, one per (x, y) across all archives."""
    seen, found = set(), []
    for path in archives:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["starts"]:
            x, y = entry["position"]
            if not (EXIT_X[0] <= x <= EXIT_X[1]) or len(entry["lines"]) > max_frames:
                continue
            if (x, y) in seen:
                continue
            seen.add((x, y))
            found.append({"start": Start.from_json(entry), "archive": str(path)})
    found.sort(key=lambda c: (c["start"].position[1], c["start"].frames))
    return found[:limit]


def attempt(env: CelesteRoomEnv, start: Start, held: str) -> dict:
    """Replay a prefix, then hold one input for a few frames. Returns what happened."""
    _, info = env.reset(options={"start": start})
    if info["start"] != "archive":
        return {"ending": None, "problem": info["start_problem"]}
    action = parse_line(f"1,{','.join(held)}" if held else "1")
    lines = list(start.lines)
    for _ in range(CONTINUATION_FRAMES):
        _, _, terminated, _, info = env.step(action)
        lines.append(to_line(info["applied_action"]))
        if terminated:
            return {"ending": info["ending"], "lines": lines, "frames": len(lines), "problem": None}
    return {"ending": None, "problem": "no ending within the continuation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--archives", nargs="*", type=Path,
                        default=sorted((REPO / "runs" / "train").glob("*/archive.json")))
    parser.add_argument("--limit", type=int, default=40, help="candidate cells to try")
    parser.add_argument("--max-frames", type=int, default=400, help="longest prefix worth replaying")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2

    chosen = candidates(list(args.archives), args.limit, args.max_frames)
    if not chosen:
        print("No archived cells near the exit. Run a varied-starts training run first.")
        return 2
    print(f"{len(chosen)} candidate cells from {len(args.archives)} archives, "
          f"y {chosen[0]['start'].position[1]} to {chosen[-1]['start'].position[1]}")

    output_dir = REPO / "runs" / "demonstrations" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http))
    verified, attempts = [], 0
    try:
        manifest = runtime.collect(args.game_dir, http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        for index, candidate in enumerate(chosen, start=1):
            start = candidate["start"]
            for held in CONTINUATIONS:
                attempts += 1
                result = attempt(env, start, held)
                if result["ending"] == "success":
                    verified.append({"frames": result["frames"], "lines": result["lines"],
                                     "held_after_prefix": held, "prefix_frames": start.frames,
                                     "reached": list(start.position), "archive": candidate["archive"]})
                    print(f"  [{index}/{len(chosen)}] clear from ({start.position[0]}, {start.position[1]}): "
                          f"{result['frames']} frames, held {held!r} after a {start.frames} frame prefix")
                    break
            else:
                print(f"  [{index}/{len(chosen)}] no clear from ({start.position[0]}, {start.position[1]})")
    finally:
        env.close()
        game.close()

    (output_dir / "demonstrations.json").write_text(
        json.dumps({"source": "agent-reached archive prefixes completed by a short continuation search",
                    "demonstrations": verified}, indent=1), encoding="utf-8")
    lengths = sorted(d["frames"] for d in verified)
    summary = {**git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
               "archives": [str(a) for a in args.archives], "candidates": len(chosen), "attempts": attempts,
               "verified": len(verified),
               "frames": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]}
               if lengths else None}
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{len(verified)} verified clears from {attempts} attempts"
          + (f", {lengths[0]} to {lengths[-1]} frames" if lengths else ""))
    print(f"Results: {output_dir}")
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
