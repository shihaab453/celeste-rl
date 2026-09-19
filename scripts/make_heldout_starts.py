"""Build the frozen set of held-out entry states the real Phase 3 criterion is measured on.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/make_heldout_starts.py --searches 6 --states 200

Every result this project has reported comes from one fixed start in a deterministic game, so a success rate
measures how robust a policy's own sampling is around a single route rather than whether it can play the room.
The roadmap's criterion for the phase is different and much harder: clear the room from **200 held-out
reachable entry states**, from a generator frozen before training and never sampled during it.

**How these differ from the training archive.** The archive in `celeste_rl/starts.py` holds states the agent
itself reached while training, and training samples from it. These come from fresh runs of
`scripts/find_room_exit.py`, a search that never runs during training, seeded so that they are not the routes
the demonstrations came from, and they are sampled along the whole route rather than wherever a policy happened
to get to. Nothing in the training path reads the file this writes. That separation is the entire point: a test
set a policy can be trained toward is not a test set.

**Reachable by construction.** A state is stored as the input lines that reach it from the canonical start, the
same recipe form the archive uses, so replaying it is the proof that it is legal and reachable. Every candidate
is replayed here and dropped unless it arrives where it says, alive and in room 1.

The result is written to `config/heldout_starts.json` and committed, so the set is fixed and auditable. It
carries a sha256 over the states, which the evaluator records, so a number can always name the test set it was
measured against. **Regenerating it invalidates comparisons with anything measured against the old one.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import parse_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402

DEFAULT_OUTPUT = REPO / "config" / "heldout_starts.json"
# Seeds 0 to 6 produced the demonstrations. Held-out starts begin after them so the two sets never overlap.
FIRST_SEED = 100


def search(seed: int, game_dir: Path, minutes: float) -> Path | None:
    """One fresh route from the Go-Explore style search, as a new runs/routes directory."""
    before = {p.name for p in (REPO / "runs" / "routes").glob("*")}
    result = subprocess.run(
        [str(REPO / ".venv-rl" / "Scripts" / "python.exe"), str(REPO / "scripts" / "find_room_exit.py"),
         "--seed", str(seed), "--game-dir", str(game_dir), "--max-minutes", str(minutes)],
        cwd=REPO, capture_output=True, text=True)
    after = {p.name for p in (REPO / "runs" / "routes").glob("*")} - before
    if result.returncode != 0 or not after:
        print(f"  seed {seed}: no route ({result.returncode})")
        return None
    folder = REPO / "runs" / "routes" / sorted(after)[-1]
    return folder if (folder / "route.json").exists() else None


def candidates(routes: list[Path], wanted: int, earliest: int, spacing: int) -> list[dict]:
    """Prefixes sampled along each route, spread over the whole room rather than bunched at one depth."""
    picked = []
    for path in routes:
        route = json.loads((path / "route.json").read_text(encoding="utf-8"))
        step = route.get("transition_step")
        if not step:
            continue
        lines = [format_input_line(buttons) for buttons in route["actions"][:step]]
        # Stop short of the transition: a start one frame from the exit tests nothing.
        for frame in range(earliest, max(earliest, step - spacing), spacing):
            picked.append({"route": path.name, "seed": route.get("seed"), "frames": frame,
                           "lines": lines[:frame]})
    picked.sort(key=lambda c: (c["frames"], c["route"]))
    if len(picked) <= wanted:
        return picked
    # Keep an even spread across depths rather than the first N, which would all be near the start.
    stride = len(picked) / wanted
    return [picked[int(i * stride)] for i in range(wanted)]


def validate(env: CelesteRoomEnv, candidate: dict) -> dict | None:
    """Replay a candidate and keep it only if it arrives alive in room 1, where it says it does."""
    provisional = Start(tuple(candidate["lines"]), (0.0, 0.0), "1", None)
    env.reset(options={"canonical": True})
    info = None
    for line in candidate["lines"]:
        _, _, terminated, _, info = env.step(parse_line(line))
        if terminated:
            return None
    player = info["player"] if info else None
    if player is None:
        return None
    return {"route": candidate["route"], "search_seed": candidate["seed"], "frames": provisional.frames,
            "position": [player["x"], player["y"]], "dashes": player["dashes"], "lines": list(candidate["lines"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--searches", type=int, default=6, help="fresh searches to run, seeded from 100")
    parser.add_argument("--search-minutes", type=float, default=10)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--earliest", type=int, default=20, help="do not start an episode in its first frames")
    parser.add_argument("--spacing", type=int, default=12, help="frames between sampled states along a route")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--routes", nargs="*", type=Path, help="use these route directories instead of searching")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    if args.output.exists():
        print(f"{args.output} already exists. Regenerating it would invalidate every comparison measured "
              "against the current set; delete it deliberately if that is what you mean to do.")
        return 2

    output_dir = REPO / "runs" / "heldout-starts" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)

    # The searches run before this script takes the game. Each child launches its own copy, and the launch
    # guard refuses while another game-copy process is alive, so holding the ports here makes every search
    # fail and the whole set come back empty.
    routes = list(args.routes or [])
    if not routes:
        print(f"Searching for {args.searches} fresh routes, seeds {FIRST_SEED} upwards")
        for offset in range(args.searches):
            found = search(FIRST_SEED + offset, args.game_dir, args.search_minutes)
            if found:
                print(f"  seed {FIRST_SEED + offset}: {found.name}")
                routes.append(found)
    # Any held-out route already on disk counts, so an interrupted build can be continued.
    existing = [p for p in sorted((REPO / "runs" / "routes").glob("2026*")) if (p / "route.json").exists()
                and (json.loads((p / "route.json").read_text(encoding="utf-8")).get("seed") or 0) >= FIRST_SEED]
    routes = sorted(set(routes) | set(existing))
    if not routes:
        print("No routes found, so no held-out starts.")
        return 1
    print(f"{len(routes)} held-out routes (seeds {FIRST_SEED} and up)")

    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http))
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2

        chosen = candidates(routes, args.states, args.earliest, args.spacing)
        print(f"\n{len(chosen)} candidate states from {len(routes)} routes; validating each by replay")
        states = []
        for index, candidate in enumerate(chosen, start=1):
            kept = validate(env, candidate)
            if kept:
                states.append(kept)
            if index % 25 == 0:
                print(f"  {index}/{len(chosen)}: {len(states)} valid")
    finally:
        env.close()
        game.close()

    if not states:
        print("No candidate replayed successfully.")
        return 1
    digest = hashlib.sha256(json.dumps([s["lines"] for s in states], sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "commit": git["commit"],
        "generator": "scripts/find_room_exit.py, fresh seeds from 100, sampled along the route",
        "note": "Evaluation only. Nothing in the training path may read this file.",
        "sha256": digest, "states": len(states),
        "routes": sorted({s["route"] for s in states}),
        "frames": {"min": min(s["frames"] for s in states), "max": max(s["frames"] for s in states)},
        "entries": states,
    }, indent=1), encoding="utf-8")
    xs = sorted(s["position"][0] for s in states)
    print(f"\n{len(states)} held-out starts written to {args.output}")
    print(f"  sha256 {digest[:16]}, prefixes {min(s['frames'] for s in states)} to "
          f"{max(s['frames'] for s in states)} frames")
    print(f"  x from {xs[0]} to {xs[-1]}, median {xs[len(xs) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
