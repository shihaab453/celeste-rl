"""Build the frozen set of held-out entry states the real Phase 3 criterion is measured on.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/make_heldout_starts.py --searches 6 --states 200

Every result this project has reported comes from one fixed start in a deterministic game, so a success rate
measures how robust a policy's own sampling is around a single route rather than whether it can play the room.
The roadmap's criterion for the phase is different and much harder: clear the room from **200 held-out
reachable entry states**, from a generator frozen before training and never sampled during it.

**How these differ from the training archive.** The archive in `celeste_rl/starts.py` holds states the agent
itself reached while training, and training samples from it. These come from fresh runs of
`scripts/find_room_exit.py`, a search that never runs during training. Its routes are written to a dedicated
held-out namespace, and their complete-route hashes are checked against the explicit demonstration manifest
before cloning. They are sampled along the whole route rather than wherever a policy happened to get to.
Nothing in the training path reads the file this writes. That separation is the entire point: a test set a
policy can be trained toward is not a test set.

**Reachable by construction.** A state is stored as the input lines that reach it from the canonical start, the
same recipe form the archive uses, so replaying it is the proof that it is legal and reachable. Every candidate
is replayed here and dropped unless it arrives where it says, alive and in room 1.

The result is written to `config/heldout_starts.json` and committed, so the set is fixed and auditable. It
carries a sha256 over the states, which the evaluator records, so a number can always name the test set it was
measured against. **Regenerating it invalidates comparisons with anything measured against the old one.**
"""
from __future__ import annotations

import argparse
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
from celeste_rl.demonstrations import route_sha256  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.heldout import FORMAT_VERSION, freeze_entries, manifest_sha256, state_id  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402

DEFAULT_OUTPUT = REPO / "config" / "heldout_starts.json"
HELDOUT_ROUTES = REPO / "runs" / "heldout-routes"
# This is only the deterministic search-seed starting point. Namespace and content hashes enforce separation.
FIRST_SEED = 100


def load_route(path: Path) -> dict:
    return json.loads((path / "route.json").read_text(encoding="utf-8"))


def route_recipe(path: Path) -> tuple[str, ...] | None:
    """The part of a successful route that can contribute held-out starts."""
    route = load_route(path)
    step = route.get("transition_step")
    if not step or len(route.get("actions", [])) < step:
        return None
    return tuple(format_input_line(buttons) for buttons in route["actions"][:step])


def existing_heldout_routes(routes_dir: Path) -> list[Path]:
    """Completed search routes in the dedicated held-out directory."""
    found = []
    for path in sorted(routes_dir.glob("*")):
        route_file = path / "route.json"
        if not route_file.exists():
            continue
        found.append(path)
    return found


def unused_search_seeds(routes: list[Path], count: int) -> list[int]:
    """Choose fresh seeds after an interrupted build without ever reusing one already on disk."""
    used = set()
    for path in routes:
        seed = load_route(path).get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            used.add(seed)
    chosen, seed = [], FIRST_SEED
    while len(chosen) < count:
        if seed not in used:
            chosen.append(seed)
        seed += 1
    return chosen


def unique_routes(routes: list[Path]) -> tuple[list[Path], list[Path]]:
    """Keep one path for each distinct successful action trace."""
    unique, duplicates, seen = [], [], set()
    for path in sorted(set(routes)):
        recipe = route_recipe(path)
        if recipe is None:
            continue
        if recipe in seen:
            duplicates.append(path)
            continue
        seen.add(recipe)
        unique.append(path)
    return unique, duplicates


def require_minimum_routes(routes: list[Path], minimum: int) -> None:
    if len(routes) < minimum:
        raise ValueError(f"only {len(routes)} unique routes were produced; {minimum} required")


def search_command(seed: int, game_dir: Path, minutes: float, routes_dir: Path) -> list[str]:
    return [str(REPO / ".venv-rl" / "Scripts" / "python.exe"),
            str(REPO / "scripts" / "find_room_exit.py"), "--seed", str(seed),
            "--game-dir", str(game_dir), "--max-minutes", str(minutes),
            "--output-root", str(routes_dir)]


def search(seed: int, game_dir: Path, minutes: float, routes_dir: Path = HELDOUT_ROUTES) -> Path | None:
    """One fresh route from the Go-Explore search, isolated under the held-out namespace."""
    routes_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in routes_dir.glob("*")}
    result = subprocess.run(
        search_command(seed, game_dir, minutes, routes_dir),
        cwd=REPO, capture_output=True, text=True)
    after = {p.name for p in routes_dir.glob("*")} - before
    if result.returncode != 0 or not after:
        print(f"  seed {seed}: no route ({result.returncode})")
        return None
    folder = routes_dir / sorted(after)[-1]
    return folder if (folder / "route.json").exists() else None


def candidates(routes: list[Path], earliest: int, spacing: int) -> list[dict]:
    """Every distinct sampled prefix, ordered across depths for later even selection."""
    picked, seen = [], set()
    for path in routes:
        route = load_route(path)
        step = route.get("transition_step")
        if not step:
            continue
        lines = list(route_recipe(path) or ())
        complete_route_sha256 = route_sha256(lines)
        # Stop short of the transition: a start one frame from the exit tests nothing.
        for frame in range(earliest, max(earliest, step - spacing), spacing):
            prefix = tuple(lines[:frame])
            if prefix in seen:
                continue
            seen.add(prefix)
            picked.append({"route": path.name, "route_sha256": complete_route_sha256,
                           "seed": route.get("seed"), "frames": frame,
                           "lines": list(prefix)})
    picked.sort(key=lambda c: (c["frames"], c["route"]))
    return picked


def select_states(states: list[dict], wanted: int) -> list[dict]:
    """Deduplicate validated states and select exactly `wanted`, evenly across route depths."""
    unique, seen = [], set()
    for state in sorted(states, key=lambda s: (s["frames"], s["route"])):
        key = state_id(state)
        if key not in seen:
            seen.add(key)
            unique.append(state)
    if len(unique) < wanted:
        raise ValueError(f"only {len(unique)} unique valid states were produced; {wanted} requested")
    if len(unique) == wanted:
        return unique
    stride = len(unique) / wanted
    return [unique[int(index * stride)] for index in range(wanted)]


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
    return {"route": candidate["route"], "route_sha256": candidate["route_sha256"],
            "search_seed": candidate["seed"], "frames": provisional.frames,
            "room": "1", "position": [player["x"], player["y"]], "dashes": player["dashes"],
            "lines": list(candidate["lines"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--searches", type=int, default=6,
                        help="fresh search attempts; seeds already present on disk are skipped")
    parser.add_argument("--search-minutes", type=float, default=10)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--min-routes", type=int, default=10,
                        help="minimum distinct complete-route hashes required before sampling states")
    parser.add_argument("--earliest", type=int, default=20, help="do not start an episode in its first frames")
    parser.add_argument("--spacing", type=int, default=12, help="frames between sampled states along a route")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--routes", nargs="*", type=Path, help="use these route directories instead of searching")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    if args.searches < 0:
        parser.error("--searches must be non-negative")
    if args.states <= 0:
        parser.error("--states must be positive")
    if args.min_routes <= 0:
        parser.error("--min-routes must be positive")
    if args.earliest < 0:
        parser.error("--earliest must be non-negative")
    if args.spacing <= 0:
        parser.error("--spacing must be positive")

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
    if args.routes:
        routes = list(args.routes)
    else:
        routes_dir = HELDOUT_ROUTES
        routes = existing_heldout_routes(routes_dir)
        seeds = unused_search_seeds(routes, args.searches)
        if seeds:
            print(f"Searching with {len(seeds)} unused held-out seeds: {', '.join(map(str, seeds))}")
        for seed in seeds:
            found = search(seed, args.game_dir, args.search_minutes, routes_dir)
            if found:
                print(f"  seed {seed}: {found.name}")
                routes.append(found)

    routes, duplicate_routes = unique_routes(routes)
    if duplicate_routes:
        print(f"Ignoring {len(duplicate_routes)} duplicate route trace(s): "
              + ", ".join(path.name for path in duplicate_routes))
    if not routes:
        print("No routes found, so no held-out starts.")
        return 1
    print(f"{len(routes)} unique held-out routes")
    try:
        require_minimum_routes(routes, args.min_routes)
    except ValueError as error:
        print(f"{error}. Run more independent searches. Nothing was written.")
        return 1

    chosen = candidates(routes, args.earliest, args.spacing)
    if len(chosen) < args.states:
        print(f"Only {len(chosen)} unique candidate states are available; {args.states} requested. "
              "Run more independent searches. Nothing was written.")
        return 1

    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http))
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2

        print(f"\n{len(chosen)} unique candidate states from {len(routes)} routes; validating each by replay")
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

    try:
        states = select_states(states, args.states)
    except ValueError as error:
        print(f"{error}. Run more independent searches. Nothing was written.")
        return 1
    entries = freeze_entries(states)
    digest = manifest_sha256(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "commit": git["commit"],
        "generator": "scripts/find_room_exit.py, isolated held-out route namespace, unique route hashes",
        "note": "Evaluation only. Nothing in the training path may read this file.",
        "sha256": digest, "states": len(entries),
        "routes": sorted({entry["route"] for entry in entries}),
        "frames": {"min": min(entry["frames"] for entry in entries),
                   "max": max(entry["frames"] for entry in entries)},
        "entries": entries,
    }, indent=1), encoding="utf-8")
    xs = sorted(entry["position"][0] for entry in entries)
    print(f"\n{len(entries)} held-out starts written to {args.output}")
    print(f"  sha256 {digest[:16]}, prefixes {min(entry['frames'] for entry in entries)} to "
          f"{max(entry['frames'] for entry in entries)} frames")
    print(f"  x from {xs[0]} to {xs[-1]}, median {xs[len(xs) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
