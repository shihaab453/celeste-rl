"""Build the frozen set of held-out entry states the real Phase 3 criterion is measured on.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/make_heldout_starts.py --searches 6 --states 200
    .venv-rl/Scripts/python.exe scripts/make_heldout_starts.py --task-definition config/room2.json \
        --demonstrations config/demonstrations-room2.json --routes-dir runs/room2-heldout-routes \
        --output config/heldout_starts-room2.json

A replacement set for a task that already has one names the old set with `--exclude-heldout` (repeatable), so
no new route may repeat one of its complete routes, and starts its searches at `--first-seed` in a fresh
namespace, so the seeded search does not simply rediscover the old routes.

Every result this project has reported comes from one fixed start in a deterministic game, so a success rate
measures how robust a policy's own sampling is around a single route rather than whether it can play the room.
The roadmap's criterion for the phase is different and much harder: clear the room from **200 held-out
reachable entry states**, from a generator frozen before training and never sampled during it.

**How these differ from the training archive.** The archive in `celeste_rl/starts.py` holds states the agent
itself reached while training, and training samples from it. These come from fresh runs of
`scripts/find_room_exit.py`, a search that never runs during training. Its routes are written to a dedicated
held-out namespace, and their complete-route hashes are checked against the explicit demonstration manifest
during generation and again before cloning. They are sampled along the whole route rather than wherever a
policy happened to get to.
Nothing in the training path reads the file this writes. That separation is the entire point: a test set a
policy can be trained toward is not a test set.

**Reachable by construction.** A state is stored as the input lines that reach it from the base savestate, the
same recipe form the archive uses, so replaying it is the proof that it is legal and reachable. Every candidate
is replayed here and dropped unless it arrives where it says, alive and in the task's start room.

Room 1 defaults to `config/heldout_starts.json`; named tasks get their own output or an explicit `--output`.
The result is committed so the set is fixed and auditable. It
carries a sha256 over the states, which the evaluator records, so a number can always name the test set it was
measured against. **Regenerating it invalidates comparisons with anything measured against the old one.**
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import parse_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.demonstrations import (  # noqa: E402
    DemonstrationManifestError,
    materialize_demonstrations,
    reject_heldout_overlap,
    route_sha256,
    validate_demonstration_manifest,
)
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.heldout import (  # noqa: E402
    FORMAT_VERSION,
    HeldoutManifestError,
    freeze_entries,
    manifest_sha256,
    state_id,
    validate_manifest,
)
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.tasks import (  # noqa: E402
    TaskDefinition,
    TaskDefinitionError,
    canonical_task_identity,
    resolve_task_definition,
    start_recipe,
    task_identity,
)
from celeste_rl.training.game import GameSession  # noqa: E402

DEFAULT_OUTPUT = REPO / "config" / "heldout_starts.json"
DEFAULT_DEMONSTRATIONS = REPO / "config" / "demonstrations.json"
HELDOUT_ROUTES = REPO / "runs" / "heldout-routes"
# This is only the deterministic search-seed starting point. Namespace and content hashes enforce separation.
FIRST_SEED = 100
# One record per started search, inside the route namespace. It has no route.json, so it is never read as a route.
ATTEMPTS = "_attempts"


def load_route(path: Path) -> dict:
    return json.loads((path / "route.json").read_text(encoding="utf-8"))


def task_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(REPO.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return str(value).replace("\\", "/")


def route_recipe(path: Path, definition: TaskDefinition | None = None) -> tuple[str, ...] | None:
    """The part of a successful route that can contribute held-out starts."""
    route = load_route(path)
    if definition is not None and definition.definition_path is not None:
        expected_identity = task_identity(definition)
        if "task" in route:
            try:
                declared_identity = canonical_task_identity(route["task"])
            except TaskDefinitionError as error:
                raise ValueError(f"route {path.name} has invalid task identity: {error}") from error
            if declared_identity != expected_identity:
                raise ValueError(f"route {path.name} task identity does not match {definition.name}")
        expected = task_path(definition.definition_path)
        declared = task_path(str(route.get("task_definition") or ""))
        if declared != expected:
            raise ValueError(f"route {path.name} task definition {declared!r} does not match {expected!r}")
        if (route.get("start_room"), route.get("next_room")) != (
                definition.task.start_room, definition.task.target_room):
            raise ValueError(f"route {path.name} does not match rooms "
                             f"{definition.task.start_room}->{definition.task.target_room}")
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


def attempted_seeds(routes_dir: Path) -> set[int]:
    """Seeds whose search process was started in this namespace, whether or not it produced a route.

    Read from the file names, not their contents, so a record cut short by a kill still counts."""
    seeds = set()
    for path in (routes_dir / ATTEMPTS).glob("seed-*.json"):
        match = re.fullmatch(r"seed-(\d+)\.json", path.name)
        if match:
            seeds.add(int(match.group(1)))
    return seeds


def _write_attempt(record: Path, data: dict) -> None:
    temporary = record.with_name(record.name + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, record)


def unused_search_seeds(routes: list[Path], count: int, first_seed: int = FIRST_SEED,
                        attempted: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """Choose fresh seeds from `first_seed` on, never reusing one already on disk, even after an interruption.

    `attempted` holds seeds whose search started but left no route (a failed launch, a crash, a stopped run):
    a seed counts as tried as soon as its search process starts."""
    used = set(attempted)
    for path in routes:
        seed = load_route(path).get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            used.add(seed)
    chosen, seed = [], first_seed
    while len(chosen) < count:
        if seed not in used:
            chosen.append(seed)
        seed += 1
    return chosen


def unique_routes(routes: list[Path], definition: TaskDefinition | None = None) -> tuple[list[Path], list[Path]]:
    """Keep one path for each distinct successful action trace."""
    unique, duplicates, seen = [], [], set()
    for path in sorted(set(routes)):
        recipe = route_recipe(path, definition)
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


def new_route_hashes(routes: list[Path], definition: TaskDefinition) -> list[dict]:
    """The complete task-relative route hash of every successful route, with the route it came from."""
    hashes = []
    for path in routes:
        recipe = route_recipe(path, definition if definition.definition_path else None)
        if recipe is not None:
            hashes.append({"route": path.name, "route_sha256": route_sha256(recipe)})
    return hashes


def reject_demonstration_routes(routes: list[Path], demonstrations: list[dict],
                                definition: TaskDefinition) -> None:
    """Reject evaluation routes whose complete task-relative transition trace appears in training data."""
    reject_heldout_overlap(demonstrations, new_route_hashes(routes, definition))


def excluded_heldout_routes(manifests: list[Path], identity: dict) -> tuple[list[dict], list[dict]]:
    """The route hashes of earlier held-out sets, each manifest validated against the task first.

    Returns ({route_sha256} per entry, provenance per manifest). Only the hashes are compared; no state is
    replayed."""
    excluded, provenance = [], []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise HeldoutManifestError(f"{path} is not a held-out manifest")
        entries = validate_manifest(manifest, identity)
        excluded += [{"route_sha256": entry["route_sha256"]} for entry in entries]
        provenance.append({"path": task_path(str(path)), "sha256": manifest["sha256"],
                           "routes": len({entry["route_sha256"] for entry in entries})})
    return excluded, provenance


def reject_excluded_heldout_routes(routes: list[Path], excluded: list[dict], definition: TaskDefinition) -> None:
    """Reject new routes whose complete task-relative route hash belongs to an earlier held-out set."""
    new = new_route_hashes(routes, definition)
    try:
        reject_heldout_overlap(excluded, new)
    except DemonstrationManifestError:
        excluded_hashes = {entry["route_sha256"] for entry in excluded}
        repeated = sorted(entry["route"] for entry in new if entry["route_sha256"] in excluded_hashes)
        raise ValueError("new route(s) repeat a route of an excluded held-out set: " + ", ".join(repeated)) from None


def search_command(seed: int, game_dir: Path, minutes: float, routes_dir: Path,
                   task_definition: Path | None = None) -> list[str]:
    command = [str(REPO / ".venv-rl" / "Scripts" / "python.exe"),
               str(REPO / "scripts" / "find_room_exit.py"), "--seed", str(seed),
               "--game-dir", str(game_dir), "--max-minutes", str(minutes)]
    if task_definition is not None:
        command += ["--task-definition", str(task_definition)]
    return command + ["--output-root", str(routes_dir)]


def search(seed: int, game_dir: Path, minutes: float, routes_dir: Path = HELDOUT_ROUTES,
           task_definition: Path | None = None) -> Path | None:
    """One fresh route from the Go-Explore search, isolated under the held-out namespace.

    The attempt is recorded under _attempts/ before the child starts, so the seed stays tried whatever happens
    next, and the child's output is kept beside the record, so a failure can be read without the game's logs."""
    routes_dir.mkdir(parents=True, exist_ok=True)
    attempts = routes_dir / ATTEMPTS
    attempts.mkdir(exist_ok=True)
    record = attempts / f"seed-{seed}.json"
    started = datetime.now().isoformat(timespec="seconds")
    _write_attempt(record, {"seed": seed, "started": started})
    # Listed after the record exists, so a folder this call created for its own bookkeeping is never taken
    # for the child's route folder.
    before = {p.name for p in routes_dir.glob("*")}
    # Streamed straight to the file, so the output survives even if this process is killed mid-search.
    with (attempts / f"seed-{seed}.out").open("w", encoding="utf-8") as output:
        result = subprocess.run(
            search_command(seed, game_dir, minutes, routes_dir, task_definition),
            cwd=REPO, stdout=output, stderr=subprocess.STDOUT, text=True)
    after = {p.name for p in routes_dir.glob("*") if not p.name.startswith("_")} - before
    folder = routes_dir / sorted(after)[-1] if result.returncode == 0 and after else None
    found = folder if folder is not None and (folder / "route.json").exists() else None
    _write_attempt(record, {"seed": seed, "started": started, "returncode": result.returncode,
                            "route": found.name if found else None})
    if found is None:
        print(f"  seed {seed}: no route ({result.returncode})")
    return found


def candidates(routes: list[Path], earliest: int, spacing: int,
               definition: TaskDefinition | None = None) -> list[dict]:
    """Every distinct sampled prefix, ordered across depths for later even selection."""
    definition = definition or resolve_task_definition(None)
    picked, seen = [], set()
    for path in routes:
        route = load_route(path)
        step = route.get("transition_step")
        if not step:
            continue
        relative_lines = list(route_recipe(path, definition if definition.definition_path else None) or ())
        complete_route_sha256 = route_sha256(relative_lines)
        # Stop short of the transition: a start one frame from the exit tests nothing.
        for frame in range(earliest, max(earliest, step - spacing), spacing):
            relative_prefix = tuple(relative_lines[:frame])
            prefix = start_recipe(definition, relative_prefix)
            if prefix in seen:
                continue
            seen.add(prefix)
            picked.append({"route": path.name, "route_sha256": complete_route_sha256,
                           "seed": route.get("seed"), "frames": len(prefix), "task_frames": frame,
                           "lines": list(prefix)})
    picked.sort(key=lambda c: (c["task_frames"], c["route"]))
    return picked


def select_states(states: list[dict], wanted: int) -> list[dict]:
    """Deduplicate validated states and select exactly `wanted`, evenly across route depths."""
    unique, seen = [], set()
    for state in sorted(states, key=lambda s: (s.get("task_frames", s["frames"]), s["route"])):
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


def validate(env: CelesteRoomEnv, candidate: dict, definition: TaskDefinition) -> dict | None:
    """Measure a task-relative prefix, then prove its complete base-savestate recipe replays exactly."""
    setup_frames = definition.start.frames if definition.start is not None else 0
    relative_lines = candidate["lines"][setup_frames:]
    env.reset(options={"canonical": True})
    info = None
    for line in relative_lines:
        _, _, terminated, _, info = env.step(parse_line(line))
        if terminated:
            return None
    player = info["player"] if info else None
    if player is None:
        return None
    start = Start(tuple(candidate["lines"]), (player["x"], player["y"]),
                  definition.task.start_room, player["dashes"])
    _, replay = env.reset(options={"start": start})
    replayed_player = replay["player"]
    if (replay["start"] != "archive" or replay["elapsed"] != candidate["task_frames"]
            or replayed_player is None
            or (replayed_player["x"], replayed_player["y"]) != start.position
            or replayed_player["dashes"] != start.dashes):
        return None
    return {"route": candidate["route"], "route_sha256": candidate["route_sha256"],
            "search_seed": candidate["seed"], "frames": start.frames,
            "task_frames": candidate["task_frames"], "room": definition.task.start_room,
            "position": [player["x"], player["y"]],
            "room_position": [player["room_x"], player["room_y"]], "dashes": player["dashes"],
            "lines": list(candidate["lines"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--task-definition", type=Path,
                        help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--demonstrations", type=Path,
                        help="frozen training manifest whose routes must not overlap this evaluation set")
    parser.add_argument("--exclude-heldout", type=Path, action="append", default=[],
                        help="an earlier held-out manifest for this task whose routes must not recur; repeatable")
    parser.add_argument("--searches", type=int, default=6,
                        help="fresh search attempts; seeds already present on disk are skipped")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED,
                        help="lowest search seed; seeds already present in the namespace are still skipped")
    parser.add_argument("--search-minutes", type=float, default=10)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--min-routes", type=int, default=10,
                        help="minimum distinct complete-route hashes required before sampling states")
    parser.add_argument("--earliest", type=int, default=20, help="do not start an episode in its first frames")
    parser.add_argument("--spacing", type=int, default=12, help="frames between sampled states along a route")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--routes-dir", type=Path,
                        help="dedicated search namespace; defaults to the Room 1 namespace or a task-named one")
    parser.add_argument("--routes", nargs="*", type=Path, help="use these route directories instead of searching")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    try:
        definition = resolve_task_definition(args.task_definition)
        identity = task_identity(definition)
    except TaskDefinitionError as error:
        parser.error(str(error))
    safe_name = "".join(character if character.isalnum() or character in "-_" else "-"
                        for character in definition.name).strip("-")
    output = args.output or (DEFAULT_OUTPUT if args.task_definition is None else
                             REPO / "config" / f"heldout_starts-{safe_name}.json")
    routes_dir = args.routes_dir or (HELDOUT_ROUTES if args.task_definition is None else
                                     REPO / "runs" / f"heldout-routes-{safe_name}")
    if args.demonstrations is None:
        if args.task_definition:
            parser.error("--demonstrations is required with --task-definition")
        args.demonstrations = DEFAULT_DEMONSTRATIONS
    if args.searches < 0:
        parser.error("--searches must be non-negative")
    if args.first_seed < 0:
        parser.error("--first-seed must be non-negative")
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
    if output.exists():
        print(f"{output} already exists. Regenerating it would invalidate every comparison measured "
              "against the current set; delete it deliberately if that is what you mean to do.")
        return 2
    try:
        demonstration_manifest = json.loads(args.demonstrations.read_text(encoding="utf-8"))
        demonstration_entries = validate_demonstration_manifest(demonstration_manifest, identity)
        materialize_demonstrations(demonstration_entries, REPO, identity)
    except (OSError, json.JSONDecodeError, DemonstrationManifestError) as error:
        print(f"Cannot verify demonstration separation: {error}")
        return 2
    try:
        excluded_routes, excluded_provenance = excluded_heldout_routes(args.exclude_heldout, identity)
    except (OSError, json.JSONDecodeError, HeldoutManifestError) as error:
        print(f"Cannot verify separation from an earlier held-out set: {error}")
        return 2

    output_root = REPO / "runs" / "heldout-starts"
    if args.task_definition:
        output_root /= safe_name
    output_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)

    # The searches run before this script takes the game. Each child launches its own copy, and the launch
    # guard refuses while another game-copy process is alive, so holding the ports here makes every search
    # fail and the whole set come back empty.
    if args.routes:
        routes = list(args.routes)
    else:
        routes = existing_heldout_routes(routes_dir)
        seeds = unused_search_seeds(routes, args.searches, args.first_seed, attempted_seeds(routes_dir))
        if seeds:
            print(f"Searching with {len(seeds)} unused held-out seeds: {', '.join(map(str, seeds))}")
        for seed in seeds:
            found = search(seed, args.game_dir, args.search_minutes, routes_dir, args.task_definition)
            if found:
                print(f"  seed {seed}: {found.name}")
                routes.append(found)

    try:
        routes, duplicate_routes = unique_routes(routes, definition if args.task_definition else None)
        reject_demonstration_routes(routes, demonstration_entries, definition)
        reject_excluded_heldout_routes(routes, excluded_routes, definition)
    except (ValueError, DemonstrationManifestError) as error:
        print(f"Cannot use held-out routes: {error}")
        return 2
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

    chosen = candidates(routes, args.earliest, args.spacing, definition)
    if len(chosen) < args.states:
        print(f"Only {len(chosen)} unique candidate states are available; {args.states} requested. "
              "Run more independent searches. Nothing was written.")
        return 1

    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http), task=definition.task, task_start=definition.start)
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2

        print(f"\n{len(chosen)} unique candidate states from {len(routes)} routes; validating each by replay")
        states = []
        for index, candidate in enumerate(chosen, start=1):
            kept = validate(env, candidate, definition)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "commit": git["commit"],
        "generator": "scripts/find_room_exit.py, isolated held-out route namespace, unique route hashes",
        "note": "Evaluation only. Nothing in the training path may read this file.",
        "task": identity,
        **({"excluded_heldout": excluded_provenance} if excluded_provenance else {}),
        "sha256": digest, "states": len(entries),
        "routes": sorted({entry["route"] for entry in entries}),
        "frames": {"min": min(entry["frames"] for entry in entries),
                    "max": max(entry["frames"] for entry in entries)},
        "task_frames": {"min": min(entry["task_frames"] for entry in entries),
                         "max": max(entry["task_frames"] for entry in entries)},
        "entries": entries,
    }, indent=1), encoding="utf-8")
    xs = sorted(entry["room_position"][0] for entry in entries)
    print(f"\n{len(entries)} held-out starts written to {output}")
    print(f"  sha256 {digest[:16]}, task-relative prefixes "
          f"{min(entry['task_frames'] for entry in entries)} to "
          f"{max(entry['task_frames'] for entry in entries)} frames")
    print(f"  room-local x from {xs[0]} to {xs[-1]}, median {xs[len(xs) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
