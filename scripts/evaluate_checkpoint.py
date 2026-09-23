"""Measure a saved policy by playing it, with an interval rather than a bare rate.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/evaluate_checkpoint.py --checkpoint runs/train/<run>/checkpoints/latest.zip
    .venv-rl/Scripts/python.exe scripts/evaluate_checkpoint.py --task-definition config/room2.json \
        --checkpoint runs/train/<run>/checkpoints/latest.zip --episodes 50 --deterministic-repeats 2 \
        --demonstration-starts config/demonstrations-room2.json --demonstrations-sha256 <manifest sha256> \
        --dataset runs/clone/<task>/<run>/dataset.npz --dataset-sha256 <file sha256> \
        --room-bounds 240 -184 320 180 --start-every 40 --start-episodes 5

A training run's own evaluations use 50 episodes, which is enough to steer a run and not enough to report. This
plays a checkpoint for as many episodes as asked and reports a **Wilson score interval**, which is the honest
way to state a success rate from a finite sample: 42 of 50 is not "84%", it is 84% with a 95% interval of about
71% to 92%, and the difference matters when comparing two arms.

It also records the checkpoint's sha256, so a number can always be traced to the exact weights that produced
it, and runs deterministic episodes, which do not depend on sampling luck.

**Demonstration starts** (optional) are training-side starts built in memory, never written as a manifest:
prefixes every `--start-every` task frames along each route of the frozen demonstration manifest, stopping
`--start-every` frames before the route's transition. Expected positions and dash counts come from the cloning
dataset recorded from those same routes. Dataset row k of a trajectory is the observation before action k + 1,
so it is the state after k task-relative actions, and row 0 is the task start. Each dataset trajectory is matched
to its route by comparing the recorded applied actions with the route's lines, never by manifest order. Positions
are reconstructed as `round(bounds + normalised * size)` and accepted only if they reproduce the stored float32
feature exactly and row 0 of every trajectory is the task start. The environment's start check then validates
every replay; a start that does not replay stops the run instead of being recorded under the wrong label.
These starts overlap the demonstrations by design and must never be used as held-out data; this script never
reads a held-out file and refuses any input path that names one.

Play order, fixed: the canonical start first, then demonstration starts in route order (dataset trajectory
order) and prefix order. At each start, all stochastic episodes, then all deterministic repeats. Without the new
options this is the original behaviour: `--episodes` stochastic canonical episodes, then one deterministic.

Results go to runs/evaluation/<timestamp>/ (named tasks: runs/evaluation/<task>/<timestamp>/): results.json,
episodes.jsonl with one row per episode, and starts.json when demonstration starts are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import apply_disabled, disabled_mask, parse_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.demonstrations import (  # noqa: E402
    DemonstrationManifestError,
    file_sha256,
    materialize_demonstrations,
    validate_demonstration_manifest,
)
from celeste_rl.env import CANONICAL_START, CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.reward import RewardConfig  # noqa: E402
from celeste_rl.schema import MENU_INPUTS, PLAYER_FEATURE_NAMES  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.tasks import (  # noqa: E402
    TaskDefinition,
    TaskDefinitionError,
    resolve_task_definition,
    start_recipe,
    task_identity,
)
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import new_progress, run_episode, update_progress  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

CANONICAL_ID = "canonical"
PLAY_ORDER = ("canonical start first, then demonstration starts in dataset trajectory order and prefix order; "
              "at each start all stochastic episodes, then all deterministic repeats")
FEATURE = {name: index for index, name in enumerate(PLAYER_FEATURE_NAMES)}


class StartBuildError(ValueError):
    """Demonstration starts cannot be built with the evidence required."""


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% Wilson score interval. Unlike the textbook normal interval it stays inside 0 to 1 and behaves at
    the extremes, where a run of all successes or none is exactly where a rate is most tempting to overstate."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def refuse_heldout_paths(paths) -> None:
    """Demonstration starts overlap training data by design; nothing held-out may be an input to them."""
    for path in paths:
        if path is not None and any("heldout" in part.lower() for part in Path(path).parts):
            raise StartBuildError(f"refusing a held-out input for demonstration starts: {path}")


def require_sha256(path: Path, expected: str, what: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise StartBuildError(f"{what} {path} has SHA-256 {actual}, expected {expected}")
    return actual


def _exact_integers(normalised: np.ndarray, origin: float, size: float, what: str) -> np.ndarray:
    """Integers whose (value - origin) / size reproduces every stored float32 feature exactly."""
    values = np.rint(origin + normalised.astype(np.float64) * size)
    recomputed = np.array([np.float32((value - origin) / size) for value in values], dtype=np.float32)
    if not np.array_equal(recomputed, normalised.astype(np.float32)):
        raise StartBuildError(f"{what} does not reconstruct exactly with origin {origin} and size {size}")
    return values.astype(int)


def match_trajectories(demonstrations: list[dict], trajectory: np.ndarray, actions: np.ndarray,
                       disabled=MENU_INPUTS) -> dict[int, dict]:
    """Map each dataset trajectory to the one demonstration whose applied actions it recorded."""
    mask = disabled_mask(disabled)
    expected = {demo["route_sha256"]: np.stack([apply_disabled(parse_line(line), mask) for line in demo["lines"]])
                for demo in demonstrations}
    mapping, used = {}, set()
    for index in sorted(int(value) for value in np.unique(trajectory)):
        recorded = actions[trajectory == index]
        matches = [demo for demo in demonstrations
                   if expected[demo["route_sha256"]].shape == recorded.shape
                   and np.array_equal(expected[demo["route_sha256"]], recorded)]
        if len(matches) != 1:
            raise StartBuildError(f"dataset trajectory {index} matches {len(matches)} demonstrations, not exactly one")
        if matches[0]["route_sha256"] in used:
            raise StartBuildError(f"demonstration {matches[0]['name']} matches more than one dataset trajectory")
        used.add(matches[0]["route_sha256"])
        mapping[index] = matches[0]
    missing = [demo["name"] for demo in demonstrations if demo["route_sha256"] not in used]
    if missing:
        raise StartBuildError(f"demonstrations with no dataset trajectory: {missing}")
    return mapping


def demonstration_starts(demonstrations: list[dict], trajectory: np.ndarray, actions: np.ndarray,
                         player: np.ndarray, definition: TaskDefinition, bounds: tuple[float, float, float, float],
                         every: int) -> list[dict]:
    """Starts every `every` task frames along each demonstration, with positions checked against the dataset.

    `player` is the current-frame player feature array (rows x features), aligned with `trajectory`.
    """
    if every <= 0:
        raise StartBuildError("--start-every must be positive")
    bx, by, bw, bh = bounds
    xs = _exact_integers(player[:, FEATURE["position_x"]], bx, bw, "position x")
    ys = _exact_integers(player[:, FEATURE["position_y"]], by, bh, "position y")
    dashes = _exact_integers(player[:, FEATURE["dashes"]], 0, 2, "dashes")
    if definition.start is not None:
        anchor, anchor_dashes, room = tuple(definition.start.position), definition.start.dashes, definition.start.room
    else:
        anchor, anchor_dashes, room = CANONICAL_START["position"], CANONICAL_START["dashes"], CANONICAL_START["room"]
    mapping = match_trajectories(demonstrations, trajectory, actions)
    starts = []
    for index, demo in mapping.items():
        rows = np.flatnonzero(trajectory == index)
        first = rows[0]
        if (xs[first], ys[first]) != tuple(anchor) or (anchor_dashes is not None and dashes[first] != anchor_dashes):
            raise StartBuildError(f"trajectory {index} does not begin at the task start {anchor}: "
                                  f"({xs[first]}, {ys[first]}) with {dashes[first]} dashes")
        transition = len(demo["lines"])
        for prefix in range(every, transition - every + 1, every):
            row = rows[prefix]  # the state after `prefix` task-relative actions
            lines = start_recipe(definition, demo["lines"][:prefix])
            starts.append({
                "start_id": f"{demo['name']}@{prefix}",
                "route": demo["name"],
                "route_sha256": demo["route_sha256"],
                "dataset_trajectory": index,
                "prefix_frames": prefix,
                "transition_step": transition,
                "room": room,
                "position": [int(xs[row]), int(ys[row])],
                "dashes": int(dashes[row]),
                "recipe_frames": len(lines),
                "recipe_sha256": hashlib.sha256(json.dumps(list(lines)).encode("utf-8")).hexdigest(),
                "lines": list(lines),
            })
    return starts


def start_list_record(starts: list[dict]) -> dict:
    """The start list as written before any episode, with a hash over everything except the full recipes
    (each recipe is represented by its own hash)."""
    listed = [{key: value for key, value in start.items() if key != "lines"} for start in starts]
    digest = hashlib.sha256(json.dumps(listed, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"starts": listed, "sha256": digest}


def as_start(entry: dict) -> Start:
    return Start(tuple(entry["lines"]), tuple(entry["position"]), entry["room"], entry["dashes"])


def play_from_start(model, env: CelesteRoomEnv, start: Start, deterministic: bool) -> dict:
    """One episode from a demonstration start. A start that falls back to the task start is a failure, never an
    episode: the row carries the problem and no ending."""
    obs, info = env.reset(options={"start": start})
    if info["start"] != "archive" or info["start_problem"] is not None:
        problem = info["start_problem"] or f"the environment used a {info['start']} start instead"
        return {"ending": None, "length": 0, "problem": problem, **new_progress()}
    progress = update_progress(new_progress(), info)
    length = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        length += 1
        update_progress(progress, info)
        if terminated or truncated:
            return {"ending": info["ending"], "length": length, "problem": None, **progress}


def play_all(model, env, plan: list[dict], write_row, canonical_episode=run_episode,
             start_episode=play_from_start) -> tuple[list[dict], str | None]:
    """Play every planned episode in order. Stops at the first start problem and returns it."""
    rows = []
    for item in plan:
        if item["start_id"] == CANONICAL_ID:
            episode = canonical_episode(model, env, deterministic=item["mode"] == "deterministic")
        else:
            episode = start_episode(model, env, as_start(item["start"]), item["mode"] == "deterministic")
        problem = episode.get("problem")
        row = {"order": len(rows), "start_id": item["start_id"], "mode": item["mode"], "repeat": item["repeat"],
               "ending": episode["ending"], "length": episode["length"], "max_x": episode.get("max_x"),
               "end_x": episode.get("end_x"), "end_y": episode.get("end_y"), "problem": problem,
               "episode": episode}
        if item["start_id"] != CANONICAL_ID:
            start = item["start"]
            row.update(route=start["route"], route_sha256=start["route_sha256"],
                       prefix_frames=start["prefix_frames"], start_position=start["position"],
                       start_dashes=start["dashes"])
        rows.append(row)
        write_row(row)
        if problem is not None:
            return rows, f"start {item['start_id']} did not replay: {problem}"
    return rows, None


def episode_plan(canonical_episodes: int, deterministic_repeats: int, starts: list[dict],
                 start_episodes: int) -> list[dict]:
    plan = [{"start_id": CANONICAL_ID, "mode": "stochastic", "repeat": i} for i in range(canonical_episodes)]
    plan += [{"start_id": CANONICAL_ID, "mode": "deterministic", "repeat": i} for i in range(deterministic_repeats)]
    for start in starts:
        plan += [{"start_id": start["start_id"], "start": start, "mode": "stochastic", "repeat": i}
                 for i in range(start_episodes)]
        plan += [{"start_id": start["start_id"], "start": start, "mode": "deterministic", "repeat": i}
                 for i in range(deterministic_repeats)]
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-definition", type=Path,
                        help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--episodes", type=int, default=200, help="stochastic episodes at the canonical start")
    parser.add_argument("--deterministic-repeats", type=int, default=1,
                        help="deterministic episodes per start; the game is deterministic, so repeats should agree")
    parser.add_argument("--demonstration-starts", type=Path,
                        help="frozen demonstration manifest whose routes supply in-memory evaluation starts")
    parser.add_argument("--demonstrations-sha256", help="required with --demonstration-starts: the manifest's sha256")
    parser.add_argument("--dataset", type=Path, help="cloning dataset recorded from those demonstrations")
    parser.add_argument("--dataset-sha256", help="required with --dataset: the file's SHA-256")
    parser.add_argument("--room-bounds", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="the task room's Level.Bounds, to turn normalised positions back into world pixels")
    parser.add_argument("--start-every", type=int, default=40, help="task frames between demonstration starts")
    parser.add_argument("--start-episodes", type=int, default=5, help="stochastic episodes per demonstration start")
    parser.add_argument("--seed", type=int, default=0,
                        help="the evaluation's own sampling seed; without it every run of this script replays "
                             "the same episodes, because loading a checkpoint re-seeds the global generators "
                             "from the seed that checkpoint was saved with")
    parser.add_argument("--reward-version", default="rew-v2")
    parser.add_argument("--shaping-scale", type=float, default=2.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    try:
        definition = resolve_task_definition(args.task_definition)
        identity = task_identity(definition)
    except TaskDefinitionError as error:
        parser.error(str(error))
    if args.episodes < 0 or args.deterministic_repeats < 1 or args.start_episodes < 0:
        parser.error("--episodes and --start-episodes must be non-negative and --deterministic-repeats positive")
    if args.demonstration_starts is not None:
        missing = [name for name, value in (("--demonstrations-sha256", args.demonstrations_sha256),
                                            ("--dataset", args.dataset), ("--dataset-sha256", args.dataset_sha256),
                                            ("--room-bounds", args.room_bounds)) if value is None]
        if missing:
            parser.error(f"--demonstration-starts also needs {', '.join(missing)}")

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2

    starts, starts_provenance = [], None
    if args.demonstration_starts is not None:
        try:
            refuse_heldout_paths([args.demonstration_starts, args.dataset])
            manifest = json.loads(args.demonstration_starts.read_text(encoding="utf-8"))
            entries = validate_demonstration_manifest(manifest, identity)
            if manifest["sha256"] != args.demonstrations_sha256:
                raise StartBuildError(f"demonstration manifest sha256 {manifest['sha256']} is not the pinned "
                                      f"{args.demonstrations_sha256}")
            demonstrations = materialize_demonstrations(entries, REPO, identity)
            require_sha256(args.dataset, args.dataset_sha256, "dataset")
            stored = np.load(args.dataset)
            starts = demonstration_starts(demonstrations, stored["trajectory"], stored["actions"],
                                          stored["obs_player"][:, 0, :], definition, tuple(args.room_bounds),
                                          args.start_every)
        except (OSError, KeyError, json.JSONDecodeError, DemonstrationManifestError, StartBuildError) as error:
            print(f"Cannot build demonstration starts: {error}")
            return 2
        starts_provenance = {
            "manifest": str(args.demonstration_starts), "manifest_sha256": manifest["sha256"],
            "dataset": str(args.dataset), "dataset_sha256": args.dataset_sha256,
            "room_bounds": list(args.room_bounds), "start_every": args.start_every,
            "stochastic_per_start": args.start_episodes,
            "routes": {demo["name"]: sum(1 for s in starts if s["route"] == demo["name"]) for demo in demonstrations},
        }

    output_root = REPO / "runs" / "evaluation"
    if args.task_definition:
        output_root /= definition.name
    output_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if starts:
        # Written before any episode, so the starts are fixed by a file that exists before the first result.
        record = start_list_record(starts)
        (output_dir / "starts.json").write_text(json.dumps({**starts_provenance, **record}, indent=2),
                                                encoding="utf-8")
        starts_provenance.update(starts_file="starts.json", starts_sha256=record["sha256"], starts=len(starts))
    plan = episode_plan(args.episodes, args.deterministic_repeats, starts, args.start_episodes)

    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    kwargs = {"task": definition.task, "task_start": definition.start} if args.task_definition else {}
    env = CelesteRoomEnv(LockstepBridge(http),
                         reward_config=RewardConfig(version=args.reward_version,
                                                    shaping_scale=args.shaping_scale), **kwargs)
    aborted = None
    try:
        manifest = runtime.collect(args.game_dir, http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        model = SupervisedPPO.load(args.checkpoint, env=env, device="cpu")
        # SB3's load re-seeds torch, numpy and python from the checkpoint's saved seed, so without this every
        # evaluation of a given checkpoint draws the identical episode stream however many times it is run.
        model.set_random_seed(args.seed)
        print(f"{args.checkpoint} at {model.num_timesteps:,} accepted steps, {args.episodes} canonical episodes, "
              f"{len(starts)} demonstration starts, {len(plan)} episodes in all, sampling seed {args.seed}")
        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_file:
            def write_row(row: dict) -> None:
                flat = {key: value for key, value in row.items() if key != "episode"}
                episode_file.write(json.dumps({**flat, "checkpoint": str(args.checkpoint),
                                               "checkpoint_sha256": checkpoint_sha256, "task": identity},
                                              default=str) + "\n")
                episode_file.flush()
                if (row["order"] + 1) % 25 == 0:
                    print(f"  {row['order'] + 1:>5}/{len(plan)}")
            rows, aborted = play_all(model, env, plan, write_row)
    finally:
        env.close()
        game.close()

    canonical = [row for row in rows if row["start_id"] == CANONICAL_ID]
    episodes = [row["episode"] for row in canonical if row["mode"] == "stochastic"]
    deterministic_runs = [row["episode"] for row in canonical if row["mode"] == "deterministic"]
    successes = [e for e in episodes if e["ending"] == "success"]
    low, high = wilson(len(successes), len(episodes))
    results = {
        **git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "accepted_steps": int(model.num_timesteps),
        "reward_version": args.reward_version, "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else None,
        "wilson_95": [round(low, 4), round(high, 4)],
        "endings": dict(Counter(e["ending"] for e in episodes)),
        "success_length": {"median": statistics.median(e["length"] for e in successes),
                           "min": min(e["length"] for e in successes),
                           "max": max(e["length"] for e in successes)} if successes else None,
        "evaluation_seed": args.seed,
        "median_max_x": (statistics.median(x for x in (e["max_x"] for e in episodes) if x is not None)
                         if any(e["max_x"] is not None for e in episodes) else None),
        "deterministic": deterministic_runs[0] if deterministic_runs else None,
        "task": identity,
        "episodes_file": "episodes.jsonl",
        "play_order": PLAY_ORDER,
        "deterministic_repeats": args.deterministic_repeats,
        "canonical_deterministic_endings": [e["ending"] for e in deterministic_runs],
        "demonstration_starts": starts_provenance,
        "aborted": aborted,
    }
    if starts:
        by_start = {}
        for row in rows:
            if row["start_id"] != CANONICAL_ID and row["problem"] is None:
                entry = by_start.setdefault(row["start_id"], {"stochastic": [0, 0], "deterministic": [0, 0]})
                entry[row["mode"]][0] += row["ending"] == "success"
                entry[row["mode"]][1] += 1
        results["demonstration_start_successes"] = by_start
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    if aborted:
        print(f"\nSTOPPED: {aborted}. Rows so far are in {output_dir / 'episodes.jsonl'}.")
        return 3
    rate = f"{results['success_rate']:.1%}" if results["success_rate"] is not None else "n/a"
    print(f"\nsuccess {len(successes)}/{len(episodes)} = {rate}, 95% interval {low:.1%} to {high:.1%}")
    print(f"endings {results['endings']}, median max x {results['median_max_x']}, "
          f"deterministic {results['deterministic']['ending'] if results['deterministic'] else None}")
    if args.deterministic_repeats > 1:
        print(f"canonical deterministic repeats: {results['canonical_deterministic_endings']}")
    if starts:
        print(f"demonstration starts: {len(starts)}, rows in {output_dir / 'episodes.jsonl'}")
    if successes:
        print(f"clear length: median {results['success_length']['median']} frames "
              f"({results['success_length']['min']} to {results['success_length']['max']})")
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
