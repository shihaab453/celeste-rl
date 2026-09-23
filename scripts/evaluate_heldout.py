"""Measure a policy against the frozen held-out entry states: the project's real Phase 3 criterion.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/evaluate_heldout.py --checkpoint runs/train/<run>/checkpoints/latest.zip
    .venv-rl/Scripts/python.exe scripts/evaluate_heldout.py --task-definition config/room2.json \
        --starts config/heldout_starts-room2.json --checkpoint runs/train/<run>/checkpoints/latest.zip

`scripts/evaluate_checkpoint.py` measures one start many times, which says how reliable a policy is around a
single route. This measures many starts, which says whether it can play the room. They are different claims and
the second is the one the roadmap asks for.

Each held-out state is replayed from the base savestate through the task start and the policy then plays from
there, once per state
by default, so the reported rate is over states rather than over repeats of one state. The frozen set's sha256
is recorded next to the result, so a number always names the complete test set it was measured against.

States sampled along one route are correlated, and repeated policies evaluated on the same states are also
correlated. This script therefore reports descriptive totals and per-route summaries, not an ordinary binomial
confidence interval. Cluster-aware uncertainty belongs in the analysis across routes and policy seeds.

Room 1 results go to `runs/heldout-evaluation/<timestamp>/results.json`; named tasks add a task directory.
Every attempted episode is written to `episodes.jsonl`.

Field meanings in episodes.jsonl: `max_x`, despite its name, and `end_x` are both the player's world x on the
episode's final frame; `max_x_episode` is the true running maximum. In `scripts/evaluate_checkpoint.py` rows,
`max_x` is the running maximum and `end_x` is the last frame that had a player. The two `end_x` values agree
whenever the final frame has a player, which was true of every Room 2 held-out row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.heldout import HeldoutManifestError, validate_manifest  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.reward import RewardConfig  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.tasks import (  # noqa: E402
    TaskDefinition,
    TaskDefinitionError,
    resolve_task_definition,
    task_identity,
)
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import new_progress, update_progress  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

DEFAULT_SET = REPO / "config" / "heldout_starts.json"


def entry_start(entry: dict, definition: TaskDefinition) -> tuple[Start, int, tuple[float, float]]:
    """Build a replayable start and fail closed on task-relative provenance mismatches."""
    lines = tuple(entry["lines"])
    setup = definition.start.lines if definition.start is not None else ()
    if lines[:len(setup)] != tuple(setup):
        raise HeldoutManifestError("held-out entry does not begin with the task-start recipe")
    task_frames = entry.get("task_frames", len(lines) - len(setup))
    if task_frames != len(lines) - len(setup):
        raise HeldoutManifestError("held-out entry task_frames does not match its replay recipe")
    if entry["room"] != definition.task.start_room:
        raise HeldoutManifestError("held-out entry room does not match the task start room")
    room_position = tuple(entry.get("room_position", entry["position"]))
    if definition.start is not None and "room_position" not in entry:
        raise HeldoutManifestError("later-room held-out entries must record room_position")
    return Start(lines, tuple(entry["position"]), entry["room"], entry["dashes"]), task_frames, room_position


def play(model, env: CelesteRoomEnv, start: Start, deterministic: bool) -> dict:
    obs, info = env.reset(options={"start": start})
    if info["start"] != "archive":
        return {"ending": None, "problem": info["start_problem"]}
    # The running maximum uses the training records' definition, starting from the first frame after the
    # start replay.
    progress = update_progress(new_progress(), info)
    length = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        length += 1
        update_progress(progress, info)
        if terminated or truncated:
            final_x = info["player"] and info["player"]["x"]
            return {"ending": info["ending"], "length": length,
                    # Despite its name, max_x is the player's world x on the final frame, not a maximum. It is
                    # kept unchanged for compatibility with the frozen result files; end_x is the same value
                    # under an honest name, and max_x_episode is the true running maximum.
                    "max_x": final_x, "end_x": final_x, "max_x_episode": progress["max_x"],
                    "problem": None}


def outcome_summary(episodes: list[dict]) -> dict:
    usable = [episode for episode in episodes if episode["problem"] is None]
    successes = [episode for episode in usable if episode["ending"] == "success"]
    return {
        "attempts": len(episodes),
        "episodes": len(usable),
        "stale_starts": len(episodes) - len(usable),
        "successes": len(successes),
        "success_rate": len(successes) / len(usable) if usable else None,
        "endings": dict(Counter(episode["ending"] for episode in usable)),
        "success_length_median": statistics.median(episode["length"] for episode in successes)
        if successes else None,
    }


def route_summaries(episodes: list[dict]) -> dict[str, dict]:
    grouped = {}
    for episode in episodes:
        grouped.setdefault(episode["route"], []).append(episode)
    summaries = {}
    for route, rows in sorted(grouped.items()):
        summaries[route] = {
            "search_seeds": sorted({row["search_seed"] for row in rows}),
            "states": len({row["state_id"] for row in rows}),
            **outcome_summary(rows),
        }
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-definition", type=Path,
                        help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--starts", type=Path)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--repeats", type=int, default=1, help="episodes per held-out state")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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
    if args.starts is None:
        if args.task_definition:
            parser.error("--starts is required with --task-definition")
        args.starts = DEFAULT_SET
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    if not args.starts.exists():
        print(f"No held-out set at {args.starts}. Build one with scripts/make_heldout_starts.py.")
        return 2
    frozen = json.loads(args.starts.read_text(encoding="utf-8"))
    try:
        entries = validate_manifest(frozen, identity)
        prepared = [entry_start(entry, definition) for entry in entries]
    except HeldoutManifestError as error:
        print(f"Invalid held-out set {args.starts}: {error}")
        return 2

    output_root = REPO / "runs" / "heldout-evaluation"
    if args.task_definition:
        output_root /= definition.name
    output_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http),
                         task=definition.task, task_start=definition.start,
                         reward_config=RewardConfig(version=args.reward_version, shaping_scale=args.shaping_scale))
    episodes = []
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        model = SupervisedPPO.load(args.checkpoint, env=env, device="cpu")
        model.set_random_seed(args.seed)
        print(f"{args.checkpoint} against {len(entries)} held-out starts "
              f"({args.repeats} episode(s) each), set {frozen['sha256'][:16]}")
        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_file:
            for index, (entry, prepared_start) in enumerate(zip(entries, prepared), start=1):
                start, task_frames, room_position = prepared_start
                for repeat in range(args.repeats):
                    result = play(model, env, start, args.deterministic)
                    episode = {
                        **result,
                        "state_id": entry["state_id"],
                        "route": entry["route"],
                        "search_seed": entry["search_seed"],
                        "start_frames": task_frames,
                        "start_replay_frames": entry["frames"],
                        "start_room": entry["room"],
                        "start_position": entry["position"],
                        "start_dashes": entry["dashes"],
                        "start_room_position": list(room_position),
                        "start_x": room_position[0],
                        "repeat": repeat,
                        "checkpoint": str(args.checkpoint),
                        "checkpoint_sha256": checkpoint_sha256,
                        "task": identity,
                    }
                    episodes.append(episode)
                    episode_file.write(json.dumps(episode, default=str) + "\n")
                if index % 25 == 0:
                    summary = outcome_summary(episodes)
                    rate = summary["success_rate"] or 0.0
                    print(f"  {index}/{len(entries)}: {rate:.0%} so far")
    finally:
        env.close()
        game.close()

    summary = outcome_summary(episodes)
    usable = [episode for episode in episodes if episode["problem"] is None]
    successes = [e for e in usable if e["ending"] == "success"]
    # Where it fails matters more than how often: a policy that only clears from near the exit is not playing.
    by_depth = {}
    for episode in usable:
        bucket = f"x {int(episode['start_x'] // 40) * 40}-{int(episode['start_x'] // 40) * 40 + 39}"
        hit, total = by_depth.get(bucket, (0, 0))
        by_depth[bucket] = (hit + (episode["ending"] == "success"), total + 1)
    by_route = route_summaries(episodes)
    route_rates = [route["success_rate"] for route in by_route.values() if route["success_rate"] is not None]
    results = {
        **git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
        "task": identity,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "heldout_set": str(args.starts), "heldout_sha256": frozen["sha256"],
        "heldout_file_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest(),
        "heldout_format_version": frozen["format_version"],
        "heldout_states": len(entries), "repeats": args.repeats,
        "deterministic": args.deterministic, "evaluation_seed": args.seed,
        **summary,
        "episodes_file": "episodes.jsonl",
        "uncertainty": "not computed: states are clustered within routes",
        "route_macro_success_rate": statistics.mean(route_rates) if route_rates else None,
        "by_route": by_route,
        "by_start_x": {k: {"successes": v[0], "episodes": v[1], "rate": round(v[0] / v[1], 3)}
                        for k, v in sorted(by_depth.items())},
        "depth_coordinate": "room-local x",
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    rate = f"{results['success_rate']:.1%}" if results["success_rate"] is not None else "n/a"
    print(f"\nheld-out success {len(successes)}/{len(usable)} = {rate} descriptive")
    if summary["stale_starts"]:
        print(f"  {summary['stale_starts']} starts no longer replay and were excluded")
    print("  by route: " + ", ".join(f"{name} {value['success_rate']:.0%} ({value['episodes']})"
                                      for name, value in by_route.items() if value["success_rate"] is not None))
    print("  by starting x: " + ", ".join(f"{k} {v['rate']:.0%} ({v['episodes']})"
                                          for k, v in results["by_start_x"].items()))
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
