"""Re-run a finished run's first rollout from its own manifest and check it comes out the same.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/reproduce_check.py --run-dir runs/train/<run>

**Why this exists.** Two bugs in this project were found by luck rather than by checking. An empty
`StartArchive` is falsy, so `archive.sample if archive else None` silently handed the environment no sampler
and every episode of a varied-starts run began canonically while the config said otherwise; that surfaced
because someone looked at the ratio of start kinds two minutes in. Stable-Baselines3's `load()` re-seeds torch,
numpy and python globally from the donor checkpoint's saved seed, so two differently-seeded fine-tuning runs
produced byte-identical episodes; that surfaced because someone compared md5 hashes. **A bug that made a run
different but wrong would have survived both.**

The game is deterministic given the same inputs from the same savestate, and a run records its own config, so a
reported result can simply be re-run and compared. This replays the first rollout of a finished run under its
recorded configuration and diffs the resulting episodes against what that run recorded. It costs one rollout,
about ten seconds of stepping.

What a mismatch means, in rough order of likelihood: the code has changed since the run (check the commit in
the manifest), the game build has changed (check the runtime pins), the run was not actually seeded the way it
says, or something in the training path depends on state it should not. All four are worth knowing before a
number from that run is published.

Results go to runs/reproduce-check/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import TrainConfig, build_environment, train  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402

# The fields of an episode record that a reproduction must match exactly. Timings and memory are excluded
# because they are wall-clock dependent and not part of what the game computed.
COMPARED = ("ending", "length", "return", "start", "start_frames", "max_x", "min_y", "end_x", "end_y")


def episodes_of(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def compare(original: list[dict], repeated: list[dict]) -> dict:
    """Field-by-field over the episodes the shorter run produced."""
    shared = min(len(original), len(repeated))
    differences = []
    for index in range(shared):
        for field in COMPARED:
            before, after = original[index].get(field), repeated[index].get(field)
            if before != after:
                differences.append({"episode": index + 1, "field": field, "original": before, "repeated": after})
    return {"episodes_compared": shared, "original_episodes": len(original), "repeated_episodes": len(repeated),
            "differences": differences[:50], "difference_count": len(differences),
            "identical": not differences and shared > 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="a finished run to reproduce")
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--rollouts", type=int, default=1, help="how many rollouts to replay")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    stored = manifest["config"]
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(stored) - known)
    if unknown:
        print(f"{args.run_dir} was written by an older TrainConfig; it no longer has {unknown}. "
              "A reproduction would not be comparing like with like.")
        return 2
    config = TrainConfig(**{**stored, "disabled_inputs": tuple(stored["disabled_inputs"]),
                            "total_timesteps": stored["n_steps"] * args.rollouts,
                            # Checkpoints and evaluations only cost time here; the episodes are the evidence.
                            "checkpoint_every": stored["n_steps"] * args.rollouts, "eval_every": 0})
    original_commit = manifest["sessions"][0]["provenance"].get("commit")
    if original_commit and original_commit != git["commit"]:
        print(f"Note: that run was made at {original_commit[:8]} and this is {git['commit'][:8]}. "
              "A difference may be a real code change rather than a fault.")

    output_dir = REPO / "runs" / "reproduce-check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    print(f"Replaying {args.rollouts} rollout(s) of {args.run_dir} ({config.n_steps * args.rollouts:,} steps), "
          f"seed {config.seed}, reward {config.reward_version}"
          + (f", from {config.init_from}" if config.init_from else ""))

    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = build_environment(LockstepBridge(http), config, output_dir)
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        train(config, output_dir / "run", env, {**git, "reproduces": str(args.run_dir)})
    finally:
        env.close()
        game.close()

    result = compare(episodes_of(args.run_dir / "episodes.jsonl"),
                     episodes_of(output_dir / "run" / "episodes.jsonl"))
    result.update({**git, "run_dir": str(args.run_dir), "original_commit": original_commit,
                   "runtime_problems": problems, "rollouts": args.rollouts})
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if result["identical"]:
        print(f"\nIDENTICAL: {result['episodes_compared']} episodes match on {', '.join(COMPARED)}")
    else:
        print(f"\nDIFFERENT: {result['difference_count']} field differences over "
              f"{result['episodes_compared']} compared episodes")
        for difference in result["differences"][:6]:
            print(f"  episode {difference['episode']} {difference['field']}: "
                  f"{difference['original']!r} became {difference['repeated']!r}")
    print(f"Results: {output_dir / 'results.json'}")
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
