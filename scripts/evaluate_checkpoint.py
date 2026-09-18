"""Measure a saved policy by playing it, with an interval rather than a bare rate.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/evaluate_checkpoint.py --checkpoint runs/train/<run>/checkpoints/latest.zip

A training run's own evaluations use 50 episodes, which is enough to steer a run and not enough to report. This
plays a checkpoint for as many episodes as asked and reports a **Wilson score interval**, which is the honest
way to state a success rate from a finite sample: 42 of 50 is not "84%", it is 84% with a 95% interval of about
71% to 92%, and the difference matters when comparing two arms.

It also records the checkpoint's sha256, so a number can always be traced to the exact weights that produced
it, and runs one deterministic episode, which does not depend on sampling luck.

Results go to runs/evaluation/<timestamp>/results.json.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.reward import RewardConfig  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import run_episode  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0,
                        help="the evaluation's own sampling seed; without it every run of this script replays "
                             "the same episodes, because loading a checkpoint re-seeds the global generators "
                             "from the seed that checkpoint was saved with")
    parser.add_argument("--reward-version", default="rew-v2")
    parser.add_argument("--shaping-scale", type=float, default=2.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2

    output_dir = REPO / "runs" / "evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http),
                         reward_config=RewardConfig(version=args.reward_version,
                                                    shaping_scale=args.shaping_scale))
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
        print(f"{args.checkpoint} at {model.num_timesteps:,} accepted steps, {args.episodes} episodes, "
              f"sampling seed {args.seed}")
        episodes = []
        for index in range(args.episodes):
            episodes.append(run_episode(model, env, deterministic=False))
            if (index + 1) % 25 == 0:
                rate = sum(1 for e in episodes if e["ending"] == "success") / len(episodes)
                print(f"  {index + 1:>4}/{args.episodes}: {rate:.0%} so far")
        deterministic = run_episode(model, env, deterministic=True)
    finally:
        env.close()
        game.close()

    successes = [e for e in episodes if e["ending"] == "success"]
    low, high = wilson(len(successes), len(episodes))
    results = {
        **git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "accepted_steps": int(model.num_timesteps),
        "reward_version": args.reward_version, "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "wilson_95": [round(low, 4), round(high, 4)],
        "endings": dict(Counter(e["ending"] for e in episodes)),
        "success_length": {"median": statistics.median(e["length"] for e in successes),
                           "min": min(e["length"] for e in successes),
                           "max": max(e["length"] for e in successes)} if successes else None,
        "evaluation_seed": args.seed,
        "median_max_x": (statistics.median(x for x in (e["max_x"] for e in episodes) if x is not None)
                         if any(e["max_x"] is not None for e in episodes) else None),
        "deterministic": deterministic,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nsuccess {len(successes)}/{len(episodes)} = {results['success_rate']:.1%}, "
          f"95% interval {low:.1%} to {high:.1%}")
    print(f"endings {results['endings']}, median max x {results['median_max_x']}, "
          f"deterministic {deterministic['ending']}")
    if successes:
        print(f"clear length: median {results['success_length']['median']} frames "
              f"({results['success_length']['min']} to {results['success_length']['max']})")
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
