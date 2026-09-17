"""Train PPO on room 1 of Chapter 1 (Phase 3), or resume a run.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/train_room1.py --seed 0
    .venv-rl/Scripts/python.exe scripts/train_room1.py --seed 0 --total-timesteps 20000 --eval-every 10000   # smoke run
    .venv-rl/Scripts/python.exe scripts/train_room1.py --resume runs/train/<run>

The game is launched (minimized) and closed by this script. If a bridge fault finds the game process gone, it is
relaunched before the recovery reset. Records are described in celeste_rl/training/run.py; the run directory
defaults to runs/train/<timestamp>-seed<seed>.

Like the live checks, this refuses to start with uncommitted changes or a runtime that differs from
config/pinned_runtime.json unless told otherwise, because a campaign result must name its code and runtime.
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
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.reward import RewardConfig  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import TrainConfig, train  # noqa: E402
from celeste_rl.training.supervisor import TrainingAborted  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="continue the run in this directory from checkpoints/latest.zip")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    for item in fields(TrainConfig):
        if item.name != "disabled_inputs":
            parser.add_argument(f"--{item.name.replace('_', '-')}", type=type(item.default), default=None)
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
        config = TrainConfig(**{**stored, "disabled_inputs": tuple(stored["disabled_inputs"])})
        overrides = {k: v for k, v in vars(args).items() if k in stored and v is not None}
        if overrides:
            parser.error(f"a resumed run keeps its config; remove {sorted(overrides)}")
    else:
        values = {item.name: getattr(args, item.name) for item in fields(TrainConfig)
                  if item.name != "disabled_inputs" and getattr(args, item.name) is not None}
        config = TrainConfig(**values)
        run_dir = args.run_dir or REPO / "runs" / "train" / f"{datetime.now():%Y%m%d-%H%M%S}-seed{config.seed}"

    git = runtime.git_state()
    if git["uncommitted_changes"] and not args.allow_dirty:
        print(runtime.DIRTY_MESSAGE + "\n  " + "\n  ".join(git["changed_paths"]))
        return 2

    game = GameSession(args.game_dir)
    http = CelesteBridge(Path(run_dir) / "episode.tas")
    reward = RewardConfig(version=config.reward_version, gamma=config.gamma, shaping_scale=config.shaping_scale)
    env = CelesteRoomEnv(LockstepBridge(http), disabled_inputs=config.disabled_inputs, reward_config=reward)
    try:
        manifest = runtime.collect(args.game_dir, http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            if not args.allow_runtime_mismatch:
                return 2
        provenance = {**git, "runtime": manifest, "runtime_problems": problems,
                      "attributable": not git["uncommitted_changes"] and not problems}
        print(f"Run {run_dir} ({'resuming' if args.resume else 'new'}), config {asdict(config)}")
        try:
            model = train(config, Path(run_dir), env, provenance, on_fault=game.on_fault, resume=bool(args.resume),
                          health=game.health)
        except TrainingAborted as aborted:
            print(f"ABORTED: {aborted}")
            return 1
        print(f"Finished: {model.num_timesteps:,} accepted steps, faults {model.fault_stats['discarded_rollouts']} "
              f"discarded rollouts, game relaunches {len(game.relaunches)}. Records in {run_dir}")
        return 0
    finally:
        env.close()
        game.close()


if __name__ == "__main__":
    sys.exit(main())
