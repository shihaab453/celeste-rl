"""How far into room 1 do saved policies actually get?

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/checkpoint_progress.py --run-dir runs/train/unshaped-seed0

The training records say how episodes end, not where the player reached. This replays checkpoints from a finished
run and records, per episode, the furthest right and highest point reached and where the episode ended, plus the
milestones passed. An untrained policy of the same architecture is included as a baseline, so "the agent learned
to reach further" can be told apart from "random inputs reach that far anyway".

Room 1 runs from the start ledge at x 19 rightwards over a spike pit, up a platform and out through the top near
x 261, so larger x and smaller y are progress. Milestones are stated in the results, not learned from.

Results are written to runs/checkpoint-progress/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.policy import CelestePolicy, policy_kwargs  # noqa: E402
from celeste_rl.training.run import TrainConfig  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

# Landmarks of room 1, from its tile map and the recorded exit route.
MILESTONES = {"left the start ledge (x>40)": ("x", 40), "crossed the first spike pit (x>80)": ("x", 80),
              "reached the middle (x>144)": ("x", 144), "reached the right side (x>208)": ("x", 208),
              "under the exit (x>240)": ("x", 240), "upper half (y<90)": ("y", 90), "near the top (y<24)": ("y", 24)}


class PositionBridge:
    """Passes calls to the lockstep bridge and keeps the last reply, so the player's position can be read."""

    def __init__(self, bridge):
        self.bridge, self.last = bridge, None

    @property
    def http(self):
        return self.bridge.http

    def reset(self):
        self.last = self.bridge.reset()
        return self.last

    def step(self, buttons="", dash_only="", move_only=""):
        self.last = self.bridge.step(buttons, dash_only, move_only)
        return self.last

    def close(self):
        self.bridge.close()


def play(model, env: CelesteRoomEnv, bridge: PositionBridge, deterministic: bool) -> dict:
    obs, _ = env.reset()
    start = bridge.last.state["Player"]["Position"]
    max_x, min_y, end = start["X"], start["Y"], start
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, _, info = env.step(action)
        position = (bridge.last.state or {}).get("Player", {}).get("Position")
        if position:
            max_x, min_y, end = max(max_x, position["X"]), min(min_y, position["Y"]), position
        if terminated:
            return {"ending": info["ending"], "length": info["elapsed"], "max_x": max_x, "min_y": min_y,
                    "end_x": end["X"], "end_y": end["Y"]}


def summarise(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "endings": dict(Counter(e["ending"] for e in episodes)),
        "median_length": statistics.median(e["length"] for e in episodes),
        "max_x": {"median": statistics.median(e["max_x"] for e in episodes), "best": max(e["max_x"] for e in episodes)},
        "min_y": {"median": statistics.median(e["min_y"] for e in episodes), "best": min(e["min_y"] for e in episodes)},
        "milestones": {name: sum(1 for e in episodes if (e["max_x"] > value if axis == "x" else e["min_y"] < value))
                       for name, (axis, value) in MILESTONES.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--episodes", type=int, default=20, help="sampled episodes per checkpoint")
    parser.add_argument("--checkpoints", default="", help="comma separated names; default: a spread plus best and latest")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2

    checkpoint_dir = args.run_dir / "checkpoints"
    if args.checkpoints:
        names = [n.strip() for n in args.checkpoints.split(",")]
    else:
        steps = sorted(checkpoint_dir.glob("step_*.zip"))
        names = [p.name for p in steps[:: max(1, len(steps) // 6)]] + ["latest.zip", "best.zip"]
    config = TrainConfig(**{**json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))["config"],
                           "disabled_inputs": ("S", "Q", "N")})

    output_dir = REPO / "runs" / "checkpoint-progress" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    game = GameSession(args.game_dir)
    bridge = PositionBridge(LockstepBridge(CelesteBridge(output_dir / "episode.tas")))
    env = CelesteRoomEnv(bridge, disabled_inputs=config.disabled_inputs)
    results = {**git, "run_dir": str(args.run_dir), "episodes_per_checkpoint": args.episodes, "milestones": MILESTONES,
               "checkpoints": {}}
    try:
        manifest = runtime.collect(args.game_dir, bridge.http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        results["runtime"], results["runtime_problems"] = manifest, problems
        results["attributable"] = runtime.attributable(git, problems)

        untrained = SupervisedPPO(CelestePolicy, env, policy_kwargs=policy_kwargs(config.action_bias),
                                  n_steps=64, batch_size=64,
                                  device="cpu", seed=config.seed)
        for name, model in [("untrained (same architecture, seed)", untrained)] + \
                           [(n, SupervisedPPO.load(checkpoint_dir / n, env=env, device="cpu")) for n in names]:
            started = time.perf_counter()
            sampled = [play(model, env, bridge, deterministic=False) for _ in range(args.episodes)]
            greedy = play(model, env, bridge, deterministic=True)
            entry = {"sampled": summarise(sampled), "deterministic": greedy, "seconds": round(time.perf_counter() - started, 1)}
            results["checkpoints"][name] = entry
            (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            summary = entry["sampled"]
            print(f"{name:34s} max_x median {summary['max_x']['median']:6.1f} best {summary['max_x']['best']:6.1f} | "
                  f"min_y median {summary['min_y']['median']:6.1f} best {summary['min_y']['best']:6.1f} | "
                  f"len {summary['median_length']:6.1f} | {summary['endings']} | "
                  f"milestones {[k.split(' (')[0] for k, v in summary['milestones'].items() if v]}")
    finally:
        env.close()
        game.close()
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
