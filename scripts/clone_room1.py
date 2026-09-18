"""Phase 3B: clone demonstrated play of room 1, then measure the cloned policy by playing it.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/clone_room1.py
    .venv-rl/Scripts/python.exe scripts/clone_room1.py --dataset runs/clone/<run>/dataset.npz   # refit, no game

Three stages, and only the third answers the project's question.

**Stage 1 (needs the game): record what the demonstrations saw.** A route is a list of input lines; cloning
needs the observation the policy would have had at each frame, so every demonstration is replayed through
`CelesteRoomEnv` from the canonical start and its observations captured. A replay that does not end in
`success` is dropped with a note rather than trained on. The pairs are saved so later fits need no game.

**Stage 2 (CPU): clone.** Whole demonstrations are held out, never frames (see `celeste_rl/cloning.py` for
why). Held-out accuracy is reported against the always-zero baseline, because the inputs are sparse.

**Stage 3 (needs the game): play.** The cloned policy runs the Phase 3 evaluation protocol from the canonical
start, stochastic and deterministic. **This is the number that matters.** Held-out accuracy says the policy
predicts the right button most of the time; only playing says whether it clears the room, because a cloned
policy's first mistake puts it in a state no demonstration visited and errors compound from there.

Demonstration sources are kept separate in the records, because they are not equivalent:

- `route`: independent Go-Explore style searches (`scripts/find_room_exit.py`), about 2.2 inputs held per
  frame, which is what real play looks like.
- `archive`: clears extracted from the training archives (`scripts/extract_demonstrations.py`), about 8.4
  inputs per frame, because they are the twelve-button policy's solutions.

Results go to runs/clone/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import parse_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.cloning import OBS_KEYS, Demonstrations, accuracy, clone, split_by_trajectory  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.observation import observation_space  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.policy import CelestePolicy, policy_kwargs  # noqa: E402
from celeste_rl.training.run import run_episode  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402


class SpacesOnly(gym.Env):
    """Enough environment for PPO to build a policy when there is no game to attach to."""

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = gym.spaces.MultiBinary(len(ACTION_INPUTS))

    def reset(self, *, seed=None, options=None):  # pragma: no cover
        raise RuntimeError("never stepped")

    def step(self, action):  # pragma: no cover
        raise RuntimeError("never stepped")


def collect_sources(args) -> list[dict]:
    """Every demonstration available, as input lines with their provenance."""
    found = []
    for path in sorted(REPO.glob("runs/routes/2026*/route.json")):
        route = json.loads(path.read_text(encoding="utf-8"))
        step = route.get("transition_step")
        if not step or len(route["actions"]) < step:
            continue
        lines = [format_input_line(buttons) for buttons in route["actions"][:step]]
        found.append({"kind": "route", "name": path.parent.name, "lines": lines})
    fixture = json.loads((REPO / "tests" / "fixtures" / "room1_exit_dash_route.json").read_text(encoding="utf-8"))
    found.append({"kind": "route", "name": "fixture",
                  "lines": [format_input_line(b) for b in fixture["actions"][:fixture["transition_step"]]]})
    if not args.routes_only:
        for path in sorted(REPO.glob("runs/demonstrations/*/demonstrations.json")):
            for index, demo in enumerate(json.loads(path.read_text(encoding="utf-8"))["demonstrations"]):
                found.append({"kind": "archive", "name": f"{path.parent.name}#{index}", "lines": demo["lines"]})
    # Identical input sequences are one demonstration, however many places they came from.
    unique, seen = [], set()
    for demo in found:
        key = tuple(demo["lines"])
        if key not in seen:
            seen.add(key)
            unique.append(demo)
    return unique


def record(env: CelesteRoomEnv, demos: list[dict]) -> tuple[Demonstrations, list[dict]]:
    """Replay every demonstration and keep the observation it saw at each frame."""
    obs_rows, actions, trajectory, provenance = [], [], [], []
    for index, demo in enumerate(demos):
        obs, _ = env.reset(options={"canonical": True})
        rows, taken, ending = [], [], None
        for line in demo["lines"]:
            rows.append({key: np.asarray(obs[key]) for key in OBS_KEYS})
            obs, _, terminated, _, info = env.step(parse_line(line))
            taken.append(np.asarray(info["applied_action"], dtype=np.int8))
            if terminated:
                ending = info["ending"]
                break
        kept = ending == "success"
        provenance.append({**{k: v for k, v in demo.items() if k != "lines"}, "frames": len(taken),
                           "ending": ending, "used": kept})
        print(f"  {demo['kind']:>7} {demo['name']:<28} {len(taken):>4} frames, {ending or 'no ending'}"
              f"{'' if kept else '  DROPPED'}")
        if not kept:
            continue
        obs_rows.extend(rows)
        actions.extend(taken)
        trajectory.extend([index] * len(taken))
    if not obs_rows:
        raise SystemExit(f"None of the {len(demos)} demonstrations replayed to a success, so there is nothing "
                         "to clone. The routes no longer reproduce against this game build.")
    stacked = {key: np.stack([row[key] for row in obs_rows]) for key in OBS_KEYS}
    return Demonstrations(stacked, np.stack(actions), np.asarray(trajectory), provenance), provenance


def save(path: Path, data: Demonstrations) -> None:
    np.savez_compressed(path, actions=data.actions, trajectory=data.trajectory,
                        **{f"obs_{key}": data.obs[key] for key in OBS_KEYS})


def load(path: Path) -> Demonstrations:
    stored = np.load(path)
    return Demonstrations({key: stored[f"obs_{key}"] for key in OBS_KEYS}, stored["actions"],
                          stored["trajectory"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--dataset", type=Path, help="refit these recorded pairs instead of replaying")
    parser.add_argument("--routes-only", action="store_true", help="exclude the archive-extracted clears")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--no-play", action="store_true", help="skip stage 3")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    output_dir = REPO / "runs" / "clone" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    results = {**git, "args": {k: str(v) for k, v in vars(args).items()}}

    game = env = None
    try:
        if args.dataset:
            data, results["provenance"] = load(args.dataset), [{"dataset": str(args.dataset)}]
            print(f"Refitting {len(data)} frames from {args.dataset}")
        else:
            demos = collect_sources(args)
            print(f"{len(demos)} demonstrations to replay")
            game = GameSession(args.game_dir)
            http = CelesteBridge(output_dir / "episode.tas")
            env = CelesteRoomEnv(LockstepBridge(http))
            manifest = runtime.collect(args.game_dir, http._prefix_lines())
            problems = runtime.check(manifest, runtime.load_pins())
            if problems and not args.allow_runtime_mismatch:
                print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
                return 2
            results["runtime_problems"] = problems
            results["attributable"] = runtime.attributable(git, problems)
            data, results["provenance"] = record(env, demos)
            save(output_dir / "dataset.npz", data)

        train, holdout = split_by_trajectory(data, args.holdout, args.seed)
        print(f"\n{len(data)} frames over {len(data.trajectories)} demonstrations: "
              f"{len(train.trajectories)} to train on, {len(holdout.trajectories)} held out")
        model = PPO(CelestePolicy, SpacesOnly(), policy_kwargs=policy_kwargs(), device="cpu", seed=args.seed)
        results["before"] = {"train": accuracy(model.policy, train), "holdout": accuracy(model.policy, holdout)}
        results["cloning"] = clone(model.policy, train, holdout, args.epochs, args.batch_size,
                                   args.learning_rate, args.seed, report_every=max(1, args.epochs // 10))
        final = results["cloning"]["history"][-1]
        print(f"\n{'':>10} {'input acc':>10} {'baseline':>9} {'frame acc':>10} {'baseline':>9}")
        for name in ("train", "holdout"):
            m = final[name]
            print(f"{name:>10} {m['input_accuracy']:>10.4f} {m['always_zero_input_accuracy']:>9.4f} "
                  f"{m['frame_accuracy']:>10.4f} {m['always_zero_frame_accuracy']:>9.4f}")
        model.save(output_dir / "cloned.zip")

        if not args.no_play and env is not None:
            print(f"\nPlaying the cloned policy: {args.eval_episodes} stochastic episodes plus one deterministic")
            played = SupervisedPPO.load(output_dir / "cloned.zip", env=env, device="cpu")
            stochastic = [run_episode(played, env, deterministic=False) for _ in range(args.eval_episodes)]
            deterministic = run_episode(played, env, deterministic=True)
            successes = [e for e in stochastic if e["ending"] == "success"]
            results["play"] = {
                "episodes": len(stochastic),
                "success_rate": len(successes) / len(stochastic),
                "endings": {e: sum(1 for x in stochastic if x["ending"] == e)
                            for e in {x["ending"] for x in stochastic}},
                "success_lengths": [e["length"] for e in successes],
                "median_max_x": float(np.median([e["max_x"] for e in stochastic if e["max_x"] is not None])),
                "best_max_x": max(e["max_x"] for e in stochastic if e["max_x"] is not None),
                "deterministic": deterministic,
            }
            p = results["play"]
            print(f"  success rate {p['success_rate']:.0%} ({len(successes)}/{len(stochastic)}), "
                  f"median max x {p['median_max_x']}, best {p['best_max_x']}, "
                  f"deterministic {deterministic['ending']}")
    finally:
        if env is not None:
            env.close()
        if game is not None:
            game.close()

    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
