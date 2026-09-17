"""Can the policy network even reproduce a known solution? (Codex finding K4, the cheapest diagnostic.)

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/route_fit_check.py
    .venv-rl/Scripts/python.exe scripts/route_fit_check.py --dataset runs/route-fit/<run>/dataset.npz   # refit, no game

Two hundred million random steps taught the unshaped campaign nothing, and the checkpoints never left
near-uniform play. Before changing the reward, rule out the cheaper explanation: that the network and the
observation cannot represent the mapping from what the agent sees to what a solution does. This is a capacity
check, not a learning result. It deliberately fits and measures on the same frames: if the network cannot
memorise 285 frames of a recorded clear, no reward will help and the features or the network are the problem.

Stage 1 (needs the game): launch the game, replay `tests/fixtures/room1_exit_dash_route.json` through
`CelesteRoomEnv`, and keep every observation together with the action the environment actually applied. The
route must end in `success`; anything else means the replay is not the recorded clear and the check stops.
The pairs are saved to `dataset.npz` so later fits need no game.

Stage 2 (CPU, no game): build the training policy exactly as `scripts/train_room1.py` does (`policy_kwargs()`,
the same observation and action spaces) and fit its action head to those pairs with binary cross entropy.

Reported, all on the fitted frames:

  per-input accuracy     over the 21 enabled inputs, and over all 24 (the 3 disabled ones are always 0)
  always-zero baseline   the same accuracy for a network that predicts every input off, because the inputs are
                         sparse and a high per-input number can be worth nothing on its own
  exact-frame accuracy   frames where all 21 enabled inputs are right at once, the honest version of the question
  worst inputs           the enabled inputs with the lowest accuracy

Pass (Codex's bar): at least 95% per-input accuracy over the enabled inputs. Results and the fitted curve go to
runs/route-fit/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from celeste_rl import runtime  # noqa: E402
from celeste_rl.actions import parse_line  # noqa: E402
from celeste_rl.bridge import CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.observation import observation_space  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.policy import policy_kwargs  # noqa: E402

DEFAULT_ROUTE = REPO / "tests" / "fixtures" / "room1_exit_dash_route.json"
OBS_KEYS = ("player", "actions", "history_valid", "grid", "context")


class _SpacesOnly(gym.Env):
    """Enough of an environment for PPO to build its policy: the two spaces and nothing else.

    Stage 2 has no game, and the policy only needs the spaces. SB3 wraps this in a DummyVecEnv at construction,
    which reads the spaces and never resets.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = gym.spaces.MultiBinary(len(ACTION_INPUTS))

    def reset(self, *, seed=None, options=None):  # pragma: no cover - never called
        raise RuntimeError("route_fit_check builds the policy only; this environment is never stepped")

    def step(self, action):  # pragma: no cover - never called
        raise RuntimeError("route_fit_check builds the policy only; this environment is never stepped")


def record_route(env: CelesteRoomEnv, actions: list[str], max_steps: int) -> dict:
    """Replay the route and keep (observation, applied action) for every step that was taken."""
    observations: list[dict] = []
    applied: list[np.ndarray] = []
    obs, _ = env.reset()
    ending, info = None, {}
    for step, buttons in enumerate(actions[:max_steps]):
        action = parse_line(format_input_line(buttons))
        observations.append({key: np.asarray(obs[key]) for key in OBS_KEYS})
        obs, _reward, terminated, _truncated, info = env.step(action)
        applied.append(np.asarray(info["applied_action"], dtype=np.int8))
        if terminated:
            ending = info["ending"]
            break
    stacked = {key: np.stack([o[key] for o in observations]) for key in OBS_KEYS}
    return {"obs": stacked, "actions": np.stack(applied), "ending": ending,
            "steps": len(applied), "last_frame": info.get("frame")}


def collect(args) -> tuple[dict, dict]:
    """Stage 1: play the route in the real game. Returns the dataset and the run's provenance."""
    route = json.loads(args.route.read_text(encoding="utf-8"))
    game = GameSession(args.game_dir)
    http = CelesteBridge(args.output_root / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http))
    try:
        manifest = runtime.collect(args.game_dir, http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            raise SystemExit("Runtime differs from the pins:\n  " + "\n  ".join(problems))
        provenance = {"runtime": manifest, "runtime_problems": problems, "route": str(args.route)}
        print(f"Replaying {args.route.name}: {len(route['actions'])} recorded frames, "
              f"transition at step {route.get('transition_step')}")
        data = record_route(env, route["actions"], args.max_steps)
    finally:
        env.close()
        game.close()

    print(f"Replay ended after {data['steps']} steps with ending {data['ending']!r} at frame {data['last_frame']}")
    if data["ending"] != "success":
        raise SystemExit(f"The route did not clear the room (ending {data['ending']!r}); "
                         "the fit would be measured on the wrong frames")
    return data, provenance


def save_dataset(path: Path, data: dict) -> None:
    np.savez_compressed(path, actions=data["actions"], **{f"obs_{key}": data["obs"][key] for key in OBS_KEYS})


def load_dataset(path: Path) -> dict:
    stored = np.load(path)
    return {"obs": {key: stored[f"obs_{key}"] for key in OBS_KEYS}, "actions": stored["actions"],
            "ending": "success", "steps": len(stored["actions"]), "last_frame": None}


def fit(data: dict, seed: int, epochs: int, batch_size: int, learning_rate: float) -> dict:
    """Stage 2: fit the training policy's action head to the recorded pairs with binary cross entropy."""
    th.manual_seed(seed)
    model = PPO("MultiInputPolicy", _SpacesOnly(), policy_kwargs=policy_kwargs(), device="cpu", seed=seed)
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)

    observations = {key: th.as_tensor(value) for key, value in data["obs"].items()}
    targets = th.as_tensor(data["actions"]).float()
    count = targets.shape[0]
    enabled = th.as_tensor([name not in MENU_INPUTS for name in ACTION_INPUTS])

    def logits_for(index: th.Tensor) -> th.Tensor:
        batch = {key: value[index] for key, value in observations.items()}
        return policy.get_distribution(batch).distribution.logits

    generator = th.Generator().manual_seed(seed)
    curve = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = th.randperm(count, generator=generator)
        losses = []
        for start in range(0, count, batch_size):
            index = order[start:start + batch_size]
            loss = F.binary_cross_entropy_with_logits(logits_for(index), targets[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            with th.no_grad():
                policy.set_training_mode(False)
                predicted = (logits_for(th.arange(count)) > 0).float()
                policy.set_training_mode(True)
            accuracy = (predicted[:, enabled] == targets[:, enabled]).float().mean().item()
            curve.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 5),
                          "enabled_input_accuracy": round(accuracy, 4)})
            print(f"  epoch {epoch:4d}  loss {np.mean(losses):.5f}  enabled-input accuracy {accuracy:.4f}")

    policy.set_training_mode(False)
    with th.no_grad():
        predicted = (logits_for(th.arange(count)) > 0).float()
    correct = predicted == targets
    per_input = correct.float().mean(dim=0)
    zeros_per_input = (targets == 0).float().mean(dim=0)
    worst = sorted(((ACTION_INPUTS[i], round(per_input[i].item(), 4)) for i in range(len(ACTION_INPUTS))
                    if enabled[i]), key=lambda item: item[1])[:5]
    return {
        "frames": count,
        "epochs": epochs,
        "seconds": round(time.perf_counter() - started, 1),
        "enabled_input_accuracy": round(correct[:, enabled].float().mean().item(), 4),
        "all_input_accuracy": round(correct.float().mean().item(), 4),
        "always_zero_enabled_accuracy": round(zeros_per_input[enabled].mean().item(), 4),
        "exact_frame_accuracy": round(correct[:, enabled].all(dim=1).float().mean().item(), 4),
        "always_zero_exact_frame_accuracy": round((targets[:, enabled] == 0).all(dim=1).float().mean().item(), 4),
        "per_input_accuracy": {ACTION_INPUTS[i]: round(per_input[i].item(), 4) for i in range(len(ACTION_INPUTS))},
        "input_on_rate": {ACTION_INPUTS[i]: round(1 - zeros_per_input[i].item(), 4)
                          for i in range(len(ACTION_INPUTS))},
        "worst_enabled_inputs": worst,
        "curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--dataset", type=Path, help="refit these recorded pairs instead of launching the game")
    parser.add_argument("--max-steps", type=int, default=400, help="stop the replay after this many frames")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pass-accuracy", type=float, default=0.95)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    if git["uncommitted_changes"] and not args.allow_dirty:
        print(runtime.DIRTY_MESSAGE + "\n  " + "\n  ".join(git["changed_paths"]))
        return 2

    args.output_root = REPO / "runs" / "route-fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_root.mkdir(parents=True)

    if args.dataset:
        data, provenance = load_dataset(args.dataset), {"dataset": str(args.dataset)}
        print(f"Refitting {data['steps']} recorded frames from {args.dataset}")
    else:
        data, provenance = collect(args)
        save_dataset(args.output_root / "dataset.npz", data)

    print(f"Fitting the policy to {data['steps']} frames for {args.epochs} epochs on CPU")
    summary = fit(data, args.seed, args.epochs, args.batch_size, args.learning_rate)
    passed = summary["enabled_input_accuracy"] >= args.pass_accuracy

    results = {**git, **provenance, "args": {k: str(v) for k, v in vars(args).items()},
               "ending": data["ending"], "pass_accuracy": args.pass_accuracy, "passed": passed, **summary}
    (args.output_root / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print()
    print(f"per-input accuracy (21 enabled)   {summary['enabled_input_accuracy']:.4f}"
          f"   always-zero baseline {summary['always_zero_enabled_accuracy']:.4f}")
    print(f"per-input accuracy (all 24)       {summary['all_input_accuracy']:.4f}")
    print(f"exact-frame accuracy              {summary['exact_frame_accuracy']:.4f}"
          f"   always-zero baseline {summary['always_zero_exact_frame_accuracy']:.4f}")
    print(f"worst enabled inputs              " +
          ", ".join(f"{name} {value:.3f}" for name, value in summary["worst_enabled_inputs"]))
    print(f"{'PASS' if passed else 'FAIL'}: per-input accuracy over the enabled inputs "
          f"{'reaches' if passed else 'is below'} {args.pass_accuracy:.2f}")
    print(f"Results: {args.output_root / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
