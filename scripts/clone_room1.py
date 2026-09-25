"""Clone demonstrated play of one hash-pinned room task, then measure the cloned policy by playing it.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/clone_room1.py
    .venv-rl/Scripts/python.exe scripts/clone_room1.py --task-definition config/room2.json \
        --demonstrations config/demonstrations-room2.json --heldout config/heldout_starts-room2.json
    .venv-rl/Scripts/python.exe scripts/clone_room1.py --dataset runs/clone/<run>/dataset.npz   # refit, no game
    ... --init-from runs/train/<run>/checkpoints/latest.zip --init-from-sha256 <sha256>   # start from those weights

Both modes require the explicit demonstration and held-out manifests. A dataset refit also requires the
`dataset.manifest.json` written beside it, and verifies all three hashes before fitting.

Three stages, and only the third answers the project's question.

**Stage 1 (needs the game): record what the demonstrations saw.** A route is a list of input lines; cloning
needs the observation the policy would have had at each frame, so every demonstration is replayed through
`CelesteRoomEnv` from the task start and its observations captured. A replay that does not end in
`success` is dropped with a note rather than trained on. The pairs are saved so later fits need no game.

**Stage 2 (CPU): clone.** Whole demonstrations are held out, never frames (see `celeste_rl/cloning.py` for
why). Held-out accuracy is reported against the always-zero baseline, because the inputs are sparse.

**Stage 3 (needs the game): play.** The cloned policy runs the evaluation protocol from the task's canonical
start, stochastic and deterministic. **This is the number that matters.** Held-out accuracy says the policy
predicts the right button most of the time; only playing says whether it clears the room, because a cloned
policy's first mistake puts it in a state no demonstration visited and errors compound from there.

Demonstration sources are kept separate in the records, because they are not equivalent:

- `route`: independent Go-Explore style searches (`scripts/find_room_exit.py`), about 2.2 inputs held per
  frame, which is what real play looks like.
- `archive`: clears extracted from the training archives (`scripts/extract_demonstrations.py`), about 8.4
  inputs per frame, because they are the twelve-button policy's solutions.

Room 1 results go to `runs/clone/<timestamp>/`; named tasks use `runs/clone/<task>/<timestamp>/`.
"""
from __future__ import annotations

import argparse
import hashlib
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
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.cloning import OBS_KEYS, Demonstrations, accuracy, clone, split_by_trajectory  # noqa: E402
from celeste_rl.demonstrations import (  # noqa: E402
    DemonstrationManifestError,
    materialize_demonstrations,
    reject_heldout_overlap,
    validate_demonstration_manifest,
    verify_dataset_manifest,
    write_dataset_manifest,
)
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.heldout import HeldoutManifestError, validate_manifest as validate_heldout_manifest  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.observation import observation_space  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.schema import FINGERPRINT as SCHEMA_FINGERPRINT  # noqa: E402
from celeste_rl.tasks import (  # noqa: E402
    TaskDefinitionError,
    resolve_task_definition,
    task_identity,
)
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.policy import CelestePolicy, policy_kwargs  # noqa: E402
from celeste_rl.training.run import run_episode  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

DEFAULT_DEMONSTRATIONS = REPO / "config" / "demonstrations.json"
DEFAULT_HELDOUT = REPO / "config" / "heldout_starts.json"


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


def initial_model(seed: int, init_from: Path | None = None) -> PPO:
    """The policy to clone into: fresh weights, or a saved policy's weights (actor and critic).

    With `init_from` the saved policy's weights replace the fresh ones, and the global generators are re-seeded
    with `seed` afterwards, because SB3's load() re-seeds them from the saved run's own seed (as in
    `celeste_rl/training/run.py`). The architecture must match exactly; a mismatch refuses.
    """
    model = PPO(CelestePolicy, SpacesOnly(), policy_kwargs=policy_kwargs(), device="cpu", seed=seed)
    if init_from is not None:
        donor = SupervisedPPO.load(init_from, device="cpu")
        # A strict load checks weight shapes only; equal spaces also rule out a different observation layout.
        if donor.observation_space != model.observation_space or donor.action_space != model.action_space:
            raise ValueError(f"{init_from} was saved with different observation or action spaces")
        model.policy.load_state_dict(donor.policy.state_dict())
        model.set_random_seed(seed)
    return model


def donor_provenance(init_from: Path) -> dict:
    """Who trained the --init-from policy, from its training run's manifest or its clone's results.json.

    A training checkpoint (runs/train/<run>/checkpoints/*.zip) must come from a finished run whose sessions ran
    clean with the current observation schema and the current disabled inputs; otherwise the weights would be read
    under a different meaning. A clone (runs/clone/.../cloned.zip) must have been fitted from a clean tree and
    records its commit. Anything else refuses.
    """
    manifest_path = init_from.parent.parent / "manifest.json"
    if init_from.parent.name == "checkpoints" and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sessions = manifest["sessions"] if isinstance(manifest["sessions"], list) else [manifest["sessions"]]
        fingerprints = sorted({s["provenance"]["runtime"]["schema"]["fingerprint"] for s in sessions})
        disabled = manifest["config"].get("disabled_inputs")
        if manifest["status"] != "finished":
            raise ValueError(f"{init_from}: its training run is {manifest['status']}, not finished")
        if fingerprints != [SCHEMA_FINGERPRINT] or list(disabled or []) != list(MENU_INPUTS) \
                or any(s["provenance"]["uncommitted_changes"] for s in sessions):
            raise ValueError(f"{init_from}: trained with schema {fingerprints}, disabled inputs {disabled} or "
                             f"uncommitted changes; this code uses schema {SCHEMA_FINGERPRINT} and {list(MENU_INPUTS)}")
        return {"kind": "training run", "manifest": manifest_path.as_posix(), "status": manifest["status"],
                "commits": sorted({s["provenance"]["commit"] for s in sessions}), "schema_fingerprint":
                SCHEMA_FINGERPRINT, "disabled_inputs": list(disabled), "accepted_steps": manifest["accepted_steps"]}
    results_path = init_from.parent / "results.json"
    if init_from.name == "cloned.zip" and results_path.exists():
        record = json.loads(results_path.read_text(encoding="utf-8"))
        if record.get("uncommitted_changes") is not False:
            raise ValueError(f"{init_from}: the clone was fitted with uncommitted changes (or does not say)")
        return {"kind": "clone", "results": results_path.as_posix(), "commits": [record["commit"]]}
    raise ValueError(f"{init_from}: no training manifest or clone record beside it, so its provenance is unknown")


def load_manifests(demonstrations_path: Path, heldout_path: Path, routes_only: bool,
                   expected_task: dict) -> tuple[dict, list[dict], dict]:
    """Validate both frozen manifests and reject content overlap before any game or model work."""
    demonstrations_manifest = json.loads(demonstrations_path.read_text(encoding="utf-8"))
    demonstration_entries = validate_demonstration_manifest(demonstrations_manifest, expected_task)
    if routes_only:
        demonstration_entries = [entry for entry in demonstration_entries if entry["kind"] == "route"]
    if len(demonstration_entries) < 2:
        raise DemonstrationManifestError("fewer than two demonstrations remain after filtering")
    heldout_manifest = json.loads(heldout_path.read_text(encoding="utf-8"))
    heldout_entries = validate_heldout_manifest(heldout_manifest, expected_task)
    reject_heldout_overlap(demonstration_entries, heldout_entries)
    return demonstrations_manifest, demonstration_entries, heldout_manifest


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
    parser.add_argument("--task-definition", type=Path,
                        help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--dataset", type=Path, help="refit these recorded pairs instead of replaying")
    parser.add_argument("--demonstrations", type=Path,
                        help="explicit frozen demonstration-source manifest")
    parser.add_argument("--heldout", type=Path,
                        help="frozen held-out manifest used for the overlap check")
    parser.add_argument("--routes-only", action="store_true",
                        help="use only route entries declared in the demonstration manifest")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--no-play", action="store_true", help="skip stage 3")
    parser.add_argument("--init-from", type=Path,
                        help="start cloning from this saved policy's weights instead of fresh ones")
    parser.add_argument("--init-from-sha256", help="required with --init-from: that file's SHA-256")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    try:
        definition = resolve_task_definition(args.task_definition)
        identity = task_identity(definition)
    except TaskDefinitionError as error:
        parser.error(str(error))
    if args.demonstrations is None:
        if args.task_definition:
            parser.error("--demonstrations is required with --task-definition")
        args.demonstrations = DEFAULT_DEMONSTRATIONS
    if args.heldout is None:
        if args.task_definition:
            parser.error("--heldout is required with --task-definition")
        args.heldout = DEFAULT_HELDOUT
    if (args.init_from is None) != (args.init_from_sha256 is None):
        parser.error("--init-from and --init-from-sha256 go together")

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    try:
        demonstrations_manifest, demonstration_entries, heldout_manifest = load_manifests(
            args.demonstrations, args.heldout, args.routes_only, identity)
        materialized_demonstrations = materialize_demonstrations(demonstration_entries, REPO, identity)
    except (OSError, json.JSONDecodeError, DemonstrationManifestError, HeldoutManifestError,
            TaskDefinitionError) as error:
        print(f"Cannot establish demonstration/held-out separation: {error}")
        return 2
    if args.init_from is not None:
        try:
            init_sha256 = hashlib.sha256(args.init_from.read_bytes()).hexdigest()
        except OSError as error:
            print(f"Cannot read --init-from: {error}")
            return 2
        if init_sha256 != args.init_from_sha256:
            print(f"--init-from sha256 is {init_sha256}, not the pinned {args.init_from_sha256}")
            return 2
        try:
            init_provenance = donor_provenance(args.init_from)
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as error:
            print(f"Cannot use --init-from: {error}")
            return 2
    output_root = REPO / "runs" / "clone"
    if args.task_definition:
        output_root /= definition.name
    output_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    results = {
        **git,
        "task": identity,
        "args": {k: str(v) for k, v in vars(args).items()},
        "manifests": {
            "demonstrations": str(args.demonstrations),
            "demonstrations_sha256": demonstrations_manifest["sha256"],
            "demonstrations_file_sha256": hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
            "heldout": str(args.heldout),
            "heldout_sha256": heldout_manifest["sha256"],
            "heldout_file_sha256": hashlib.sha256(args.heldout.read_bytes()).hexdigest(),
        },
    }

    game = env = None
    try:
        if args.dataset:
            audit = verify_dataset_manifest(args.dataset, demonstrations_manifest["sha256"],
                                            heldout_manifest["sha256"], identity)
            declared_hashes = {entry["route_sha256"] for entry in demonstration_entries}
            if not set(audit["demonstration_route_sha256s"]).issubset(declared_hashes):
                raise DemonstrationManifestError(
                    "dataset provenance contains a route absent from the demonstration manifest")
            data = load(args.dataset)
            results["provenance"] = [{"dataset": str(args.dataset), **audit}]
            print(f"Refitting {len(data)} frames from {args.dataset}")
        else:
            print(f"{len(materialized_demonstrations)} demonstrations to replay")
            game = GameSession(args.game_dir)
            http = CelesteBridge(output_dir / "episode.tas")
            env = CelesteRoomEnv(LockstepBridge(http), task=definition.task, task_start=definition.start)
            manifest = runtime.collect(args.game_dir, http._prefix_lines())
            problems = runtime.check(manifest, runtime.load_pins())
            if problems and not args.allow_runtime_mismatch:
                print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
                return 2
            results["runtime_problems"] = problems
            results["attributable"] = runtime.attributable(git, problems)
            data, results["provenance"] = record(env, materialized_demonstrations)
            dataset = output_dir / "dataset.npz"
            save(dataset, data)
            used_hashes = [item["route_sha256"] for item in results["provenance"] if item["used"]]
            results["dataset_manifest"] = write_dataset_manifest(
                dataset, demonstrations_manifest["sha256"], heldout_manifest["sha256"], used_hashes, identity)

        train, holdout = split_by_trajectory(data, args.holdout, args.seed)
        print(f"\n{len(data)} frames over {len(data.trajectories)} demonstrations: "
              f"{len(train.trajectories)} to train on, {len(holdout.trajectories)} held out")
        model = initial_model(args.seed, args.init_from)
        if args.init_from is not None:
            results["init_from"] = {"path": args.init_from.as_posix(), "sha256": args.init_from_sha256,
                                    "provenance": init_provenance}
            print(f"Cloning starts from the weights of {args.init_from}")
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
    except DemonstrationManifestError as error:
        print(f"Cannot verify cloning dataset provenance: {error}")
        return 2
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
