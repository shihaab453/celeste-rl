"""Record a saved policy's own play: the observations it saw and the actions the game applied, from the canonical start.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/record_policy_play.py --checkpoint runs/train/<run>/checkpoints/latest.zip \
        --checkpoint-sha256 <sha256> --episodes 25 --seed 20260926

Why: to copy a policy into another network (self-distillation), the copy has to learn the policy's behaviour on the
states the policy itself visits, not only on demonstration routes. This plays stochastic episodes from the task's
canonical start only, so no held-out state is ever used, and writes runs/policy-play/<task>/<timestamp>/:
dataset.npz (obs_* arrays, applied actions, and each frame's episode index as `trajectory`) and play.json (the
checkpoint and its sha256, task, seed, episodes and their endings, the dataset's sha256, commit and runtime). The
action targets for copying are computed later, offline, from the policy itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.cloning import OBS_KEYS, Demonstrations  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.tasks import TaskDefinitionError, resolve_task_definition, task_identity  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402


def record_episodes(model, env, episodes: int, deterministic: bool = False) -> tuple[Demonstrations, list[dict]]:
    """Play `episodes` episodes from the canonical start and keep every (observation, applied action) pair."""
    rows, actions, trajectory, summaries = [], [], [], []
    for index in range(episodes):
        obs, _ = env.reset(options={"canonical": True})
        length, ending = 0, None
        while True:
            rows.append({key: np.asarray(obs[key]) for key in OBS_KEYS})
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            actions.append(np.asarray(info["applied_action"], dtype=np.int8))
            trajectory.append(index)
            length += 1
            if terminated or truncated:
                ending = info.get("ending")
                break
        summaries.append({"episode": index, "ending": ending, "frames": length})
    obs = {key: np.stack([row[key] for row in rows]) for key in OBS_KEYS}
    return Demonstrations(obs, np.stack(actions), np.asarray(trajectory)), summaries


def save(path: Path, data: Demonstrations) -> None:
    np.savez_compressed(path, actions=data.actions, trajectory=data.trajectory,
                        **{f"obs_{key}": data.obs[key] for key in OBS_KEYS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="the policy's sampling seed for these episodes")
    parser.add_argument("--task-definition", type=Path, help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    try:
        definition = resolve_task_definition(args.task_definition)
        identity = task_identity(definition)
    except TaskDefinitionError as error:
        parser.error(str(error))
    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != args.checkpoint_sha256:
        print(f"--checkpoint sha256 is {checkpoint_sha256}, not the pinned {args.checkpoint_sha256}")
        return 2

    output_dir = REPO / "runs" / "policy-play" / definition.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    kwargs = {"task": definition.task, "task_start": definition.start} if args.task_definition else {}
    env = CelesteRoomEnv(LockstepBridge(http), **kwargs)
    try:
        manifest = runtime.collect(args.game_dir, http._prefix_lines())
        problems = runtime.check(manifest, runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        model = SupervisedPPO.load(args.checkpoint, env=env, device="cpu")
        model.set_random_seed(args.seed)  # SB3's load re-seeds from the checkpoint's own seed
        print(f"Recording {args.episodes} stochastic canonical episodes of {args.checkpoint}, seed {args.seed}")
        data, summaries = record_episodes(model, env, args.episodes)
    finally:
        env.close()
        game.close()

    dataset = output_dir / "dataset.npz"
    save(dataset, data)
    record = {**git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
              "task": identity, "checkpoint": args.checkpoint.as_posix(), "checkpoint_sha256": checkpoint_sha256,
              "seed": args.seed, "episodes": args.episodes, "starts": "canonical only; no held-out state is read",
              "frames": len(data), "endings": dict(Counter(s["ending"] for s in summaries)), "summaries": summaries,
              "dataset": "dataset.npz", "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    (output_dir / "play.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"{len(data)} frames over {args.episodes} episodes, endings {record['endings']}. Results: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
