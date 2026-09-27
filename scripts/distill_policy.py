"""Copy a saved policy into a fresh network from its own action probabilities (self-distillation), no game needed.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/distill_policy.py --donor runs/train/<run>/checkpoints/latest.zip \
        --donor-sha256 <sha256> --play runs/policy-play/<task>/<timestamp> --seed 0
    .venv-rl/Scripts/python.exe scripts/distill_policy.py --donor ... --donor-sha256 ... \
        --demonstration-dataset runs/clone/<run>/dataset.npz --demonstrations config/demonstrations.json \
        --heldout config/heldout_starts.json --seed 0

The "coverage ceiling" question: can a network copied from a policy play like it at all, and does it matter which
states the copy is made on? The states come either from the donor's own recorded play (record_policy_play.py; the
recording must name the same donor by sha256) or from a verified demonstration dataset (checked against both frozen
manifests, as clone_room1.py --dataset does). The targets are the donor's own action probabilities on those states
(disabled inputs 0). Whole trajectories are held out (--holdout, split by --seed, as for cloning), a fresh network is
fitted with the same cloning settings, and agreement with the donor is reported on the fitted and held-out
trajectories: inputs and whole frames where the copy's most likely choice equals the donor's, and the mean absolute
difference of their probabilities. Writes runs/distill/<task>/<timestamp>/cloned.zip and results.json. Only playing
the copy (evaluate_heldout.py) answers the question; agreement here is the open-loop check.
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
sys.path.insert(0, str(REPO / "scripts"))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.cloning import Demonstrations, clone, distilled, split_by_trajectory  # noqa: E402
from celeste_rl.demonstrations import DemonstrationManifestError  # noqa: E402
from celeste_rl.heldout import HeldoutManifestError  # noqa: E402
from celeste_rl.policy_play import play_problems, verified_play  # noqa: F401,E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.tasks import TaskDefinitionError, resolve_task_definition, task_identity  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402
from clone_room1 import initial_model, load_manifests, verified_dataset  # noqa: E402

DISABLED = [ACTION_INPUTS.index(name) for name in MENU_INPUTS]


def agreement(student, donor, data: Demonstrations) -> dict:
    """How closely `student` follows `donor` on these observations (enabled inputs only)."""
    enabled = [i for i in range(len(ACTION_INPUTS)) if i not in DISABLED]
    probabilities = []
    for policy in (student, donor):
        policy.set_training_mode(False)
        with th.no_grad():
            probabilities.append(policy.get_distribution({k: th.as_tensor(v) for k, v in data.obs.items()})
                                 .distribution.probs.numpy()[:, enabled])
    ps, pd = probabilities
    same = (ps >= 0.5) == (pd >= 0.5)
    return {"frames": len(data), "trajectories": int(len(data.trajectories)),
            "input_agreement": round(float(same.mean()), 4), "frame_agreement": round(float(same.all(1).mean()), 4),
            "mean_abs_probability_difference": round(float(np.abs(ps - pd).mean()), 4)}


def fit_student(donor_policy, data: Demonstrations, holdout: float, seed: int, epochs: int, batch_size: int,
                learning_rate: float):
    """Split by trajectory, take the donor's probabilities as targets, fit a fresh network. Returns (model, record)."""
    train, held = split_by_trajectory(data, holdout, seed)
    train, held = distilled(donor_policy, train, DISABLED), distilled(donor_policy, held, DISABLED)
    student = initial_model(seed)
    before = {"fitted": agreement(student.policy, donor_policy, train),
              "held_out": agreement(student.policy, donor_policy, held)}
    history = clone(student.policy, train, held, epochs, batch_size, learning_rate, seed,
                    report_every=max(1, epochs // 10))
    after = {"fitted": agreement(student.policy, donor_policy, train),
             "held_out": agreement(student.policy, donor_policy, held)}
    return student, {"before": before, "after": after, "cloning": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-sha256", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--play", type=Path, help="a record_policy_play.py folder recorded from the same donor")
    source.add_argument("--demonstration-dataset", type=Path, help="a verified cloning dataset (needs its manifests)")
    parser.add_argument("--demonstrations", type=Path, help="with --demonstration-dataset: its demonstration manifest")
    parser.add_argument("--heldout", type=Path, help="with --demonstration-dataset: its held-out manifest")
    parser.add_argument("--task-definition", type=Path, help="hash-pinned room task; omit for the original Room 1 task")
    parser.add_argument("--routes-only", action="store_true")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.demonstration_dataset is not None and (args.demonstrations is None or args.heldout is None):
        parser.error("--demonstration-dataset needs --demonstrations and --heldout")
    try:
        identity = task_identity(resolve_task_definition(args.task_definition))
    except TaskDefinitionError as error:
        parser.error(str(error))
    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    donor_sha256 = hashlib.sha256(args.donor.read_bytes()).hexdigest()
    if donor_sha256 != args.donor_sha256:
        print(f"--donor sha256 is {donor_sha256}, not the pinned {args.donor_sha256}")
        return 2
    try:
        if args.play is not None:
            data, states = verified_play(args.play, donor_sha256, identity)
        else:
            demonstrations_manifest, entries, heldout_manifest = load_manifests(
                args.demonstrations, args.heldout, args.routes_only, identity)
            data, audit = verified_dataset(args.demonstration_dataset, demonstrations_manifest, entries,
                                           heldout_manifest, identity)
            states = {"kind": "demonstration routes", "dataset": args.demonstration_dataset.as_posix(), **audit}
    except (OSError, KeyError, json.JSONDecodeError, ValueError, DemonstrationManifestError,
            HeldoutManifestError) as error:
        print(f"Cannot use these states: {error}")
        return 2

    donor = SupervisedPPO.load(args.donor, device="cpu")
    print(f"Distilling {args.donor} into a fresh network on {len(data)} {states['kind']} frames "
          f"({len(data.trajectories)} trajectories), seed {args.seed}")
    student, fit = fit_student(donor.policy, data, args.holdout, args.seed, args.epochs, args.batch_size,
                               args.learning_rate)
    output_dir = REPO / "runs" / "distill" / identity["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    student.save(output_dir / "cloned.zip")
    results = {**git, "task": identity, "args": {k: str(v) for k, v in vars(args).items()},
               "donor": {"path": args.donor.as_posix(), "sha256": donor_sha256}, "states": states, **fit}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    for split in ("fitted", "held_out"):
        a = fit["after"][split]
        print(f"  {split}: agrees with the donor on {a['input_agreement']:.4f} of inputs, {a['frame_agreement']:.4f} of "
              f"whole frames, mean probability difference {a['mean_abs_probability_difference']:.4f}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
