"""Randomness and confident-choice figures for the PPO anchor pilot, every arm and the control (read-only, no game).

Run from the repo root with the RL interpreter, after training:
    .venv-rl/Scripts/python.exe scripts/describe_anchor_pilot.py --output <new path>

As config/ppo-anchor-pilot.json declares (measurements.randomness and confident_choices), on the same frames for
every run:
- **Room 1:** only the 6 held-back episodes of donor 3 + k's recording (split seed k): the episodes neither the SD-k
  clone nor any anchor ever fitted. On the 19 fitted episodes the anchor arms would be measured on their own
  training data.
- **Room 2:** the Room 2 demonstration dataset (no arm is anchored there).

For the SD-k clone, then every 100k checkpoint of each run (the control's SD-k runs and the pilot's A1, A10, E0):
per-room randomness in bits per frame (the summed Bernoulli entropy of the enabled inputs over ln 2: how many fair
coin tosses the policy's choices amount to). At 500k: of the donor's confident held-back Room 1 choices (donor
probability below 0.1 or above 0.9), the share the run still makes confidently the same way, now makes uncertainly,
or makes confidently the opposite way.

Every donor, clone, recording and checkpoint is checked against its committed pin: donors in retention-pilot.json,
clones in the SD training plan, recordings in the anchor pilot plan, the control's checkpoints in the SD evaluation
plan, and the arms' checkpoints in config/campaign-ppo-anchor-eval.json (so this runs after that plan is committed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl.cloning import OBS_KEYS, Demonstrations, split_by_trajectory  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.texthash import text_sha256  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

ENABLED = [i for i, name in enumerate(ACTION_INPUTS) if name not in MENU_INPUTS]
ROOM2_DATASET = "runs/clone/chapter-1-room-2/20260924-014351/dataset.npz"
STEPS = ["step_000100352", "step_000200704", "step_000301056", "step_000401408", "step_000501760"]
ARMS = ("A1", "A10", "E0")
PLANS = {"pilot": "config/ppo-anchor-pilot.json", "sd_train": "config/campaign-mixed-self-distillation-train.json",
         "sd_eval": "config/campaign-mixed-self-distillation-ppo-eval.json", "arm_eval": "config/campaign-ppo-anchor-eval.json",
         "donors": "config/retention-pilot.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def entropy_bits(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float((-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).sum(axis=1).mean())


def confident_breakdown(donor: np.ndarray, policy: np.ndarray) -> dict:
    """Of the donor's confident choices (< 0.1 or > 0.9): still confident the same way, uncertain, flipped."""
    confident = (donor < 0.1) | (donor > 0.9)
    high = donor > 0.9
    same = np.where(high, policy > 0.9, policy < 0.1)[confident]
    flipped = np.where(high, policy < 0.1, policy > 0.9)[confident]
    return {"donor_confident_choices": int(confident.sum()), "still_confident_same": round(float(same.mean()), 3),
            "now_uncertain": round(float(1 - same.mean() - flipped.mean()), 3),
            "flipped_confident_opposite": round(float(flipped.mean()), 3)}


def probabilities(checkpoint: Path, obs: dict[str, np.ndarray]) -> np.ndarray:
    policy = SupervisedPPO.load(checkpoint, device="cpu").policy
    policy.set_training_mode(False)
    out = []
    with th.no_grad():
        for start in range(0, len(obs["player"]), 512):
            batch = {key: th.as_tensor(value[start:start + 512]) for key, value in obs.items()}
            out.append(policy.get_distribution(batch).distribution.probs.numpy()[:, ENABLED])
    return np.concatenate(out).astype(np.float64)


def load_frames(path: Path, expected: str | None) -> Demonstrations:
    if expected is not None and sha256(path) != expected:
        raise SystemExit(f"{path} does not match its pinned sha256")
    stored = np.load(path)
    return Demonstrations({key: stored[f"obs_{key}"] for key in OBS_KEYS}, stored["actions"], stored["trajectory"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plans = {name: load_json(REPO / path) for name, path in PLANS.items()}
    pins = {run["checkpoint"]: run["checkpoint_sha256"] for key in ("sd_eval", "arm_eval") for run in plans[key]["runs"]}
    for clone in plans["sd_train"]["clones"].values():
        pins[f"{clone['dir']}/cloned.zip"] = clone["sha256"]
    donors = plans["donors"]["donors"]["runs"]
    recordings = plans["pilot"]["anchor_data"]["recordings"]

    def checked(relative: str) -> Path:
        if relative not in pins:
            raise SystemExit(f"{relative} is not pinned in any committed plan")
        if sha256(REPO / relative) != pins[relative]:
            raise SystemExit(f"{relative} does not match its pinned sha256")
        return REPO / relative

    room2 = load_frames(REPO / ROOM2_DATASET, None).obs
    out = {"label": "descriptive; companion to the PPO anchor pilot analysis; aggregates only",
           "analysis_code_sha256": text_sha256(Path(__file__)),
           "inputs": {**{path: text_sha256(REPO / path) for path in PLANS.values()},
                      ROOM2_DATASET: sha256(REPO / ROOM2_DATASET)},
           "frames": {"room1": "the 6 held-back episodes of donor 3 + k's recording (split seed k)",
                      "room2": ROOM2_DATASET},
           "randomness_bits_per_frame": {}, "confident_room1_choices_at_500k": {}}
    for k in range(4):
        seed = str(3 + k)
        donor = REPO / donors[seed]["checkpoint"]
        if sha256(donor) != donors[seed]["sha256"]:
            raise SystemExit(f"donor {seed} does not match its pinned sha256")
        recording = load_frames(REPO / recordings[seed]["play"] / "dataset.npz", recordings[seed]["dataset_sha256"])
        _, held = split_by_trajectory(recording, 0.25, k)
        room1 = held.obs
        donor_room1 = probabilities(donor, room1)
        runs = {"control": f"runs/train/mixed-self-distillation-pilot-SD-k{k}",
                **{arm: f"runs/train/ppo-anchor-pilot-{arm}-k{k}" for arm in ARMS}}
        clone = checked(f"{plans['sd_train']['clones'][f'SD-k{k}']['dir']}/cloned.zip")
        clone_row = {"room1": round(entropy_bits(probabilities(clone, room1)), 3),
                     "room2": round(entropy_bits(probabilities(clone, room2)), 3)}
        table = {"donor": {"room1": round(entropy_bits(donor_room1), 3)}, "clone": clone_row}
        for name, folder in runs.items():
            curve = {}
            for step in STEPS:
                checkpoint = checked(f"{folder}/checkpoints/{step}.zip")
                p1 = probabilities(checkpoint, room1)
                curve[step] = {"room1": round(entropy_bits(p1), 3), "room2": round(entropy_bits(probabilities(checkpoint, room2)), 3)}
                if step == STEPS[-1]:
                    out["confident_room1_choices_at_500k"][f"{name}-k{k}"] = confident_breakdown(donor_room1, p1)
            table[name] = curve
        out["randomness_bits_per_frame"][f"k{k}"] = table
        print(f"k{k}: held-back Room 1 frames {len(held)} from episodes {sorted(held.trajectories.tolist())}")
        for name in runs:
            print(f"   {name}: 500k Room 1 {table[name][STEPS[-1]]['room1']} bits, Room 2 {table[name][STEPS[-1]]['room2']}; "
                  f"{out['confident_room1_choices_at_500k'][f'{name}-k{k}']}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
