"""Descriptive figures for the mixed self-distillation write-up that the analyzers do not compute (read-only, no game).

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/describe_mixed_self_distillation.py

Inputs are committed records: config/retention-pilot.json (donors), config/mixed-self-distillation-pilot.json (the
pinned recordings), config/campaign-mixed-self-distillation-train.json (the SD-k clones), and the training runs'
checkpoints (SD-k, M-k, R-k). Every donor, clone and step checkpoint is checked against the sha256 pinned in its
committed plan (donors: retention-pilot.json; clones: the training plans; steps: the evaluation plans).

Frames: Room 1 = donor 3 + k's own recorded play (all frames); Room 2 = the Room 2 demonstration dataset (all frames).
Enabled inputs only.
1. **Randomness per room:** mean over frames of the summed Bernoulli entropy of the enabled inputs, in nats and in
   bits per frame (bits = nats / ln 2: the number of fair coin tosses the policy's choices amount to; 0 = the same
   buttons every time), for the donor, the SD-k clone, SD-k at 100k / 200k / 500k, M-k clone and 500k, and R-k 500k.
2. **The donor's confident Room 1 choices at 500k** (donor probability below 0.1 or above 0.9): the share the SD-k
   500k policy still makes confidently the same way, now makes uncertainly (0.1 to 0.9), or makes confidently the
   opposite way.
3. **Output saturation:** median absolute logit on Room 2 frames for the SD-k and M-k clones.

Writes docs/results/mixed-self-distillation-descriptive.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import math

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl.cloning import OBS_KEYS  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.texthash import matches_text_hash, text_sha256  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

ENABLED = [i for i, name in enumerate(ACTION_INPUTS) if name not in MENU_INPUTS]
ROOM2_DATASET = "runs/clone/chapter-1-room-2/20260924-014351/dataset.npz"
OUTPUT = REPO / "docs" / "results" / "mixed-self-distillation-descriptive.json"
STEPS = {"100k": "step_000100352.zip", "200k": "step_000200704.zip", "500k": "step_000501760.zip"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frames(path: Path, expected: str | None = None) -> dict[str, th.Tensor]:
    if expected is not None and sha256(path) != expected:
        raise SystemExit(f"{path} does not match its recorded sha256")
    stored = np.load(path)
    return {key: th.as_tensor(stored[f"obs_{key}"]) for key in OBS_KEYS}


def logits(checkpoint: Path, obs: dict[str, th.Tensor]) -> np.ndarray:
    policy = SupervisedPPO.load(checkpoint, device="cpu").policy
    policy.set_training_mode(False)
    out = []
    with th.no_grad():
        for start in range(0, len(obs["player"]), 512):
            batch = {key: value[start:start + 512] for key, value in obs.items()}
            out.append(policy.get_distribution(batch).distribution.logits.numpy()[:, ENABLED])
    return np.concatenate(out).astype(np.float64)


def entropy(z: np.ndarray) -> dict:
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    nats = float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).sum(axis=1).mean())
    return {"nats": round(nats, 3), "bits": round(nats / math.log(2), 3)}


def pinned_checkpoints() -> dict[str, str]:
    """checkpoint path -> sha256 as pinned in the committed training and evaluation plans."""
    pins = {}
    for plan in ("config/campaign-mixed-self-distillation-ppo-eval.json", "config/campaign-mixed-imitation-pilot-eval.json",
                 "config/campaign-retention-pilot-eval.json"):
        for run in load_json(REPO / plan)["runs"]:
            pins[run["checkpoint"]] = run["checkpoint_sha256"]
    for plan in ("config/campaign-mixed-self-distillation-train.json", "config/campaign-mixed-imitation-pilot-train.json"):
        for clone in load_json(REPO / plan)["clones"].values():
            pins[f"{clone['dir']}/cloned.zip"] = clone["sha256"]
    return pins


def verified(path: Path, pins: dict[str, str]) -> Path:
    key = path.relative_to(REPO).as_posix()
    if key not in pins:
        raise SystemExit(f"{key} is not pinned in any committed plan")
    if sha256(path) != pins[key]:
        raise SystemExit(f"{key} does not match its pinned sha256")
    return path


def main() -> int:
    donors = load_json(REPO / "config/retention-pilot.json")["donors"]["runs"]
    recordings = load_json(REPO / "config/mixed-self-distillation-pilot.json")["room1_data"]["recordings"]
    train_plan_path = REPO / "config/campaign-mixed-self-distillation-train.json"
    clones = load_json(train_plan_path)["clones"]
    mixed_clones = load_json(REPO / "config/campaign-mixed-imitation-pilot-train.json")["clones"]
    room2 = frames(REPO / ROOM2_DATASET)
    pins = pinned_checkpoints()
    out = {"label": "descriptive; companion to docs/results/mixed-self-distillation-ppo.json; aggregates only",
           "analysis_code_sha256": text_sha256(Path(__file__)),
           "inputs": {"config/campaign-mixed-self-distillation-train.json": text_sha256(train_plan_path),
                      ROOM2_DATASET: sha256(REPO / ROOM2_DATASET)},
           "entropy_per_frame": {}, "confident_room1_choices_at_500k": {}, "median_abs_logit_room2": {}}
    for k in range(4):
        seed = str(3 + k)
        donor = REPO / donors[seed]["checkpoint"]
        if sha256(donor) != donors[seed]["sha256"]:
            raise SystemExit(f"donor {seed} does not match its pinned sha256")
        clone = REPO / clones[f"SD-k{k}"]["dir"] / "cloned.zip"
        if sha256(clone) != clones[f"SD-k{k}"]["sha256"]:
            raise SystemExit(f"SD-k{k} clone does not match its pinned sha256")
        room1 = frames(REPO / recordings[seed]["play"] / "dataset.npz", recordings[seed]["dataset_sha256"])
        sd_run = REPO / f"runs/train/mixed-self-distillation-pilot-SD-k{k}/checkpoints"
        policies = {"donor": donor, "SD clone": clone,
                    **{f"SD {s}": verified(sd_run / f, pins) for s, f in STEPS.items()},
                    "M clone": verified(REPO / mixed_clones[str(k)]["dir"] / "cloned.zip", pins),
                    "M 500k": verified(REPO / f"runs/train/mixed-imitation-pilot-M-k{k}/checkpoints/step_000501760.zip",
                                       pins),
                    "R 500k": verified(REPO / f"runs/train/retention-pilot-R-k{k}/checkpoints/step_000501760.zip", pins)}
        table, room1_logits = {}, {}
        for name, path in policies.items():
            z1, z2 = logits(path, room1), logits(path, room2)
            room1_logits[name] = z1
            table[name] = {"room1": entropy(z1), "room2": entropy(z2)}
            if name in ("SD clone", "M clone"):
                out["median_abs_logit_room2"][f"{name.split()[0]}-k{k}"] = round(float(np.median(np.abs(z2))), 2)
        out["entropy_per_frame"][f"k{k}"] = table
        donor_p = 1 / (1 + np.exp(-room1_logits["donor"]))
        final_p = 1 / (1 + np.exp(-room1_logits["SD 500k"]))
        confident = (donor_p < 0.1) | (donor_p > 0.9)
        high = donor_p > 0.9
        same = np.where(high, final_p > 0.9, final_p < 0.1)[confident]
        flipped = np.where(high, final_p < 0.1, final_p > 0.9)[confident]
        out["confident_room1_choices_at_500k"][f"SD-k{k}"] = {
            "donor_confident_choices": int(confident.sum()), "still_confident_same": round(float(same.mean()), 3),
            "now_uncertain": round(float(1 - same.mean() - flipped.mean()), 3),
            "flipped_confident_opposite": round(float(flipped.mean()), 3)}
        print(f"k{k} (bits): " + "; ".join(f"{n} R1 {v['room1']['bits']} R2 {v['room2']['bits']}"
                                           for n, v in table.items()))
        print(f"   confident choices at 500k: {out['confident_room1_choices_at_500k'][f'SD-k{k}']}")
    print("median |logit| on Room 2:", out["median_abs_logit_room2"])
    # How the Room 1 test episodes ended at 500k, from the committed evaluation records of the PPO-stage campaign.
    analysis = load_json(REPO / "docs/results/mixed-self-distillation-ppo.json")
    summary_path = next(p for p in sorted((REPO / "runs/campaign").glob("*-mixed-self-distillation-ppo-eval-v1"))
                        if (p / "summary.json").exists()
                        and matches_text_hash(REPO / "config/campaign-mixed-self-distillation-ppo-eval.json",
                                              load_json(p / "summary.json")["plan_sha256"])
                        and load_json(p / "summary.json")["commit"] == analysis["evaluation_commit"])
    out["inputs"][summary_path.relative_to(REPO).as_posix() + "/summary.json"] = sha256(summary_path / "summary.json")
    out["room1_endings_at_500k"] = {}
    for record in load_json(summary_path / "summary.json")["results"]:
        if record["id"].startswith("room1-SD-") and record["id"].endswith("step_000501760"):
            result_path = REPO / record["artifact"]["result_file"]
            if sha256(result_path) != record["artifact"]["result_sha256"]:
                raise SystemExit(f"{record['id']}: result file changed since the run")
            out["room1_endings_at_500k"][record["id"].split("-step")[0][6:]] = load_json(result_path)["endings"]
    print("Room 1 endings at 500k:", out["room1_endings_at_500k"])
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Results: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
