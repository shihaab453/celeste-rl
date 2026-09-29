"""Generate the PPO anchor tie-break plans from the committed pilot plans (refuses to overwrite).

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/make_tiebreak_plans.py train
    .venv-rl/Scripts/python.exe scripts/make_tiebreak_plans.py eval runs/campaign/<ts>-ppo-anchor-tiebreak-train-v1/summary.json

train: config/campaign-ppo-tiebreak-train.json, 8 runs (E0 and A1 for k = 0 to 3, interleaved). Each is the pilot's
    command for the same arm and k (config/campaign-ppo-anchor-train.json) with ONLY --seed (40 + k -> 50 + k) and
    --run-dir changed, checked below; the clone is verified against its pinned sha256; the run limit is 240 minutes. The plan pins the declaration
    (config/ppo-anchor-tiebreak.json) and the pilot plan it was derived from.
eval: config/campaign-ppo-tiebreak-eval.json, for every run the Room 1 held-out evaluation at every step checkpoint,
    the Room 2 canonical final and the Room 2 v1 held-out final. The commands are the pilot's evaluation commands
    (config/campaign-ppo-anchor-eval.json) with only the checkpoint changed; every checkpoint is pinned by sha256, and
    each run's latest.zip must hold the same policy as its step_000501760.zip.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from celeste_rl.texthash import matches_text_hash, text_sha256  # noqa: E402

DECLARATION = REPO / "config/ppo-anchor-tiebreak.json"
PILOT_TRAIN = REPO / "config/campaign-ppo-anchor-train.json"
PILOT_EVAL = REPO / "config/campaign-ppo-anchor-eval.json"
TRAIN = REPO / "config/campaign-ppo-tiebreak-train.json"
EVAL = REPO / "config/campaign-ppo-tiebreak-eval.json"
UNREAD_SETS = ("heldout_starts-room1-v2.json", "heldout_starts-room2-v2.json")  # never in a command (plan rule)
ORDER = [(arm, k) for k in range(4) for arm in ("A1", "E0")]  # interleaved, so neither arm is run first or last
PILOT_SEED, NEW_SEED = 40, 50
# The limit only catches hangs. The pilot's anchored runs took up to 177 of 180 minutes with three side by side; a
# timed-out A1 run would be killed and resumed, recreating the pilot's "only A1 runs were resumed" confound.
LIMIT_MINUTES = 240
FINAL_STEP = "step_000501760.zip"


def same_zip_contents(first: Path, second: Path) -> bool:
    """True when two checkpoints hold the same members with the same bytes (zip timestamps may differ)."""
    import zipfile
    with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
        return (sorted(a.namelist()) == sorted(b.namelist())
                and all(a.read(name) == b.read(name) for name in a.namelist()))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    named = [run["id"] for run in data["runs"] if any(name in " ".join(run["command"]) for name in UNREAD_SETS)]
    if named:
        raise SystemExit(f"refusing: these runs name a fresh held-out set that must stay unread: {named}")
    if path.exists():
        raise SystemExit(f"{path} already exists")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}: {len(data['runs'])} runs")


def value_after(command: list[str], option: str) -> int:
    """The index of the value that follows `option` in a command."""
    if command.count(option) != 1:
        raise SystemExit(f"expected exactly one {option} in {command}")
    return command.index(option) + 1


def tiebreak_command(pilot_command: list[str], k: int, run_dir: str, pilot_run_dir: str) -> list[str]:
    """The pilot's command with only --seed and --run-dir changed (each checked against the pilot's values)."""
    command = list(pilot_command)
    seed_at, dir_at = value_after(command, "--seed"), value_after(command, "--run-dir")
    if command[seed_at] != str(PILOT_SEED + k) or command[dir_at] != pilot_run_dir:
        raise SystemExit(f"the pilot command for k{k} does not carry seed {PILOT_SEED + k} and {pilot_run_dir}")
    command[seed_at], command[dir_at] = str(NEW_SEED + k), run_dir
    return command


def run_dir(arm: str, k: int) -> str:
    return f"runs/train/ppo-tiebreak-{arm}-k{k}"


def train_plan() -> None:
    pilot = load(PILOT_TRAIN)
    entries = {entry["id"]: entry for entry in pilot["runs"]}
    runs = []
    for arm, k in ORDER:
        source = entries[f"ppo-anchor-pilot-{arm}-k{k}"]
        if sha(REPO / source["init_from"]) != source["clone_sha256"]:
            raise SystemExit(f"{arm}-k{k}: the clone does not match its pinned sha256")
        if (REPO / run_dir(arm, k) / "manifest.json").exists():
            raise SystemExit(f"{run_dir(arm, k)} already holds a run")
        entry = copy.deepcopy(source)
        entry.pop("resume", None)  # the pilot's three resume flags belong to the pilot's plan only
        entry.update({
            "id": f"ppo-tiebreak-{arm}-k{k}", "stage": "ppo-anchor-tiebreak-train", "seed": NEW_SEED + k,
            "limit_minutes": LIMIT_MINUTES,
            "run_dir": run_dir(arm, k), "checkpoint": f"{run_dir(arm, k)}/checkpoints/latest.zip",
            "condition": source["condition"].replace(f"seed {PILOT_SEED + k}", f"PPO seed {NEW_SEED + k} (tie-break)"),
            "command": tiebreak_command(source["command"], k, run_dir(arm, k), source["run_dir"]),
        })
        runs.append(entry)
    write(TRAIN, {
        "name": "ppo-anchor-tiebreak-train-v1", "written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "declaration": "config/ppo-anchor-tiebreak.json", "declaration_sha256": text_sha256(DECLARATION),
        "pilot_training_plan": "config/campaign-ppo-anchor-train.json", "pilot_training_plan_sha256": text_sha256(PILOT_TRAIN),
        "label": "training for the descriptive PPO anchor tie-break (E0 against A1, new PPO seeds); nothing reads a held-out set",
        "runs": runs})


def with_checkpoint(template: list[str], checkpoint: str) -> list[str]:
    command = list(template)
    command[value_after(command, "--checkpoint")] = checkpoint
    return command


def eval_entry(entry_id: str, checkpoint: str, template: list[str], stage: str) -> dict:
    return {"id": entry_id, "stage": stage, "checkpoint": checkpoint, "checkpoint_sha256": sha(REPO / checkpoint),
            "run_dir": f"runs/evaluation/_planned-{entry_id}", "resumable": False, "limit_minutes": 30,
            "command": with_checkpoint(template, checkpoint)}


def eval_plan(summary_path: Path) -> None:
    summary = load(REPO / summary_path)
    if not matches_text_hash(TRAIN, summary["plan_sha256"]) or any(r["status"] != "ok" for r in summary["results"]):
        raise SystemExit("the training summary does not match the committed training plan, or a run is not ok")
    pilot = {entry["id"]: entry for entry in load(PILOT_EVAL)["runs"]}
    templates = {"room1": pilot["room1-A1-k0-step_000100352"]["command"], "room2": pilot["room2-A1-k0-final"]["command"],
                 "room2v1": pilot["room2v1-A1-k0-final"]["command"]}
    runs = []
    for arm, k in ORDER:
        folder = run_dir(arm, k)
        checkpoints = REPO / folder / "checkpoints"
        # The Room 2 finals use latest.zip and the Room 1 final uses the last step checkpoint: they must be one policy.
        if not same_zip_contents(checkpoints / "latest.zip", checkpoints / FINAL_STEP):
            raise SystemExit(f"{folder}: latest.zip and {FINAL_STEP} do not hold the same policy")
        for step in sorted(checkpoints.glob("step_*.zip")):
            runs.append(eval_entry(f"tb-room1-{arm}-k{k}-{step.stem}", f"{folder}/checkpoints/{step.name}",
                                   templates["room1"], "ppo-anchor-tiebreak-room1"))
        runs.append(eval_entry(f"tb-room2-{arm}-k{k}-final", f"{folder}/checkpoints/latest.zip", templates["room2"],
                               "ppo-anchor-tiebreak-room2"))
        runs.append(eval_entry(f"tb-room2v1-{arm}-k{k}-final", f"{folder}/checkpoints/latest.zip", templates["room2v1"],
                               "ppo-anchor-tiebreak-room2-heldout"))
    write(EVAL, {
        "name": "ppo-anchor-tiebreak-eval-v1", "written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "declaration": "config/ppo-anchor-tiebreak.json", "declaration_sha256": text_sha256(DECLARATION),
        "training_plan": "config/campaign-ppo-tiebreak-train.json", "training_plan_sha256": text_sha256(TRAIN),
        "training_campaign": summary_path.as_posix(), "training_campaign_sha256": sha(REPO / summary_path),
        "training_campaign_start_commit": summary["commit"],
        "label": ("descriptive PPO anchor tie-break evaluations; Room 1 and Room 2 v1 held-out aggregates only; the Room 1 "
                  "v2 set and the Room 2 v2 set are never read"), "runs": runs})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train" and len(sys.argv) == 2:
        train_plan()
    elif mode == "eval" and len(sys.argv) == 3:
        eval_plan(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: make_tiebreak_plans.py train | eval <training summary.json>")
