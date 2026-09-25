"""One machine-readable summary of the Room 2 training-stability work (descriptive, training side only).

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/summarize_room2_stability.py --output <new path>

Covers every canonical-start Room 2 fine-tuning run that the stability write-up discusses: the original arm A
(seeds 20 to 25, no stall option), Objective 6's more-routes attempt (A and B, seeds 30 to 37, no option), the
stall pilot (option on) and H2 (option on). For each run it reports the final checkpoint's play without the rule
(50 stochastic canonical episodes at seed 20260924, found by checkpoint hash in runs/evaluation; every repeat play
of the same checkpoint must agree episode for episode), training clear, stall, timeout and death shares per
100,000 accepted steps, the first 50,000-step window whose clear rate reaches 0.5, and mean policy entropy early
and late. For each H2 clone it reports the clone's own play and which routes it held back (rebuilt from
`split_by_trajectory` with the recorded seed and holdout, named by matching recorded actions). It also carries the
stall pilot's screen and the H2 gate as their pinned scripts recorded them, and hashes every input it read.
Nothing here reads a held-out set; the output holds no absolute paths. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

EVALUATIONS = "runs/evaluation/chapter-1-room-2"
PLAY_SEED, PLAY_EPISODES = 20260924, 50
EXPERIMENTS = (  # (label, plan, run filter, stall option)
    ("original-A", "config/campaign-room2-train-ab.json", lambda run: "-A-" in run["id"], False),
    ("objective6", "config/campaign-room2-demos-ab.json", lambda run: True, False),
    ("stall-pilot", "config/campaign-room2-stall-pilot.json", lambda run: True, True),
    ("h2", "config/campaign-room2-demos-stall-ab.json", lambda run: True, True),
)
PILOT_ANALYSIS = "runs/campaign/20260924-163719-room2-stall-pilot-v1/stall-pilot-analysis.json"
H2_GATE = "runs/campaign/20260925-131433-room2-demos-stall-ab-play-v1/stability-gate.json"
WINDOW, ONSET_WINDOW, TOTAL = 100_000, 50_000, 500_000


class SummaryError(ValueError):
    """The records do not support the summary."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_index(steps: int, size: int, total: int = TOTAL) -> int:
    """Window of an episode ending at `steps`: (size*w, size*(w+1)]; the tail past `total` joins the last one."""
    return min((steps - 1) // size, total // size - 1)


def training_windows(episodes: list[dict], size: int = WINDOW) -> list[dict]:
    windows = [{"episodes": 0, "success": 0, "stalled": 0, "timeout": 0, "death": 0} for _ in range(TOTAL // size)]
    for episode in episodes:
        cell = windows[window_index(episode["accepted_steps"], size)]
        cell["episodes"] += 1
        if episode["ending"] in cell:
            cell[episode["ending"]] += 1
    return windows


def first_window_reaching(episodes: list[dict], rate: float = 0.5, size: int = ONSET_WINDOW) -> int | None:
    """End step of the first `size` window whose training clear rate reaches `rate`, or None."""
    for index, cell in enumerate(training_windows(episodes, size)):
        if cell["episodes"] and cell["success"] / cell["episodes"] >= rate:
            return (index + 1) * size
    return None


def fisher_one_sided(a_events: int, a_n: int, b_events: int, b_n: int) -> float:
    """P(at least a_events of all events fall in group a), margins fixed (hypergeometric)."""
    events, total = a_events + b_events, a_n + b_n
    return sum(comb(a_n, k) * comb(b_n, events - k) for k in range(a_events, min(events, a_n) + 1)) / comb(total, events)


def mean_entropy(progress: list[dict], first: bool) -> float:
    rows = [(int(row["accepted_steps"]), -float(row["entropy_loss"])) for row in progress if row["entropy_loss"]]
    last_step = rows[-1][0]
    chosen = [value for step, value in rows if (step <= WINDOW if first else step > last_step - WINDOW)]
    return round(sum(chosen) / len(chosen), 4)


class Plays:
    """Canonical stochastic plays at the declared seed, indexed by checkpoint sha256."""

    def __init__(self):
        self.by_sha: dict[str, list[Path]] = {}
        for result_path in sorted((REPO / EVALUATIONS).glob("*/results.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("evaluation_seed") == PLAY_SEED and result.get("episodes") == PLAY_EPISODES \
                    and result.get("attributable") is True:
                self.by_sha.setdefault(result["checkpoint_sha256"], []).append(result_path.parent)

    def summary(self, checkpoint: Path, inputs: dict) -> dict | None:
        """None when the checkpoint was never played this way (Objective 6 B34 to B37)."""
        folders = self.by_sha.get(file_sha256(checkpoint), [])
        if not folders:
            return None
        outcomes = []
        for folder in folders:
            episodes_path = folder / "episodes.jsonl"
            inputs[episodes_path.relative_to(REPO).as_posix()] = file_sha256(episodes_path)
            rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            outcomes.append([(row["ending"], row["length"], row["max_x"]) for row in rows
                             if row["start_id"] == "canonical" and row["mode"] == "stochastic"])
        if any(outcome != outcomes[0] for outcome in outcomes) or len(outcomes[0]) != PLAY_EPISODES:
            raise SummaryError(f"plays of {checkpoint} disagree or are incomplete")
        endings = [ending for ending, _, _ in outcomes[0]]
        failures = sorted(max_x for ending, _, max_x in outcomes[0] if ending != "success")
        return {"clears": endings.count("success"), "timeouts": endings.count("timeout"),
                "deaths": endings.count("death"), "episodes": len(endings), "plays_found": len(folders),
                "median_failure_max_x": float(np.median(failures)) if failures else None,
                "first_play": folders[0].relative_to(REPO).as_posix()}


def run_record(run: dict, experiment: str, option: bool, plays: Plays, inputs: dict) -> dict:
    run_dir = REPO / run["run_dir"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "finished":
        raise SummaryError(f"{run['id']} is {manifest['status']}")
    for name in ("episodes.jsonl", "progress.csv"):
        inputs[f"{run['run_dir']}/{name}"] = file_sha256(run_dir / name)
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(episode["start"] != "canonical" for episode in episodes):
        raise SummaryError(f"{run['id']} has non-canonical training starts")
    progress = list(csv.DictReader((run_dir / "progress.csv").read_text(encoding="utf-8").splitlines()))
    play = plays.summary(run_dir / "checkpoints" / "latest.zip", inputs)
    command = run["command"]
    if ("--stall-frames" in command) != option:
        raise SummaryError(f"{run['id']}: stall option in the command does not match {experiment}")
    clone = Path(command[command.index("--init-from") + 1]).parent.as_posix()
    return {"id": run["id"], "experiment": experiment, "arm": run["arm"][0],  # the pilot's arms are "A-stall", "B-stall"
            "stall_option": option, "clone": clone,
            "accepted_steps": manifest["accepted_steps"], "final_play": play,
            "collapsed": None if play is None else play["clears"] < PLAY_EPISODES / 2,
            "training_windows": training_windows(episodes),
            "first_50k_window_reaching_half": first_window_reaching(episodes),
            "entropy": {"first_100k": mean_entropy(progress, True), "last_100k": mean_entropy(progress, False)}}


def clone_records(plan: dict, plays: Plays, inputs: dict) -> dict:
    from celeste_rl.demonstrations import materialize_demonstrations, validate_demonstration_manifest
    from celeste_rl.tasks import resolve_task_definition, task_identity
    from evaluate_checkpoint import match_trajectories

    identity = task_identity(resolve_task_definition(REPO / "config/room2.json"))
    records = {}
    for arm, clones in sorted(plan["clones"].items()):
        for k, clone in sorted(clones.items(), key=lambda item: int(item[0])):
            clone_dir = REPO / clone["dir"]
            if file_sha256(clone_dir / "cloned.zip") != clone["sha256"]:
                raise SummaryError(f"{clone['dir']} does not match the plan's hash")
            record = json.loads((clone_dir / "results.json").read_text(encoding="utf-8"))
            args = record["args"]
            dataset = args.get("dataset")
            dataset_path = REPO / dataset if dataset not in (None, "None") else clone_dir / "dataset.npz"
            inputs[dataset_path.relative_to(REPO).as_posix()] = file_sha256(dataset_path)
            manifest = json.loads((REPO / record["manifests"]["demonstrations"]).read_text(encoding="utf-8"))
            demonstrations = materialize_demonstrations(validate_demonstration_manifest(manifest, identity), REPO,
                                                        identity)
            data = np.load(dataset_path)
            mapping = match_trajectories(demonstrations, data["trajectory"], data["actions"])
            trajectories = np.unique(data["trajectory"])
            order = np.random.default_rng(int(args["seed"])).permutation(trajectories)
            kept = max(1, round(len(trajectories) * float(args["holdout"])))
            held = sorted(mapping[int(t)]["name"] for t in order[:kept])
            if record["before"]["holdout"]["trajectories"] != len(held):
                raise SummaryError(f"{clone['dir']}: rebuilt split does not match the recorded count")
            records[f"{arm}-k{k}"] = {"clone": clone["dir"], "sha256": clone["sha256"], "seed": int(args["seed"]),
                                      "routes": len(trajectories), "fitted": len(trajectories) - kept,
                                      "held_back": held, "play": plays.summary(clone_dir / "cloned.zip", inputs)}
            if records[f"{arm}-k{k}"]["play"] is None:
                raise SummaryError(f"{clone['dir']} has no canonical play at seed {PLAY_SEED}")
    return records


def build() -> dict:
    inputs: dict[str, str] = {}
    plays = Plays()
    runs = []
    for experiment, plan_path, keep, option in EXPERIMENTS:
        inputs[plan_path] = file_sha256(REPO / plan_path)
        plan = json.loads((REPO / plan_path).read_text(encoding="utf-8"))
        runs += [run_record(run, experiment, option, plays, inputs) for run in plan["runs"] if keep(run)]
    h2_plan = json.loads((REPO / EXPERIMENTS[-1][1]).read_text(encoding="utf-8"))
    for path in (PILOT_ANALYSIS, H2_GATE):
        inputs[path] = file_sha256(REPO / path)
    analysis = json.loads((REPO / PILOT_ANALYSIS).read_text(encoding="utf-8"))
    gate = json.loads((REPO / H2_GATE).read_text(encoding="utf-8"))
    by_group = {}
    for run in runs:
        by_group.setdefault(f"{run['experiment']}/{run['arm']}", []).append(run)
    collapses = {group: {"collapsed": sum(r["collapsed"] is True for r in members),
                         "played": sum(r["collapsed"] is not None for r in members), "runs": len(members)}
                 for group, members in sorted(by_group.items())}
    with_option = {"7-route": [r for r in runs if r["stall_option"] and r["arm"] == "A"],
                   "21-route": [r for r in runs if r["stall_option"] and r["arm"] == "B"]}
    seven, twentyone = (sum(r["collapsed"] for r in with_option[k]) for k in ("7-route", "21-route"))
    return {
        "label": "Room 2 training stability, descriptive and training side only; nothing reads a held-out set",
        "collapse_rule": f"final checkpoint clears fewer than {PLAY_EPISODES // 2} of {PLAY_EPISODES} stochastic "
                         f"canonical episodes played without the stall rule at seed {PLAY_SEED}",
        "collapses": collapses,
        "descriptive_tests": {
            "h2_A_vs_B_one_sided_fisher": fisher_one_sided(collapses["h2/A"]["collapsed"], collapses["h2/A"]["runs"],
                                                           collapses["h2/B"]["collapsed"], collapses["h2/B"]["runs"]),
            "with_option_7route_vs_21route": {"7-route": [seven, len(with_option["7-route"])],
                                              "21-route": [twentyone, len(with_option["21-route"])],
                                              "one_sided_fisher": fisher_one_sided(
                                                  seven, len(with_option["7-route"]), twentyone,
                                                  len(with_option["21-route"]))}},
        "stall_pilot_screen": {"plan_sha256": analysis["plan_sha256"],
                               "analysis_code_git_blob": analysis["analysis_code_git_blob"],
                               "screen": analysis["screen"],
                               "groups": {name: {key: value for key, value in group.items() if key != "per_seed"}
                                          for name, group in analysis["groups"].items()}},
        "h2_gate": {"plan": Path(gate["plan"]).as_posix(), "plan_sha256": gate["plan_sha256"],
                    "script_git_blob": h2_plan["stability_gate"]["script_git_blob"], "rates": gate["rates"],
                    "arms": gate["arms"], "passes": gate["passes"]},
        "h2_clones": clone_records(h2_plan, plays, inputs),
        "runs": runs,
        "inputs_sha256": dict(sorted(inputs.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise SummaryError(f"{args.output} already exists")
        summary = build()
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    except (OSError, KeyError, json.JSONDecodeError, SummaryError) as error:
        print(f"Summary refused: {error}")
        return 2
    for group, count in summary["collapses"].items():
        print(f"{group}: {count['collapsed']}/{count['played']} played collapsed ({count['runs']} runs)")
    print(f"H2 gate passes: {summary['h2_gate']['passes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
