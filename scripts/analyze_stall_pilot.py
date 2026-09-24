"""Verify and analyse the predeclared Room 2 stability pilot (config/campaign-room2-stall-pilot.json).

Run from the repo root with the RL interpreter, after both campaigns (training, then final-checkpoint play):
    .venv-rl/Scripts/python.exe scripts/analyze_stall_pilot.py --plan config/campaign-room2-stall-pilot.json \
        --training runs/campaign/<ts>-room2-stall-pilot-v1/summary.json \
        --play runs/campaign/<ts>-room2-stall-pilot-play-v1/summary.json --output <new path>

Everything here is training side and descriptive: nothing reads a held-out set. It implements only the analysis
the plan declares and refuses before computing anything if a declared run, hash, commit or file is missing or
different. `--output` is required and never overwritten.

- Primary: each final checkpoint played without the stall rule (50 stochastic canonical episodes, the plan's
  sampling seed). A run has collapsed when that clear rate is below the plan's threshold. Treatment and control
  are compared as two groups (after a run's first stalled episode the pair no longer shares a trajectory).
- Screen: the gate on the treatment's collapse count, plus the harm check on its healthy runs.
- Pairing check: in each treatment run, every episode before its first stalled ending must equal the control
  run's episode of the same index, field for field. The game and training are deterministic, so a mismatch
  means the control would not have been reproduced and the pair is flagged.
- Secondary and waste metrics per 100,000-transition window, from each run's episodes.jsonl and progress.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIRED_FIELDS = ("accepted_steps", "ending", "length", "return", "max_x", "end_x", "end_y", "max_potential")
WINDOW = 100_000


class AnalysisError(ValueError):
    """The pilot does not satisfy its predeclared analysis contract."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collapsed(rate: float, threshold: float) -> bool:
    return rate < threshold


def fisher_one_sided(treatment_collapses: int, control_collapses: int, n_treatment: int, n_control: int) -> float:
    """P(treatment collapses <= observed) under a hypergeometric null with the margins fixed."""
    total = treatment_collapses + control_collapses
    population = n_treatment + n_control
    denominator = math.comb(population, total)
    return sum(math.comb(n_treatment, k) * math.comb(n_control, total - k)
               for k in range(0, treatment_collapses + 1) if total - k <= n_control) / denominator


def first_stalled(episodes: list[dict]) -> int | None:
    return next((index for index, episode in enumerate(episodes) if episode["ending"] == "stalled"), None)


def pairing_check(treatment: list[dict], control: list[dict]) -> dict:
    """Episodes before the treatment's first stalled ending must equal the control's, field for field."""
    cut = first_stalled(treatment)
    shared = len(treatment) if cut is None else cut
    if shared > len(control):
        return {"shared_episodes": shared, "matches": False, "first_mismatch": len(control)}
    mismatch = next((index for index in range(shared)
                     if any(treatment[index].get(field) != control[index].get(field) for field in PAIRED_FIELDS)),
                    None)
    return {"shared_episodes": shared, "matches": mismatch is None, "first_mismatch": mismatch}


def window_metrics(episodes: list[dict], progress: list[dict], crossing_x: int) -> dict:
    windows: dict[int, list[dict]] = {}
    for episode in episodes:
        windows.setdefault(min(episode["accepted_steps"] // WINDOW, 4), []).append(episode)
    entropy: dict[int, list[float]] = {}
    for row in progress:
        if row.get("entropy_loss") not in ("", None):
            entropy.setdefault(min(int(row["accepted_steps"]) // WINDOW, 4), []).append(-float(row["entropy_loss"]))
    out = {}
    for key, group in sorted(windows.items()):
        frames = sum(episode["length"] for episode in group)
        wasted = sum(episode["length"] for episode in group if episode["ending"] in ("timeout", "stalled"))
        out[f"{key * 100}k-{(key + 1) * 100}k"] = {
            "episodes": len(group),
            # A headline waste measure with attempts reaching the crossing: stuck runs also die after long
            # wandering, so the timeout share alone understates the waste.
            "mean_frames_per_episode": frames / len(group),
            "clear_rate": sum(episode["ending"] == "success" for episode in group) / len(group),
            "timeout_share": sum(episode["ending"] == "timeout" for episode in group) / len(group),
            "stalled_share": sum(episode["ending"] == "stalled" for episode in group) / len(group),
            "share_of_frames_in_timeout_or_stalled_episodes": wasted / frames if frames else None,
            "attempts_reaching_crossing": sum((episode.get("max_x") or 0) >= crossing_x for episode in group),
            "mean_entropy": statistics.mean(entropy[key]) if entropy.get(key) else None,
        }
    return out


def training_record(run_dir: Path, final_window_steps: int, total_steps: int) -> dict:
    episodes = jsonl(run_dir / "episodes.jsonl")
    late = [episode for episode in episodes if episode["accepted_steps"] > total_steps - final_window_steps]
    return {"episodes": episodes, "progress": list(csv.DictReader(
        (run_dir / "progress.csv").read_text(encoding="utf-8").splitlines())),
        "late_clear_rate": sum(e["ending"] == "success" for e in late) / len(late) if late else None,
        "late_episodes": len(late)}


def verify_training(plan: dict, summary: dict, plan_hash: str) -> None:
    if summary.get("plan") != plan["name"] or summary.get("plan_sha256") != plan_hash:
        raise AnalysisError("the training campaign summary does not identify the committed plan")
    results = {entry["id"]: entry for entry in summary["results"]}
    for run in plan["runs"]:
        if results.get(run["id"], {}).get("status") != "ok":
            raise AnalysisError(f"{run['id']} did not complete ok; do not replace it or analyse a subset")
        manifest = json.loads((REPO / run["run_dir"] / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        expected = {"stall_frames": plan["treatment"]["stall_frames"], "seed": run["seed"],
                    "init_from": run["init_from"], "total_timesteps": plan["treatment"]["total_timesteps"]}
        for field, value in expected.items():
            if config.get(field) != value:
                raise AnalysisError(f"{run['id']} config {field} is {config.get(field)!r}, expected {value!r}")
        if manifest["status"] != "finished":
            raise AnalysisError(f"{run['id']} did not finish")
        commits = {session["provenance"].get("commit") for session in manifest["sessions"]}
        if commits != {summary["commit"]} or any(session["provenance"].get("uncommitted_changes")
                                                 for session in manifest["sessions"]):
            raise AnalysisError(f"{run['id']} is not attributable to the clean campaign commit")


def play_rate(result_path: Path, expected_checkpoint_sha256: str, protocol: dict) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for field, value in (("checkpoint_sha256", expected_checkpoint_sha256), ("episodes", protocol["episodes"]),
                         ("evaluation_seed", protocol["seed"])):
        if result.get(field) != value:
            raise AnalysisError(f"{result_path} {field} is {result.get(field)!r}, expected {value!r}")
    if result.get("attributable") is not True or result.get("runtime_problems"):
        raise AnalysisError(f"{result_path} is not attributable")
    return {"clear_rate": result["success_rate"], "endings": result["endings"],
            # Descriptive, never a gate: how long successful episodes take without the rule.
            "success_length": result.get("success_length"),
            "deterministic": result["deterministic"]["ending"], "result": str(result_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True, help="the training campaign's summary.json")
    parser.add_argument("--play", type=Path, required=True, help="the final-checkpoint play campaign's summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise AnalysisError(f"{args.output} already exists")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        own_blob = git_blob(Path(__file__).resolve())
        if own_blob != plan["analysis"]["analysis_code_git_blob"]:
            raise AnalysisError(f"this script's git blob {own_blob} is not the plan's pinned "
                                f"{plan['analysis']['analysis_code_git_blob']}")
        verify_training(plan, json.loads(args.training.read_text(encoding="utf-8")), file_sha256(args.plan))
        play_summary = json.loads(args.play.read_text(encoding="utf-8"))
        play_results = {entry["id"]: entry for entry in play_summary["results"]}
        protocol = plan["primary"]["protocol"]
        threshold = plan["primary"]["collapse_below"]
        groups = {}
        for group_name, group in plan["groups"].items():
            rows = {}
            for pair in group["pairs"]:
                run = next(r for r in plan["runs"] if r["id"] == pair["treatment"])
                checkpoint = REPO / run["run_dir"] / "checkpoints" / "latest.zip"
                play = play_results.get(f"play-{run['id']}")
                if play is None or play.get("status") != "ok":
                    raise AnalysisError(f"no completed final-checkpoint play for {run['id']}")
                treatment_play = play_rate(REPO / play["artifact"]["result_file"], file_sha256(checkpoint), protocol)
                control = pair["control"]
                if file_sha256(REPO / control["checkpoint"]) != control["checkpoint_sha256"]:
                    raise AnalysisError(f"control checkpoint {control['checkpoint']} changed")
                if control.get("play_result"):
                    # Played before the plan was written, and pinned by hash.
                    if file_sha256(REPO / control["play_result"]) != control["play_result_sha256"]:
                        raise AnalysisError(f"control play result {control['play_result']} changed")
                    control_result = REPO / control["play_result"]
                else:
                    # Played in the play campaign, alongside the treatment.
                    entry = play_results.get(f"play-{Path(control['run_dir']).name}")
                    if entry is None or entry.get("status") != "ok":
                        raise AnalysisError(f"no completed final-checkpoint play for {control['run_dir']}")
                    control_result = REPO / entry["artifact"]["result_file"]
                control_play = play_rate(control_result, control["checkpoint_sha256"], protocol)
                total = plan["treatment"]["total_timesteps"]
                treated = training_record(REPO / run["run_dir"], plan["secondary"]["final_window_steps"], total)
                controlled = training_record(REPO / control["run_dir"], plan["secondary"]["final_window_steps"], total)
                rows[str(run["seed"])] = {
                    "treatment": {**treatment_play, "collapsed": collapsed(treatment_play["clear_rate"], threshold),
                                  "late_training_clear_rate": treated["late_clear_rate"],
                                  "training_episodes": len(treated["episodes"]),
                                  "windows": window_metrics(treated["episodes"], treated["progress"],
                                                            plan["secondary"]["crossing_x"])},
                    "control": {**control_play, "collapsed": collapsed(control_play["clear_rate"], threshold),
                                "late_training_clear_rate": controlled["late_clear_rate"],
                                "training_episodes": len(controlled["episodes"]),
                                "windows": window_metrics(controlled["episodes"], controlled["progress"],
                                                          plan["secondary"]["crossing_x"])},
                    "pairing": pairing_check(treated["episodes"], controlled["episodes"]),
                    "borderline": [side for side, play_row in (("treatment", treatment_play),
                                                               ("control", control_play))
                                   if abs(play_row["clear_rate"] - threshold) <= plan["primary"]["borderline"]],
                }
            t_collapses = sum(row["treatment"]["collapsed"] for row in rows.values())
            c_collapses = sum(row["control"]["collapsed"] for row in rows.values())
            healthy_t = [row["treatment"]["clear_rate"] for row in rows.values() if not row["treatment"]["collapsed"]]
            healthy_c = [row["control"]["clear_rate"] for row in rows.values() if not row["control"]["collapsed"]]
            harm = (bool(healthy_t) and bool(healthy_c) and
                    statistics.median(healthy_t) < statistics.median(healthy_c) - plan["screen"]["harm_points"])
            groups[group_name] = {
                "counts_toward_screen": group["counts_toward_screen"],
                "per_seed": rows,
                "treatment_collapses": t_collapses, "control_collapses": c_collapses, "runs": len(rows),
                "fisher_one_sided_p": fisher_one_sided(t_collapses, c_collapses, len(rows), len(rows)),
                "healthy_median_clear_rate": {"treatment": statistics.median(healthy_t) if healthy_t else None,
                                              "control": statistics.median(healthy_c) if healthy_c else None},
                "harm_flag": harm,
                "pairing_failures": [seed for seed, row in rows.items() if not row["pairing"]["matches"]],
            }
        screened = [g for g in groups.values() if g["counts_toward_screen"]]
        screen = {"passes": all(g["treatment_collapses"] <= plan["screen"]["max_treatment_collapses"]
                                and not g["harm_flag"] for g in screened),
                  "label": plan["screen"]["label"]}
        report = {"plan": str(args.plan), "plan_sha256": file_sha256(args.plan),
                  "analysis_code_git_blob": own_blob, "label": plan["label"], "screen": screen, "groups": groups}
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    except (OSError, KeyError, json.JSONDecodeError, AnalysisError, StopIteration) as error:
        print(f"Analysis refused: {error}")
        return 2
    for name, group in groups.items():
        print(f"{name}: treatment collapses {group['treatment_collapses']}/{group['runs']}, control "
              f"{group['control_collapses']}/{group['runs']}, one-sided Fisher p {group['fisher_one_sided_p']:.3f}, "
              f"harm flag {group['harm_flag']}, pairing failures {group['pairing_failures']}")
    print(f"screen passes: {screen['passes']} ({screen['label']})")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
