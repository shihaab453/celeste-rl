"""Verify and analyse the predeclared Room 2 deterministic-versus-stochastic noise check.

Run from the repo root with the RL interpreter, after the campaign has finished:
    .venv-rl/Scripts/python.exe scripts/analyze_noise_check.py --plan config/campaign-room2-noise-check.json \
        --campaign runs/campaign/<timestamp>-room2-noise-check-v1/summary.json --output <path>/analysis.json

It implements only the analysis declared in the plan and fails closed before computing anything if a declared
run, file, hash, start or row is missing, duplicated, stopped or inconsistent. `--output` is required, so reading
a campaign never overwrites an earlier analysis by accident. The script also refuses unless its own git blob id
equals the plan's pinned analysis_code.git_blob, and records that blob in its output.

Everything here is diagnostic. It is not a confirmatory A/B analysis and does not change the published Room 2
result.

Definitions, all fixed by the plan:
- A start's deterministic success is the mean over its deterministic repeats. Repeats that disagree are a
  validity problem: they are counted and reported, and the mean is still used, never only the first repeat.
- A start's stochastic success is its sampled successes divided by its sampled episodes.
- Per checkpoint, the primary difference is the equal-weight mean over route clusters of the mean over that
  route's primary starts of (deterministic minus stochastic). Primary starts are demonstration starts whose
  recorded start position x is below the plan's threshold. The all-starts secondary uses every demonstration
  start the same way. The canonical start is reported separately.
- The primary start count per route must equal the plan's primary_route_counts, not only the total.
- Failures are episodes whose ending is not success. They are banded by `end_x` (in these rows the last frame
  with a player, from the training progress helper) and separately by the true running maximum `max_x`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CANONICAL_ID = "canonical"
MODES = ("stochastic", "deterministic")
BANDS = (("before 360", None, 360), ("360 to 459", 360, 460), ("after 459", 460, None))


class AnalysisError(ValueError):
    """The campaign does not satisfy the plan's analysis contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start_list_sha256(starts: list[dict]) -> str:
    """The same hash evaluate_checkpoint.start_list_record writes, over the listed fields without recipes."""
    listed = [{key: value for key, value in start.items() if key != "lines"} for start in starts]
    return hashlib.sha256(json.dumps(listed, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def band_of(x) -> str:
    if x is None:
        return "no position"
    for name, low, high in BANDS:
        if (low is None or x >= low) and (high is None or x < high):
            return name
    raise AssertionError(x)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rows(entry: dict, protocol: dict, starts: list[dict], rows: list[dict], task: dict) -> None:
    """Exactly the declared rows: every start and mode, repeats 0..n-1, no problems, the right checkpoint."""
    expected = {(CANONICAL_ID, "stochastic"): protocol["canonical_stochastic_episodes"],
                (CANONICAL_ID, "deterministic"): protocol["deterministic_repeats_per_start"]}
    for start in starts:
        expected[(start["start_id"], "stochastic")] = protocol["stochastic_episodes_per_demonstration_start"]
        expected[(start["start_id"], "deterministic")] = protocol["deterministic_repeats_per_start"]
    seen: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        if row.get("problem") is not None:
            raise AnalysisError(f"{entry['id']} has a problem row at {row.get('start_id')}: {row['problem']}")
        if row.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
            raise AnalysisError(f"{entry['id']} row checkpoint hash mismatch")
        if row.get("task") != task:
            raise AnalysisError(f"{entry['id']} row task identity mismatch")
        key = (row.get("start_id"), row.get("mode"))
        if key not in expected:
            raise AnalysisError(f"{entry['id']} has an undeclared row {key}")
        if row.get("ending") is None:
            raise AnalysisError(f"{entry['id']} row {key} has no ending")
        seen.setdefault(key, []).append(row.get("repeat"))
    for key, count in expected.items():
        if sorted(seen.get(key, [])) != list(range(count)):
            raise AnalysisError(f"{entry['id']} {key} has repeats {sorted(seen.get(key, []))}, expected 0..{count - 1}")


def checkpoint_measures(starts: list[dict], rows: list[dict], x_threshold: float) -> dict:
    """The plan's per-checkpoint quantities from already-validated rows."""
    success: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        success.setdefault((row["start_id"], row["mode"]), []).append(int(row["ending"] == "success"))

    def per_start(start_id: str) -> dict:
        deterministic = success[(start_id, "deterministic")]
        stochastic = success[(start_id, "stochastic")]
        det = sum(deterministic) / len(deterministic)
        sto = sum(stochastic) / len(stochastic)
        return {"deterministic": det, "stochastic": sto, "difference": det - sto,
                "deterministic_repeats_disagree": len(set(deterministic)) > 1}

    by_start = {start["start_id"]: {**per_start(start["start_id"]), "route": start["route"],
                                    "start_x": start["position"][0]} for start in starts}

    def route_macro(selected: list[str]) -> dict:
        routes: dict[str, list[float]] = {}
        for start_id in selected:
            routes.setdefault(by_start[start_id]["route"], []).append(by_start[start_id]["difference"])
        route_values = {route: sum(values) / len(values) for route, values in sorted(routes.items())}
        return {"difference": sum(route_values.values()) / len(route_values) if route_values else None,
                "routes": route_values, "starts_per_route": {r: len(v) for r, v in sorted(routes.items())},
                "starts": len(selected)}

    primary_ids = [s["start_id"] for s in starts if s["position"][0] < x_threshold]
    canonical = per_start(CANONICAL_ID)
    failures = {mode: {"by_end_x": Counter(), "by_max_x": Counter()} for mode in MODES}
    endings = {mode: Counter() for mode in MODES}
    for row in rows:
        endings[row["mode"]][row["ending"]] += 1
        if row["ending"] != "success":
            failures[row["mode"]]["by_end_x"][band_of(row.get("end_x"))] += 1
            failures[row["mode"]]["by_max_x"][band_of(row.get("max_x"))] += 1
    disagreements = [start_id for start_id, value in by_start.items() if value["deterministic_repeats_disagree"]]
    if canonical["deterministic_repeats_disagree"]:
        disagreements.insert(0, CANONICAL_ID)
    return {
        "primary": route_macro(primary_ids),
        "all_starts": route_macro([s["start_id"] for s in starts]),
        "canonical": canonical,
        "deterministic_disagreements": {"count": len(disagreements), "starts": disagreements},
        "failures": {mode: {kind: dict(sorted(counts.items())) for kind, counts in value.items()}
                     for mode, value in failures.items()},
        "endings": {mode: dict(sorted(counts.items())) for mode, counts in endings.items()},
        "by_start": by_start,
    }


def summarise(plan: dict, measures: dict[str, dict]) -> dict:
    """The checkpoint-level summary and the predeclared decision, over the declared checkpoints only."""
    thresholds = plan["decision_rule"]["thresholds"]
    declared = [entry for entry in plan["runs"] if entry.get("arm") is not None]
    values = {entry["id"]: measures[entry["id"]]["primary"]["difference"] * 100 for entry in declared}
    # A small tolerance so a difference of exactly +10 or +3 points is not lost to floating point.
    at_least = sum(value >= thresholds["favour_points"] - 1e-9 for value in values.values())
    below = sum(value < thresholds["null_points"] - 1e-9 for value in values.values())
    if at_least >= thresholds["favour_min_checkpoints"]:
        decision = "favour"
    elif below >= thresholds["null_min_checkpoints"]:
        decision = "null"
    else:
        decision = "inconclusive"
    by_arm = {}
    for arm in sorted({entry["arm"] for entry in declared}):
        arm_values = [values[entry["id"]] for entry in declared if entry["arm"] == arm]
        by_arm[arm] = {"median_points": statistics.median(arm_values), "min_points": min(arm_values),
                       "max_points": max(arm_values)}
    ordered = sorted(values.values())
    return {
        "primary_points_by_checkpoint": values,
        "median_points": statistics.median(ordered),
        "range_points": [ordered[0], ordered[-1]],
        "checkpoints_at_least_favour_points": at_least,
        "checkpoints_below_null_points": below,
        "by_arm_descriptive": by_arm,
        "decision": decision,
        "decision_meaning": plan["decision_rule"]["outcomes"][decision],
        "deterministic_disagreements_total": sum(measures[e["id"]]["deterministic_disagreements"]["count"]
                                                 for e in plan["runs"]),
    }


def _git_blob(commit: str, path: str) -> str:
    result = subprocess.run(["git", "rev-parse", f"{commit}:{path}"], cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise AnalysisError(f"cannot read {path} at {commit}: {result.stderr.strip()}")
    return result.stdout.strip()


def script_git_blob() -> str:
    """This file's git blob id. git hash-object applies the repo's line-ending rules, so a CRLF checkout on
    Windows and an LF checkout elsewhere give the same id; a raw sha256 of the file would not."""
    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    result = subprocess.run(["git", "hash-object", "--", relative], cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise AnalysisError(f"cannot compute the git blob of {relative}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_campaign(plan_path: Path, summary_path: Path, *, script_blob: str | None = None,
                  git_blob: Callable[[str, str], str] = _git_blob) -> tuple[dict, dict]:
    """Validate the campaign against the plan and return (plan, per-run data).

    `script_blob` is this script's git blob id (computed when not given); `git_blob(commit, path)` looks up a
    file's blob at a commit. Both are parameters only so tests can supply them.
    """
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pinned_blob = plan["analysis"]["analysis_code"]["git_blob"]
    own_blob = script_git_blob() if script_blob is None else script_blob
    if own_blob != pinned_blob:
        raise AnalysisError(f"this analysis script is blob {own_blob}, but the plan pins {pinned_blob}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("plan") != plan["name"] or summary.get("plan_sha256") != file_sha256(plan_path):
        raise AnalysisError("the campaign summary was not produced from this plan file")
    tool = plan["tool"]
    if git_blob(summary["commit"], tool["script"]) != git_blob(tool["commit"], tool["script"]):
        raise AnalysisError("the campaign ran a different evaluate_checkpoint.py from the pinned tool commit")
    protocol = plan["evaluation_protocol"]
    expected_starts_sha = plan["starts"]["demonstration"]["expected_start_list_sha256"]
    results_by_id = {}
    for record in summary.get("results", []):
        if record.get("id") in results_by_id:
            raise AnalysisError(f"the campaign summary lists {record['id']} twice")
        results_by_id[record.get("id")] = record
    data = {}
    for entry in plan["runs"]:
        record = results_by_id.get(entry["id"])
        if record is None or record.get("status") != "ok":
            raise AnalysisError(f"{entry['id']} did not finish ok: {record and record.get('status')}")
        artifact = record.get("artifact") or {}
        if artifact.get("problem") or not artifact.get("result_file") or not artifact.get("episodes_file"):
            raise AnalysisError(f"{entry['id']} has no complete result artifact")
        result_path, episodes_path = REPO / artifact["result_file"], REPO / artifact["episodes_file"]
        if file_sha256(result_path) != artifact.get("result_sha256") \
                or file_sha256(episodes_path) != artifact.get("episodes_sha256"):
            raise AnalysisError(f"{entry['id']} result or episode file changed after the campaign")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checks = {
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "evaluation_seed": protocol["evaluation_seed"],
            "deterministic_repeats": protocol["deterministic_repeats_per_start"],
            "aborted": None,
            "task": plan["task"],
            "attributable": True,
        }
        for field, value in checks.items():
            if result.get(field) != value:
                raise AnalysisError(f"{entry['id']} result {field} is {result.get(field)!r}; expected {value!r}")
        provenance = result.get("demonstration_starts") or {}
        starts_file = result_path.parent / (provenance.get("starts_file") or "starts.json")
        starts_record = json.loads(starts_file.read_text(encoding="utf-8"))
        starts = starts_record["starts"]
        if (provenance.get("starts_sha256") != expected_starts_sha or starts_record.get("sha256") != expected_starts_sha
                or start_list_sha256(starts) != expected_starts_sha):
            raise AnalysisError(f"{entry['id']} did not use the declared start list")
        if provenance.get("stochastic_per_start") != protocol["stochastic_episodes_per_demonstration_start"]:
            raise AnalysisError(f"{entry['id']} used a different number of sampled episodes per start")
        rows = _read_jsonl(episodes_path)
        validate_rows(entry, protocol, starts, rows, plan["task"])
        data[entry["id"]] = {"starts": starts, "rows": rows, "result_file": artifact["result_file"],
                             "result_sha256": artifact["result_sha256"],
                             "episodes_sha256": artifact["episodes_sha256"]}
    return plan, data


def analyse(plan: dict, data: dict) -> dict:
    threshold = plan["analysis"]["primary_start_x_below"]
    measures = {run_id: checkpoint_measures(item["starts"], item["rows"], threshold) for run_id, item in data.items()}
    expected_primary = plan["analysis"]["primary_starts"]
    expected_routes = {route: count for route, count in
                       plan["starts"]["demonstration"]["primary_route_counts"].items() if route != "total"}
    for run_id, value in measures.items():
        if value["primary"]["starts"] != expected_primary:
            raise AnalysisError(f"{run_id} has {value['primary']['starts']} primary starts, expected {expected_primary}")
        if value["primary"]["starts_per_route"] != expected_routes:
            raise AnalysisError(f"{run_id} has primary starts per route {value['primary']['starts_per_route']}, "
                                f"expected {expected_routes}")
    reference = [entry["id"] for entry in plan["runs"] if entry.get("arm") is None]
    return {
        "label": plan["analysis"]["label"],
        "scope": plan["decision_rule"]["scope"],
        "summary": summarise(plan, measures),
        "reference_runs": {run_id: {"note": next(e["role"] for e in plan["runs"] if e["id"] == run_id),
                                    "primary_points": measures[run_id]["primary"]["difference"] * 100}
                           for run_id in reference},
        "checkpoints": measures,
        "artifacts": {run_id: {k: v for k, v in item.items() if k not in ("starts", "rows")}
                      for run_id, item in data.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True, help="the campaign's summary.json")
    parser.add_argument("--output", type=Path, required=True, help="where to write the analysis JSON")
    args = parser.parse_args()
    if args.output.exists():
        print(f"{args.output} already exists; choose a new path rather than overwrite an analysis")
        return 2
    try:
        own_blob = script_git_blob()
        plan, data = load_campaign(args.plan, args.campaign, script_blob=own_blob)
        report = analyse(plan, data)
    except (OSError, KeyError, json.JSONDecodeError, AnalysisError) as error:
        print(f"Analysis refused: {error}")
        return 2
    report = {"plan": plan["name"], "plan_sha256": file_sha256(args.plan), "campaign": str(args.campaign),
              "analysis_code_git_blob": own_blob, **report}
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = report["summary"]
    print("Diagnostic only; not a confirmatory claim.")
    for run_id, points in summary["primary_points_by_checkpoint"].items():
        print(f"  {run_id:<26} primary deterministic minus stochastic {points:+.1f} points")
    print(f"median {summary['median_points']:+.1f}, at least +{plan['decision_rule']['thresholds']['favour_points']}: "
          f"{summary['checkpoints_at_least_favour_points']}, below +{plan['decision_rule']['thresholds']['null_points']}: "
          f"{summary['checkpoints_below_null_points']}, deterministic disagreements "
          f"{summary['deterministic_disagreements_total']}")
    print(f"decision: {summary['decision']}: {summary['decision_meaning']}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
