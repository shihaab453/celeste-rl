"""Descriptive analysis of the PPO anchor tie-break, as config/ppo-anchor-tiebreak.json declares (read-only).

Run from the repo root with the RL interpreter, after the tie-break evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_ppo_tiebreak.py \
        runs/campaign/<ts>-ppo-anchor-tiebreak-eval-v1/summary.json --output <new path>

Each new run's one-unit drop (clone - 500k) / (donor before - clone floor) is computed from its own Room 1 held-out
curve exactly as the pilot's analyzer does, then pooled with the pilot's E0 and A1 runs (docs/results/ppo-anchor-pilot.json,
8 runs per arm). The declared rule, applied mechanically: carry E0 forward unless BOTH (1) at least 3 more E0 runs than
A1 runs have a drop above 0.25 (strictly), and (2) E0's pooled median drop is more than 0.10 above A1's. Everything
else (Room 2, per-clone comparison, the drop averaged over 200k to 500k, each batch on its own) is reported and cannot
move the decision.

Record checks, refusing on failure: the evaluation summary matches the committed evaluation plan (line-ending safe);
the plan's pins of the training summary, the training plan and the declaration still match their files; every result
is attributable and pins the planned checkpoint; the training summary matches the committed training plan with every
run ok at 10 threads; E0's manifest records ent_coef 0 and no anchor; A1's runs carry the identical anchor record with
lambda 1 in every session and pass the anchor.csv checks of the pilot's analyzer (one row per update, replays only where
a resumed session explains them, nonzero anchor gradient); nothing names the fresh v2 held-out sets. Resumed runs are
reported by name, with the commit of every session.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_ppo_anchor_pilot import (  # noqa: E402
    FINAL,
    UNREAD_SETS,
    anchored_run_problems,
    e0_problems,
    load,
    one_unit_drop,
    pin_problems,
    replayed_anchor_rows,
    session_commits,
    sha,
)
from celeste_rl.texthash import matches_text_hash, text_sha256  # noqa: E402

ARMS = ("A1", "E0")
HIGH, COUNT_GAP, MEDIAN_GAP = 0.25, 3, 0.10  # the declared thresholds: "above", "at least", "more than"
DECLARATION = REPO / "config/ppo-anchor-tiebreak.json"
# Half of the decision's inputs are the pilot's drops: the pilot result is pinned by its line-ending safe text hash.
PILOT_RESULT = REPO / "docs/results/ppo-anchor-pilot.json"
PILOT_RESULT_SHA256 = "fbbf136858b4d6b7f198d705814dee7347c9bed5aa1f20c00a3ab8758c6911ea"
CHECKPOINTS_PER_RUN = 5
ROOM2_FINALS_NEEDED = 6  # the pilot's "at least 3 of 4 canonical finals at 45 or more", over 8 pooled runs


def room2_check(v1_median: float, finals: list[int], threshold: float) -> dict:
    """The pilot's Room 2 conditions on the pooled runs of one arm: the Room 2 v1 median at least the threshold, and at
    least 6 of 8 canonical finals at 45 or more. Applied to whichever arm the rule names; it never changes the answer."""
    at_45 = sum(clears >= 45 for clears in finals)
    return {"v1_median": v1_median, "v1_threshold": threshold, "v1_condition_met": v1_median >= threshold,
            "canonical_finals_at_45_or_more": f"{at_45} of {len(finals)}",
            "finals_condition_met": at_45 >= ROOM2_FINALS_NEEDED,
            "passes": v1_median >= threshold and at_45 >= ROOM2_FINALS_NEEDED}


def decide_tiebreak(e0_drops: list[float], a1_drops: list[float]) -> dict:
    """The declared rule: A1 only if E0 has at least COUNT_GAP more runs above HIGH AND E0's median drop is more than
    MEDIAN_GAP above A1's; otherwise E0 (the declared tie-break, simplicity and cost)."""
    e0_high, a1_high = sum(drop > HIGH for drop in e0_drops), sum(drop > HIGH for drop in a1_drops)
    # Rounded before the strict comparison: in floating point 0.4 - 0.3 is 0.10000000000000003, which would count an
    # exact 0.10 gap as "more than 0.10" (the drops themselves are rounded to 4 decimals).
    gap = round(statistics.median(e0_drops) - statistics.median(a1_drops), 6)
    count_met, median_met = e0_high - a1_high >= COUNT_GAP, gap > MEDIAN_GAP
    carry = "A1" if count_met and median_met else "E0"
    return {"carry_forward": carry, "e0_runs_above_0.25": e0_high, "a1_runs_above_0.25": a1_high,
            "count_gap": e0_high - a1_high, "count_condition_met": count_met,
            "median_drop": {"E0": round(statistics.median(e0_drops), 4), "A1": round(statistics.median(a1_drops), 4)},
            "median_gap_e0_minus_a1": round(gap, 4), "median_condition_met": median_met,
            "why": (f"A1 only if E0 has at least {COUNT_GAP} more runs above {HIGH} AND a median more than {MEDIAN_GAP} "
                    f"higher; here the count gap is {e0_high - a1_high} ({'met' if count_met else 'not met'}) and the "
                    f"median gap {gap:.4f} ({'met' if median_met else 'not met'}), so {carry}")}


def drops_along(route_macros: list[float], donor_before: float, clone_floor: float) -> list[float]:
    """The one-unit drop at every checkpoint after the clone (route_macros[0] is the clone point)."""
    return [one_unit_drop(route_macros[0], value, donor_before, clone_floor) for value in route_macros[1:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plan_path = REPO / "config/campaign-ppo-tiebreak-eval.json"
    plan, summary = load(plan_path), load(REPO / args.summary)
    if not matches_text_hash(plan_path, summary["plan_sha256"]):
        raise SystemExit("the evaluation summary does not match the committed evaluation plan")
    train_path = REPO / "config/campaign-ppo-tiebreak-train.json"
    train_summary = load(REPO / plan["training_campaign"])
    if (not matches_text_hash(train_path, train_summary["plan_sha256"])
            or any(r["status"] != "ok" for r in train_summary["results"])
            or train_summary.get("threads_per_job") != 10):
        raise SystemExit("the training summary does not match the committed training plan, has a run not ok, or did "
                         "not record 10 threads")
    pins = pin_problems(plan, REPO / plan["training_campaign"], train_path, DECLARATION, "declaration_sha256")
    if pins:
        raise SystemExit("the evaluation plan's pins do not hold:\n  " + "\n  ".join(pins))
    planned = {run["id"]: run for run in plan["runs"]}
    named = [run["id"] for run in plan["runs"] if any(name in " ".join(run["command"]) for name in UNREAD_SETS)]
    if named:
        raise SystemExit(f"the evaluation plan names a fresh held-out set that must stay unread: {named}")
    results = {}
    for item in summary["results"]:
        if item["status"] != "ok":
            raise SystemExit(f"{item['id']} ended {item['status']}")
        result_path = REPO / item["artifact"]["result_file"]
        if sha(result_path) != item["artifact"]["result_sha256"]:
            raise SystemExit(f"{item['id']}: result file changed since the run")
        result = load(result_path)
        if result["checkpoint_sha256"] != planned[item["id"]]["checkpoint_sha256"] or not result["attributable"]:
            raise SystemExit(f"{item['id']}: wrong checkpoint or not attributable")
        if any(name in json.dumps(result) for name in UNREAD_SETS):
            raise SystemExit(f"{item['id']}: the result names a fresh held-out set that must stay unread")
        results[item["id"]] = result
    if set(results) != set(planned):
        raise SystemExit(f"missing evaluations: {sorted(set(planned) - set(results))}")

    problems = []
    train_entries = {entry["id"]: entry for entry in load(train_path)["runs"]}
    for k in range(4):
        for arm in ARMS:
            folder = REPO / f"runs/train/ppo-tiebreak-{arm}-k{k}"
            problems += e0_problems(folder) if arm == "E0" else anchored_run_problems(folder, 1.0)
            config = load(folder / "manifest.json")["config"]
            entry = train_entries[f"ppo-tiebreak-{arm}-k{k}"]
            if config.get("seed") != 50 + k or config.get("seed") != entry["seed"]:
                problems.append(f"{folder}: seed {config.get('seed')}, planned {50 + k}")
            if config.get("init_from") != entry["init_from"]:
                problems.append(f"{folder}: started from {config.get('init_from')}, planned {entry['init_from']}")
    if problems:
        raise SystemExit("record checks failed:\n  " + "\n  ".join(problems))

    resumed, commits, replayed = {}, {}, {}
    for record in train_summary["results"]:
        manifest = load(REPO / record["run_dir"] / "manifest.json")
        sessions = len(manifest["sessions"])
        attempts = len(record.get("attempts", [])) or 1
        commits[record["id"]] = session_commits(manifest)
        if sessions > 1 or attempts > 1:
            resumed[record["id"]] = {"sessions": sessions, "runner_attempts": attempts,
                                     "session_commits": commits[record["id"]],
                                     "note": "valid, but not bit-identical to an uninterrupted run"}
        anchor_csv = REPO / record["run_dir"] / "anchor.csv"
        if anchor_csv.exists():
            count = replayed_anchor_rows(list(csv.DictReader(anchor_csv.read_text(encoding="utf-8").splitlines())), manifest)
            if count:
                replayed[record["id"]] = count

    retention = load(REPO / "docs/results/retention-pilot.json")
    clone_floor = retention["floor"]["clone"]["route_macro"]
    clones = load(REPO / "docs/results/mixed-self-distillation-clone.json")["clones"]
    if not matches_text_hash(PILOT_RESULT, PILOT_RESULT_SHA256):
        raise SystemExit(f"{PILOT_RESULT.name} is not the pilot result the declaration pools with (text sha256 "
                         f"{PILOT_RESULT_SHA256[:16]}...)")
    pilot = load(PILOT_RESULT)
    if pilot["clone_floor"] != clone_floor:
        raise SystemExit("the pilot result and the retention floor disagree on the clone floor")

    new_runs, batches = {}, {arm: {"pilot": [], "new": []} for arm in ARMS}
    for arm in ARMS:
        for k in range(4):
            clone = clones[f"SD-k{k}"]
            curve = [clone["room1"]["route_macro"]]
            names = sorted(i for i in planned if i.startswith(f"tb-room1-{arm}-k{k}-step_"))
            for entry_id in names:
                curve.append(round(results[entry_id]["route_macro_success_rate"], 4))
            if len(names) != CHECKPOINTS_PER_RUN or not names[-1].endswith(FINAL):
                raise SystemExit(f"{arm}-k{k}: {len(names)} Room 1 checkpoints evaluated, expected "
                                 f"{CHECKPOINTS_PER_RUN} ending at {FINAL}")
            along = drops_along(curve, clone["donor_before"], clone_floor)
            drop = along[-1]
            new_runs[f"{arm}-k{k}"] = {
                "ppo_seed": 50 + k, "donor_before": clone["donor_before"],
                "room1_route_macro_clone_then_checkpoints": curve, "drop_at_each_checkpoint": [round(d, 4) for d in along],
                "drop_one_unit": drop, "room2_final_clears_of_50": results[f"tb-room2-{arm}-k{k}-final"]["successes"],
                "room2_v1_final": round(results[f"tb-room2v1-{arm}-k{k}-final"]["route_macro_success_rate"], 4)}
            batches[arm]["new"].append(drop)
            batches[arm]["pilot"].append(pilot["runs"][f"{arm}-k{k}"]["drop_one_unit"])

    pooled = {arm: batches[arm]["pilot"] + batches[arm]["new"] for arm in ARMS}
    decision = decide_tiebreak(pooled["E0"], pooled["A1"])

    def pilot_run(arm, k):
        return pilot["runs"][f"{arm}-k{k}"]

    def late_mean(run):  # mean drop over 200k to 500k (the last four checkpoints), same formula
        curve = run["room1_curve"] if "room1_curve" in run else None
        macros = [point["route_macro"] for point in curve] if curve else run["room1_route_macro_clone_then_checkpoints"]
        return round(statistics.mean(drops_along(macros, run["donor_before"], clone_floor)[1:]), 4)

    def all_drops(run):  # the drop at every checkpoint after the clone, for a pilot run or a new run
        macros = ([point["route_macro"] for point in run["room1_curve"]] if "room1_curve" in run
                  else run["room1_route_macro_clone_then_checkpoints"])
        return drops_along(macros, run["donor_before"], clone_floor)

    runs_by_arm = {arm: [pilot_run(arm, k) for k in range(4)] + [new_runs[f"{arm}-k{k}"] for k in range(4)] for arm in ARMS}
    reference = pilot["room2_v1_reference"]
    room2_by_arm = {arm: {"finals": [run["room2_final_clears_of_50"] for run in runs_by_arm[arm]],
                          "v1_median": round(statistics.median(run["room2_v1_final"] for run in runs_by_arm[arm]), 4)}
                    for arm in ARMS}
    carried = decision["carry_forward"]
    decision["room2_check_of_the_named_arm"] = {
        "arm": carried, **room2_check(room2_by_arm[carried]["v1_median"], room2_by_arm[carried]["finals"],
                                      reference["threshold"]),
        "note": ("the pilot's Room 2 conditions applied to whichever arm the rule names; a failure goes to the owner and "
                 "the orchestrator as a recorded choice before the confirmation and does not change the rule's answer")}

    descriptive = {
        "runs_with_a_drop_above_0.25_at_any_checkpoint_of_8": {
            arm: sum(max(all_drops(run)) > HIGH for run in runs_by_arm[arm]) for arm in ARMS},
        "room2_v1_pooled_median_against_the_m_k_reference": {
            arm: {"median": room2_by_arm[arm]["v1_median"], "m_k_median": reference["m_k_median"],
                  "threshold": reference["threshold"],
                  "at_least_threshold": room2_by_arm[arm]["v1_median"] >= reference["threshold"]} for arm in ARMS},
        "per_clone_drop": {f"k{k}": {"E0_pilot": pilot_run("E0", k)["drop_one_unit"], "E0_new": new_runs[f"E0-k{k}"]["drop_one_unit"],
                                     "A1_pilot": pilot_run("A1", k)["drop_one_unit"], "A1_new": new_runs[f"A1-k{k}"]["drop_one_unit"]}
                           for k in range(4)},
        "median_drop_by_batch": {arm: {name: round(statistics.median(values), 4) for name, values in batches[arm].items()}
                                 for arm in ARMS},
        "mean_drop_200k_to_500k_median_of_runs": {arm: round(statistics.median(
            [late_mean(pilot_run(arm, k)) for k in range(4)] + [late_mean(new_runs[f"{arm}-k{k}"]) for k in range(4)]), 4)
            for arm in ARMS},
        "room2_pooled": {arm: {
            "canonical_finals_median": statistics.median(
                [pilot_run(arm, k)["room2_final_clears_of_50"] for k in range(4)]
                + [new_runs[f"{arm}-k{k}"]["room2_final_clears_of_50"] for k in range(4)]),
            "canonical_finals_at_45_or_more_of_8": sum(
                clears >= 45 for clears in [pilot_run(arm, k)["room2_final_clears_of_50"] for k in range(4)]
                + [new_runs[f"{arm}-k{k}"]["room2_final_clears_of_50"] for k in range(4)]),
            "v1_median": round(statistics.median(
                [pilot_run(arm, k)["room2_v1_final"] for k in range(4)]
                + [new_runs[f"{arm}-k{k}"]["room2_v1_final"] for k in range(4)]), 4)} for arm in ARMS},
        "note": "descriptive only: none of this can move the declared decision",
    }
    report = {"label": load(DECLARATION)["label"], "analysis_code_sha256": text_sha256(Path(__file__)),
              "declaration_sha256": text_sha256(DECLARATION), "pilot_result_sha256": PILOT_RESULT_SHA256,
              "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "training_campaign_start_commit": train_summary["commit"],
              "training_session_commits": commits, "clone_floor": clone_floor, "decision": decision,
              "pooled_drops": {arm: {"pilot": batches[arm]["pilot"], "new": batches[arm]["new"]} for arm in ARMS},
              "new_runs": new_runs, "descriptive": descriptive, "resumed_runs": resumed or "none",
              **({"anchor_rows_replayed_by_a_resume": replayed} if replayed else {}),
              "record_checks": ("passed: plan hashes and the evaluation plan's pins, attributable results, 10 threads, "
                                "E0 ent_coef 0, A1 anchor records and anchor.csv, no v2 set named")}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("decision:", json.dumps(decision))
    print("median drop by batch:", json.dumps(descriptive["median_drop_by_batch"]))
    print("resumed runs:", json.dumps(report["resumed_runs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
