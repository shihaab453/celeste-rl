"""Descriptive sensitivity of a held-out A/B result to held-out states that coincide with demonstration states.

Run from the repo root with the RL interpreter, after scripts/analyze_heldout_ab.py has accepted the campaign:
    .venv-rl/Scripts/python.exe scripts/heldout_overlap_sensitivity.py --plan <evaluation plan> \
        --campaign runs/campaign/<ts>-<name>/summary.json --exclude <state id list json> --output <new path>

The exclusion file holds {"all_coinciding": [...], "late_coinciding": [...]}: state ids of the held-out set whose
(task frame, position, dashes) equal a demonstration state. For each list the paired route-macro analysis is
repeated without those states, using the committed analyzer's own validation and statistics, so the numbers
cannot drift from the primary. Descriptive only: never the primary, never a decision; removing states changes the
route weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analyze_heldout_ab import AnalysisError, _repo_path, analyse, file_sha256, validate_run  # noqa: E402
from celeste_rl.heldout import validate_manifest  # noqa: E402


def load_rows(plan: dict, campaign: dict) -> tuple[dict, list[int], list[str]]:
    """The accepted rows per (seed, arm), validated exactly as the primary analyzer does."""
    protocol = plan["evaluation_protocol"]
    heldout = json.loads(_repo_path(protocol["heldout_set"]).read_text(encoding="utf-8"))
    entries = validate_manifest(heldout)
    if heldout["sha256"] != protocol["heldout_sha256"]:
        raise AnalysisError("held-out manifest differs from the plan")
    by_id = {entry["state_id"]: entry for entry in entries}
    results = {entry["id"]: entry for entry in campaign["results"]}
    rows = {}
    for run in plan["runs"]:
        if file_sha256(_repo_path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise AnalysisError(f"{run['id']} checkpoint differs from its predeclared hash")
        _, run_rows = validate_run(run, results[run["id"]], campaign["commit"], protocol, heldout, by_id)
        rows[(run["train_seed"], run["arm"])] = run_rows
    return rows, plan["analysis"]["matched_training_seeds"], heldout["routes"]


def without(rows: dict, excluded: set[str]) -> dict:
    return {key: [row for row in values if row["state_id"] not in excluded] for key, values in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise AnalysisError(f"{args.output} already exists")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
        if campaign.get("plan_sha256") != file_sha256(args.plan):
            raise AnalysisError("campaign summary does not identify the exact plan")
        lists = json.loads(args.exclude.read_text(encoding="utf-8"))
        rows, seeds, routes = load_rows(plan, campaign)
        declaration = plan["analysis"]
        report = {"label": "descriptive sensitivity only; never the primary", "plan": str(args.plan),
                  "exclusions": str(args.exclude), "results": {}}
        for name in ("all_coinciding", "late_coinciding"):
            excluded = set(lists[name])
            if not excluded:
                report["results"][name] = {"excluded_states": 0, "note": "nothing to exclude"}
                continue
            stats = analyse(without(rows, excluded), seeds, routes, declaration["bootstrap_samples"],
                            declaration["bootstrap_seed"])
            report["results"][name] = {"excluded_states": len(excluded), "primary": stats["primary"]}
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    except (OSError, KeyError, json.JSONDecodeError, AnalysisError, ValueError) as error:
        print(f"Sensitivity refused: {error}")
        return 2
    for name, result in report["results"].items():
        primary = result.get("primary")
        print(f"{name}: {result['excluded_states']} states excluded" +
              (f"; B minus A {primary['B_minus_A']:+.1%}, p {primary['exact_two_sided_paired_sign_flip_p']:.4f}"
               if primary else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
