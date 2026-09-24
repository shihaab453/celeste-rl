"""Apply a plan's predeclared stability gate to final-checkpoint play results (training side only).

Run from the repo root with the RL interpreter, after the play campaign:
    .venv-rl/Scripts/python.exe scripts/check_stability_gate.py --plan <training plan> \
        --play runs/campaign/<ts>-<name>/summary.json [--play <another summary.json>] --output <new path>

The training plan declares `stability_gate` = {"collapse_below": rate, "max_collapses_per_arm": n,
"protocol": {"episodes": e, "seed": s}}. Each declared run's final latest.zip must have exactly one completed play
(entry id `play-<run id>`, possibly spread over several play campaigns) whose result names that checkpoint's current
hash, the declared episode count and seed. A run collapses when its clear rate is below the threshold; the gate
passes when no arm has more collapses than allowed. Refuses rather than guesses if anything is missing or differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class GateError(ValueError):
    """The play results do not satisfy the plan's gate contract."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(rates: dict[str, dict[str, float]], collapse_below: float, max_collapses: int) -> dict:
    """rates: arm -> run id -> clear rate. Returns per-arm collapses and whether the gate passes."""
    arms = {}
    for arm, by_run in sorted(rates.items()):
        collapsed = sorted(run for run, rate in by_run.items() if rate < collapse_below)
        arms[arm] = {"runs": len(by_run), "collapsed": collapsed, "collapses": len(collapsed),
                     "passes": len(collapsed) <= max_collapses}
    return {"arms": arms, "passes": all(arm["passes"] for arm in arms.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--play", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise GateError(f"{args.output} already exists")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        declared = plan["stability_gate"]
        plays = {}
        for summary_path in args.play:
            for entry in json.loads(summary_path.read_text(encoding="utf-8"))["results"]:
                if entry["id"] in plays:
                    raise GateError(f"{entry['id']} appears in more than one play campaign")
                plays[entry["id"]] = entry
        rates: dict[str, dict[str, float]] = {}
        for run in plan["runs"]:
            entry = plays.get(f"play-{run['id']}")
            if entry is None or entry.get("status") != "ok":
                raise GateError(f"no completed play for {run['id']}")
            result = json.loads((REPO / entry["artifact"]["result_file"]).read_text(encoding="utf-8"))
            checkpoint_sha = file_sha256(REPO / run["checkpoint"])
            for field, value in (("checkpoint_sha256", checkpoint_sha),
                                 ("episodes", declared["protocol"]["episodes"]),
                                 ("evaluation_seed", declared["protocol"]["seed"])):
                if result.get(field) != value:
                    raise GateError(f"{run['id']} play {field} is {result.get(field)!r}, expected {value!r}")
            if result.get("attributable") is not True:
                raise GateError(f"{run['id']} play is not attributable")
            rates.setdefault(run["arm"], {})[run["id"]] = result["success_rate"]
        report = {"plan": str(args.plan), "plan_sha256": file_sha256(args.plan), "rates": rates,
                  **gate(rates, declared["collapse_below"], declared["max_collapses_per_arm"])}
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    except (OSError, KeyError, json.JSONDecodeError, GateError) as error:
        print(f"Gate refused: {error}")
        return 2
    for arm, result in report["arms"].items():
        print(f"arm {arm}: {result['collapses']}/{result['runs']} collapsed {result['collapsed']}")
    print(f"gate passes: {report['passes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
