"""Kill the game during training and check that the run recovers, or aborts cleanly (Phase 0 crash recovery test).

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/crash_recovery_check.py

The real training loop (celeste_rl.training.run.train with SupervisedPPO and GameSession) runs on the game. A bridge
wrapper kills the game process (TerminateProcess, like a crash) at chosen moments:

  Recover run (4,096 accepted steps):
    A. mid-rollout: before environment step 700
    B. during the recovery reset that follows A (a second consecutive fault)
    C. during an episode-end reset inside a later rollout (the first reset after step 2,500)
    Pass: each kill ends as a recorded fault, the game is relaunched each time, the run finishes all 4,096 accepted
    steps with status "finished", and no update used data from a discarded rollout (the supervisor's contract,
    tested offline; here the step and episode counts are checked for consistency).

  Abort run: kill before step 300 and at every reset after it.
    Pass: TrainingAborted after three consecutive faults, two relaunches, status "aborted" and a loadable
    checkpoints/aborted.zip.

Results are written to runs/crash-recovery/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.run import TrainConfig, train  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO, TrainingAborted  # noqa: E402


class KillingBridge:
    """Passes calls to the lockstep bridge and kills the game process at scheduled moments."""

    def __init__(self, bridge: LockstepBridge, game: GameSession, kill_before_step: int | None = None,
                 kill_first_recovery_reset: bool = False, kill_first_reset_after_step: int | None = None,
                 kill_every_reset_after_first_kill: bool = False):
        self.bridge, self.game = bridge, game
        self.kill_before_step = kill_before_step
        self.kill_first_recovery_reset = kill_first_recovery_reset
        self.kill_first_reset_after_step = kill_first_reset_after_step
        self.kill_every_reset_after_first_kill = kill_every_reset_after_first_kill
        self.steps = self.resets = 0
        self.kills: list[dict] = []
        self._relaunches_seen = 0

    @property
    def http(self):
        return self.bridge.http

    def _kill(self, moment: str) -> None:
        process = self.game.process
        process.kill()
        process.wait(10)
        self.kills.append({"moment": moment, "step": self.steps, "reset": self.resets, "pid": process.pid,
                           "time": time.strftime("%H:%M:%S")})

    def reset(self):
        self.resets += 1
        relaunched = len(self.game.relaunches) > self._relaunches_seen
        self._relaunches_seen = len(self.game.relaunches)
        if self.kills and self.kill_every_reset_after_first_kill:
            self._kill("reset after first kill")
        elif relaunched and self.kill_first_recovery_reset:
            self.kill_first_recovery_reset = False
            self._kill("recovery reset after a relaunch")
        elif self.kill_first_reset_after_step is not None and self.steps >= self.kill_first_reset_after_step:
            self.kill_first_reset_after_step = None
            self._kill("episode reset inside a rollout")
        return self.bridge.reset()

    def step(self, buttons="", dash_only="", move_only=""):
        self.steps += 1
        if self.kill_before_step is not None and self.steps == self.kill_before_step:
            self.kill_before_step = None
            self._kill("mid-rollout step")
        return self.bridge.step(buttons, dash_only, move_only)

    def close(self):
        self.bridge.close()


def run_case(name: str, game_dir: Path, output_dir: Path, provenance: dict, config: TrainConfig, kills: dict) -> dict:
    game = GameSession(game_dir)
    http = CelesteBridge(output_dir / name / "episode.tas")
    bridge = KillingBridge(LockstepBridge(http), game, **kills)
    env = CelesteRoomEnv(bridge, disabled_inputs=config.disabled_inputs)
    run_dir = output_dir / name
    outcome = {"kills": bridge.kills}
    started = time.perf_counter()
    try:
        try:
            model = train(config, run_dir, env, provenance, on_fault=game.on_fault)
            outcome["raised"] = None
        except TrainingAborted as aborted:
            model = None
            outcome["raised"] = f"TrainingAborted: {aborted}"
    finally:
        env.close()
        game.close()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    progress_rows = (run_dir / "progress.csv").read_text(encoding="utf-8").splitlines()[1:] if (run_dir / "progress.csv").exists() else []
    outcome.update({
        "seconds": round(time.perf_counter() - started, 1),
        "status": manifest["status"],
        "accepted_steps": manifest["accepted_steps"],
        "fault_stats": manifest["fault_stats"],
        "relaunches": game.relaunches,
        "progress_rows": len(progress_rows),
        "episodes_recorded": sum(1 for _ in (run_dir / "episodes.jsonl").open(encoding="utf-8")) if (run_dir / "episodes.jsonl").exists() else 0,
        "aborted_checkpoint_loads": None,
    })
    aborted_path = run_dir / "checkpoints" / "aborted.zip"
    if aborted_path.exists():
        outcome["aborted_checkpoint_loads"] = SupervisedPPO.load(aborted_path, device="cpu").num_timesteps
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    if git["uncommitted_changes"] and not args.allow_dirty:
        print(runtime.DIRTY_MESSAGE + "\n  " + "\n  ".join(git["changed_paths"]))
        return 2
    output_dir = REPO / "runs" / "crash-recovery" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    manifest = runtime.collect(args.game_dir, CelesteBridge(output_dir / "prefix.tas")._prefix_lines())
    # Settings and file hashes are readable without the game; the running versions come from the last log.
    problems = runtime.check(manifest, runtime.load_pins())
    if problems and not args.allow_runtime_mismatch:
        print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
        return 2
    provenance = {**git, "runtime": manifest, "runtime_problems": problems,
                  "attributable": not git["uncommitted_changes"] and not problems}
    config = TrainConfig(total_timesteps=4096, n_steps=512, batch_size=256, n_epochs=2, checkpoint_every=1024,
                         eval_every=0)
    results = {**{k: v for k, v in provenance.items() if k != "runtime"}, "runtime": manifest,
               "config": config.__dict__, "cases": {}}

    def save():
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("Recover run: kills mid-rollout, during the following recovery reset, and during a later episode reset")
    recover = run_case("recover", args.game_dir, output_dir, provenance, config,
                       {"kill_before_step": 700, "kill_first_recovery_reset": True, "kill_first_reset_after_step": 2500})
    moments = [k["moment"] for k in recover["kills"]]
    stages = [f["stage"] for f in recover["fault_stats"]["faults"]]
    recover["passed"] = (
        recover["raised"] is None and recover["status"] == "finished" and recover["accepted_steps"] == 4096
        and moments == ["mid-rollout step", "recovery reset after a relaunch", "episode reset inside a rollout"]
        and len(recover["relaunches"]) == 3 and stages == ["rollout", "recovery reset", "rollout"]
        and recover["progress_rows"] == 8 and recover["fault_stats"]["accepted_transitions"] == 4096)
    results["cases"]["recover"] = recover
    save()
    print(f"   {'PASS' if recover['passed'] else 'FAIL'}: status {recover['status']}, accepted {recover['accepted_steps']}, "
          f"kills {moments}, fault stages {stages}, relaunches {len(recover['relaunches'])}, {recover['seconds']} s")

    print("Abort run: kill mid-rollout and at every reset after it")
    abort = run_case("abort", args.game_dir, output_dir, provenance, config,
                     {"kill_before_step": 300, "kill_every_reset_after_first_kill": True})
    abort["passed"] = (
        abort["raised"] is not None and "3 consecutive" in abort["raised"] and abort["status"] == "aborted"
        and len(abort["relaunches"]) == 2 and abort["aborted_checkpoint_loads"] == 0)
    results["cases"]["abort"] = abort
    save()
    print(f"   {'PASS' if abort['passed'] else 'FAIL'}: {abort['raised']}, status {abort['status']}, "
          f"relaunches {len(abort['relaunches'])}, aborted checkpoint at {abort['aborted_checkpoint_loads']} steps, {abort['seconds']} s")

    passed = recover["passed"] and abort["passed"]
    results["passed"] = passed
    save()
    print(("Crash recovery passed." if passed else "CRASH RECOVERY FAILED.") + f" Results: {output_dir / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
