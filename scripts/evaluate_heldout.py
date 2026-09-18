"""Measure a policy against the frozen held-out entry states: the project's real Phase 3 criterion.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/evaluate_heldout.py --checkpoint runs/train/<run>/checkpoints/latest.zip

`scripts/evaluate_checkpoint.py` measures one start many times, which says how reliable a policy is around a
single route. This measures many starts, which says whether it can play the room. They are different claims and
the second is the one the roadmap asks for.

Each held-out state is replayed from the canonical start and the policy then plays from there, once per state
by default, so the reported rate is over states rather than over repeats of one state. The frozen set's sha256
is recorded next to the result, so a number always names the test set it was measured against.

**On the criterion itself.** The roadmap says 99% of 200 held-out states. As a point estimate that is 198 of
200, but 200 of 200 has a 95% Wilson lower bound of about 98.1%, so 200 states can never demonstrate a rate
above 99% with confidence. This script reports the interval and leaves the judgement to the reader rather than
printing a pass or fail against a threshold that cannot be met as written.

Results go to runs/heldout-evaluation/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.reward import RewardConfig  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.training.game import GameSession  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from evaluate_checkpoint import wilson  # noqa: E402

DEFAULT_SET = REPO / "config" / "heldout_starts.json"


def play(model, env: CelesteRoomEnv, start: Start, deterministic: bool) -> dict:
    obs, info = env.reset(options={"start": start})
    if info["start"] != "archive":
        return {"ending": None, "problem": info["start_problem"]}
    length = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        length += 1
        if terminated or truncated:
            return {"ending": info["ending"], "length": length, "max_x": info["player"] and info["player"]["x"],
                    "problem": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starts", type=Path, default=DEFAULT_SET)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--repeats", type=int, default=1, help="episodes per held-out state")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-version", default="rew-v2")
    parser.add_argument("--shaping-scale", type=float, default=2.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    if refusal:
        print(refusal)
        return 2
    if not args.starts.exists():
        print(f"No held-out set at {args.starts}. Build one with scripts/make_heldout_starts.py.")
        return 2
    frozen = json.loads(args.starts.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps([e["lines"] for e in frozen["entries"]], sort_keys=True).encode()).hexdigest()
    if digest != frozen["sha256"]:
        print(f"{args.starts} does not match its own sha256. It has been edited since it was frozen.")
        return 2

    output_dir = REPO / "runs" / "heldout-evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    game = GameSession(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    env = CelesteRoomEnv(LockstepBridge(http),
                         reward_config=RewardConfig(version=args.reward_version, shaping_scale=args.shaping_scale))
    episodes = []
    try:
        problems = runtime.check(runtime.collect(args.game_dir, http._prefix_lines()), runtime.load_pins())
        if problems and not args.allow_runtime_mismatch:
            print("Runtime differs from the pins:\n  " + "\n  ".join(problems))
            return 2
        model = SupervisedPPO.load(args.checkpoint, env=env, device="cpu")
        model.set_random_seed(args.seed)
        print(f"{args.checkpoint} against {len(frozen['entries'])} held-out starts "
              f"({args.repeats} episode(s) each), set {frozen['sha256'][:16]}")
        for index, entry in enumerate(frozen["entries"], start=1):
            start = Start(tuple(entry["lines"]), tuple(entry["position"]), "1", entry["dashes"])
            for _ in range(args.repeats):
                result = play(model, env, start, args.deterministic)
                episodes.append({**result, "start_frames": entry["frames"], "route": entry["route"],
                                 "start_x": entry["position"][0]})
            if index % 25 == 0:
                done = [e for e in episodes if e["problem"] is None]
                rate = sum(1 for e in done if e["ending"] == "success") / len(done) if done else 0
                print(f"  {index}/{len(frozen['entries'])}: {rate:.0%} so far")
    finally:
        env.close()
        game.close()

    usable = [e for e in episodes if e["problem"] is None]
    stale = len(episodes) - len(usable)
    successes = [e for e in usable if e["ending"] == "success"]
    low, high = wilson(len(successes), len(usable)) if usable else (0.0, 1.0)
    # Where it fails matters more than how often: a policy that only clears from near the exit is not playing.
    by_depth = {}
    for episode in usable:
        bucket = f"x {int(episode['start_x'] // 40) * 40}-{int(episode['start_x'] // 40) * 40 + 39}"
        hit, total = by_depth.get(bucket, (0, 0))
        by_depth[bucket] = (hit + (episode["ending"] == "success"), total + 1)
    results = {
        **git, "runtime_problems": problems, "attributable": runtime.attributable(git, problems),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "heldout_set": str(args.starts), "heldout_sha256": frozen["sha256"],
        "heldout_states": len(frozen["entries"]), "repeats": args.repeats,
        "deterministic": args.deterministic, "evaluation_seed": args.seed,
        "episodes": len(usable), "stale_starts": stale,
        "successes": len(successes), "success_rate": len(successes) / len(usable) if usable else None,
        "wilson_95": [round(low, 4), round(high, 4)],
        "endings": dict(Counter(e["ending"] for e in usable)),
        "success_length_median": statistics.median(e["length"] for e in successes) if successes else None,
        "by_start_x": {k: {"successes": v[0], "episodes": v[1], "rate": round(v[0] / v[1], 3)}
                       for k, v in sorted(by_depth.items())},
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nheld-out success {len(successes)}/{len(usable)} = "
          f"{results['success_rate']:.1%}, 95% interval {low:.1%} to {high:.1%}")
    if stale:
        print(f"  {stale} starts no longer replay and were excluded")
    print("  by starting x: " + ", ".join(f"{k} {v['rate']:.0%} ({v['episodes']})"
                                          for k, v in results["by_start_x"].items()))
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
