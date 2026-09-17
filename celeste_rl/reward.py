"""rew-v1: the room task's reward (Phase 2 spec, section 8).

  +1 on success (once), -1 on any failure ending, and a time cost on every step including the last
  (-0.001 per second of normal play, -1/60,000 per frame). No survival bonus. Shaping is off in v1; any
  later potential-based term gets its own version.

Known and accepted trade-off (decision D5): a policy certain to fail scores about -1.00002 by failing at
once and -1.03 by timing out, so it slightly prefers failing early. Trying for the full 30 seconds is still
better whenever its success chance exceeds about 1.5%.

The reward is computed from the ending cause only, never from anything the observation encoder produces,
and is returned outside the observation.
"""
from __future__ import annotations

from dataclasses import dataclass

from celeste_rl.endings import FAILURES, SUCCESS

REWARD_VERSION = "rew-v1"


@dataclass(frozen=True)
class RewardConfig:
    success: float = 1.0
    failure: float = -1.0
    time_per_frame: float = -1.0 / 60_000
    # The learner's discount (decision D2). Recorded here because any future shaping term must use the same value.
    gamma: float = 1.0


def reward_components(ending: str | None, config: RewardConfig) -> dict[str, float]:
    if ending is not None and ending != SUCCESS and ending not in FAILURES:
        raise ValueError(f"Unknown ending {ending!r}")
    return {
        "completion": config.success if ending == SUCCESS else 0.0,
        "failure": config.failure if ending in FAILURES else 0.0,
        "time": config.time_per_frame,
        "shaping": 0.0,
    }
