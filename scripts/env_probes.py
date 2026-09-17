"""Live probes for CelesteRoomEnv against the real game (Phase 2 spec, section 11, probes P1-P10).

Run from the repo root with the RL interpreter (the CelesteRLLockstep mod must be installed, Steam running):
    .venv-rl/Scripts/python.exe scripts/env_probes.py
    .venv-rl/Scripts/python.exe scripts/env_probes.py --probes P1,P5 --episodes 50

P11 (extras do not change the game) is scripts/extras_noninterference.py.

  P1  exit route, 10 repeats: success once at the recorded transition step, identical observations
  P2  death route, 10 repeats: death on the death event step, identical observations
  P3  neutral inputs: timeout at exactly 1,800
  P4  a pause and unpause cycle repeated: still timeout at exactly 1,800
  P5  named mechanics with expected values: jump speed, held versus tapped jump, alternate jump binding
      re-press, buffered jump window (only while held), crouch collider, crouch dash, dash-only opposing directions
  P6  100 resets, then 10 repeats of the exit, death and a movement route with 5 / 50 / 500 ms client delays:
      identical observations and events
  P7  random policy, N episodes with all 24 inputs and N with pause, quick restart and journal disabled: every
      ending classified; zero faults with menus disabled; with menus enabled every fault is replayed to check
      it reproduces
  P8  random trajectories checked against the game's own collision query: no schema violations, tile cells
      that the grid marks solid collide, empty cells do not, the player's collider never overlaps a solid
  P9  throughput, matched runs of the environment and the raw bridge with the same actions and reset points:
      the environment keeps at least 80% of raw steps/s; its own per-step overhead p95 under 200 microseconds
  P10 environment runs (exit, death, random episodes) replayed as plain TAS playback with export: every
      expected player frame matches

Results are written to runs/env-probes/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process, runtime  # noqa: E402
from celeste_rl.actions import INDEX, parse_line, to_line, to_parts  # noqa: E402
from celeste_rl.bridge import BridgeError, CelesteBridge, format_input_line  # noqa: E402
from celeste_rl.endings import DEATH, SUCCESS, TIMEOUT  # noqa: E402
from celeste_rl.env import BridgeFault, CelesteRoomEnv  # noqa: E402
from celeste_rl.export_compare import compare_files  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, CELL_SIZE, DEADLINE_FRAMES, GRID_ANCHOR, GRID_SIZE, MENU_INPUTS  # noqa: E402

ALL_PROBES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
N = len(ACTION_INPUTS)


# Plumbing

class RecordingBridge:
    """Passes calls to the lockstep bridge, keeping the last reply, optional per-frame states and bridge time."""

    def __init__(self, bridge: LockstepBridge):
        self.bridge = bridge
        self.last = None
        self.record = False
        self.trace: list[dict] = []
        self.bridge_seconds: list[float] = []
        self.reset_seconds: list[float] = []

    @property
    def http(self):
        return self.bridge.http

    def reset(self):
        start = time.perf_counter()
        observation = self.bridge.reset()
        self.reset_seconds.append(time.perf_counter() - start)
        self.last = observation
        self.trace = [{"frame": observation.tas_frame, "state": observation.state}] if self.record else []
        return observation

    def step(self, buttons="", dash_only="", move_only=""):
        start = time.perf_counter()
        observation = self.bridge.step(buttons, dash_only, move_only)
        self.bridge_seconds.append(time.perf_counter() - start)
        self.last = observation
        if self.record:
            self.trace.append({"frame": observation.tas_frame, "state": observation.state})
        return observation

    def close(self):
        pass  # the probe session owns the real bridge


def vector(buttons: str = "", dash_only: str = "", move_only: str = "") -> np.ndarray:
    return parse_line(format_input_line(buttons, dash_only, move_only))


def route_vectors(name: str) -> tuple[list[np.ndarray], dict]:
    route = json.loads((REPO / "tests" / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))
    return [vector(buttons) for buttons in route["actions"]], route


def digest(obs: dict) -> str:
    h = hashlib.sha1()
    for key in sorted(obs):
        h.update(key.encode())
        h.update(np.ascontiguousarray(obs[key]).tobytes())
    return h.hexdigest()


def random_actions(seed: int, count: int, disabled: tuple[str, ...], p: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = (rng.random((count, N)) < p).astype(np.int8)
    for name in disabled:
        actions[:, INDEX[name]] = 0
    return actions


def play(env: CelesteRoomEnv, actions, delay_after: int | None = None, delay: float = 0.0) -> dict:
    """Reset and play actions until the episode ends or the actions run out."""
    obs, _ = env.reset()
    digests, events, total = [digest(obs)], [], 0.0
    ending, step = None, 0
    for step, action in enumerate(actions, 1):
        obs, reward, terminated, _, info = env.step(action)
        digests.append(digest(obs))
        events.append(info["events"])
        total += reward
        if step == delay_after:
            time.sleep(delay)
        if terminated:
            ending = info["ending"]
            break
    return {"ending": ending, "step": step, "return": total, "digests": digests, "events": events}


class Session:
    def __init__(self, game_dir: Path, output_dir: Path):
        self.game_dir, self.output_dir = game_dir, output_dir
        self.process = game_process.launch(game_dir, focus=False)
        game_process.set_window_mode(self.process.pid, "minimized")
        self.http = CelesteBridge(output_dir / "episode.tas")
        self.lockstep = LockstepBridge(self.http)
        self.recorder = RecordingBridge(self.lockstep)

    def env(self, disabled: tuple[str, ...] = ()) -> CelesteRoomEnv:
        return CelesteRoomEnv(self.recorder, disabled_inputs=disabled)

    def close(self):
        self.lockstep.close()
        game_process.stop(self.process)


def result(passed: bool, **details) -> dict:
    return {"passed": bool(passed), **details}


# Probes

def probe_repeats(session: Session, route: str, expected_ending: str, repeats: int = 10) -> dict:
    actions, fixture = route_vectors(route)
    env = session.env()
    runs = [play(env, actions) for _ in range(repeats)]
    expected_step = fixture["transition_step"]
    outcomes = Counter((r["ending"], r["step"]) for r in runs)
    identical = all(r["digests"] == runs[0]["digests"] and r["events"] == runs[0]["events"] for r in runs)
    successes = sum(1 for events in runs[0]["events"] for e in events if e["type"] == "transition")
    return result(outcomes == Counter({(expected_ending, expected_step): repeats}) and identical,
                  expected=[expected_ending, expected_step], outcomes={f"{k[0]}@{k[1]}": v for k, v in outcomes.items()},
                  identical_across_repeats=identical, returns=sorted({round(r["return"], 6) for r in runs}),
                  transition_events_in_run=successes)


def probe_p1(session):
    return probe_repeats(session, "room1_exit_dash_route", SUCCESS)


def probe_p2(session):
    # The death route's fixture "transition_step" is the recorded death step (74).
    return probe_repeats(session, "room1_spike_death_route", DEATH)


def probe_p3(session):
    run = play(session.env(), [np.zeros(N, dtype=np.int8)] * (DEADLINE_FRAMES + 10))
    return result(run["ending"] == TIMEOUT and run["step"] == DEADLINE_FRAMES, ending=run["ending"], step=run["step"],
                  return_=round(run["return"], 6))


def probe_p4(session):
    pause = vector("S")
    cycle = [pause] + [np.zeros(N, dtype=np.int8)] * 29 + [pause] + [np.zeros(N, dtype=np.int8)] * 29
    actions = (cycle * (DEADLINE_FRAMES // len(cycle) + 2))[:DEADLINE_FRAMES + 10]
    try:
        run = play(session.env(disabled=()), actions)
    except BridgeFault as fault:
        return result(False, fault=str(fault))
    types = Counter(e["type"] for events in run["events"] for e in events)
    return result(run["ending"] == TIMEOUT and run["step"] == DEADLINE_FRAMES and types["pause"] > 0,
                  ending=run["ending"], step=run["step"], event_counts=dict(types))


def probe_p5(session):
    env = session.env(disabled=())
    rec = session.recorder
    checks = {}

    def raw():
        state, extras = rec.last.state, rec.last.extras
        return state, extras

    def steps(actions):
        """Play from a fresh reset; returns per-step (state, extras, ending)."""
        env.reset()
        out = []
        for action in actions:
            _, _, terminated, _, info = env.step(action)
            out.append((*raw(), info["ending"]))
            if terminated:
                break
        return out

    def speed_y(entry):
        return entry[0]["Player"]["Speed"]["Y"] if entry[0] else None

    def y(entry):
        return entry[0]["Player"]["Position"]["Y"] if entry[0] else None

    def on_ground(entry):
        return entry[0]["Player"]["OnGround"] if entry[0] else None

    neutral = np.zeros(N, dtype=np.int8)
    J, K = vector("J"), vector("JK")

    def first_grounded_after_leaving(run):
        """Index of the first grounded reply after the player has been airborne."""
        airborne = False
        for i, entry in enumerate(run):
            airborne = airborne or not on_ground(entry)
            if airborne and on_ground(entry):
                return i
        return None

    # Jump speed: pressing jump on the ground sets Speed.Y to JumpSpeed (-105) on that frame. OnGround in a reply is
    # sampled before that frame's movement, so it still reads True; the player is higher on the next frame.
    first = steps([J, neutral])
    checks["jump_speed"] = result(speed_y(first[0]) == -105 and y(first[1]) < 144, speed_y=speed_y(first[0]),
                                  y_next=y(first[1]))

    # Held versus tapped jump: holding keeps the variable jump going (VarJumpTime 0.2 s), so the apex is higher.
    tapped = steps([J] + [neutral] * 60)
    held = steps([J] * 60)
    apex_tapped, apex_held = min(y(e) for e in tapped), min(y(e) for e in held)
    landing = first_grounded_after_leaving(tapped)
    held_landing = first_grounded_after_leaving(held)
    checks["held_jump_higher"] = result(apex_held <= apex_tapped - 8, apex_tapped=apex_tapped, apex_held=apex_held,
                                        tapped_landing_index=landing, held_landing_index=held_landing)

    # Buffered jump: a jump press is buffered for 0.08 s, but only while the button stays held (Monocle clears the
    # buffer on release). Pressed 3 frames before landing and held: jumps on the landing frame. Pressed 3 frames
    # before and released: no jump. Pressed 8 frames before and held: the buffer has expired, no jump.
    def buffered(press_before: int, hold: bool):
        actions = [J] + [neutral] * 40
        for i in range(landing - press_before, len(actions) if hold else landing - press_before + 1):
            actions[i] = J
        run = steps(actions)
        return speed_y(run[landing]) == -105, round(run[landing - press_before][1]["input_buffers"]["Jump"], 3)

    inside_held, outside_held, inside_released = buffered(3, True), buffered(8, True), buffered(3, False)
    checks["buffered_jump_window"] = result(inside_held[0] and not inside_released[0] and not outside_held[0],
                                            pressed_3_before_held=inside_held, pressed_3_before_released=inside_released,
                                            pressed_8_before_held=outside_held)

    # Alternate binding: holding J through landing does not jump again; pressing K while J is held does.
    held_through = steps([J] * (held_landing + 6))
    no_rejump = all(speed_y(e) != -105 for e in held_through[held_landing:])
    k_press = steps([J] * (held_landing + 3) + [K] + [J] * 3)
    k_rejump = speed_y(k_press[held_landing + 3]) == -105
    checks["alternate_binding_repress"] = result(no_rejump and k_rejump, held_j_rejumps=not no_rejump, k_press_rejumps=k_rejump)

    # Crouch collider: ducking shrinks the collider from 11 to 6 px, and it returns after release.
    duck = steps([vector("D")] * 5 + [neutral] * 3)
    heights = [e[1]["player"]["Collider"]["H"] for e in duck]
    checks["crouch_collider"] = result(duck[4][1]["player"]["Ducking"] and heights[4] == 6 and heights[-1] == 11
                                       and not duck[-1][1]["player"]["Ducking"], heights=heights)

    # Crouch dash: the crouch dash binding starts a dash while ducking.
    crouch_dash = steps([vector("RZ")] + [vector("R")] * 3)
    states = [e[1]["player"]["State"] if e[1] else None for e in crouch_dash]
    ducking = [e[1]["player"]["Ducking"] if e[1] else None for e in crouch_dash]
    checks["crouch_dash"] = result(2 in states and any(d for s, d in zip(states, ducking) if s == 2), states=states, ducking=ducking)

    # Dash-only opposing directions: CelesteTAS resolves left + right to left and up + down to down. The dash aim is
    # read when the dash begins, after its opening freeze (4 frames), so the directions are held through that window.
    # Each case is set up so the result differs from the facing direction.
    def dash_dir(dash_only, pre=()):
        run = steps(list(pre) + [vector("X", dash_only=dash_only)] + [vector(dash_only=dash_only)] * 8)
        return next((e[1]["player"]["DashDir"] for e in run[len(pre):] if e[1] and e[1]["player"]["DashDir"] != {"X": 0, "Y": 0}), None)

    face_left = [vector("L")] * 2
    right_facing_left = dash_dir("R", face_left)
    left_right_facing_right = dash_dir("LR")
    up_down = dash_dir("UD")
    checks["dash_only_opposites"] = result(
        right_facing_left == {"X": 1, "Y": 0} and left_right_facing_right == {"X": -1, "Y": 0} and up_down == {"X": 0, "Y": 1},
        right_while_facing_left=right_facing_left, left_and_right_while_facing_right=left_right_facing_right,
        up_and_down=up_down)

    return result(all(c["passed"] for c in checks.values()), checks=checks)


def probe_p6(session):
    env = session.env()
    first, _ = env.reset()
    reference = digest(first)
    resets_identical = all(digest(env.reset()[0]) == reference for _ in range(100))
    routes = {"exit": route_vectors("room1_exit_dash_route")[0], "death": route_vectors("room1_spike_death_route")[0],
              "movement": list(random_actions(12345, 300, MENU_INPUTS, p=0.05))}
    delays = [0, 0, 0, 0, 0, 0, 0, 0.005, 0.05, 0.5]
    routes_identical = {}
    for name, actions in routes.items():
        runs = [play(env, actions, delay_after=40, delay=delay) for delay in delays]
        routes_identical[name] = all((r["digests"], r["events"], r["ending"]) == (runs[0]["digests"], runs[0]["events"], runs[0]["ending"])
                                     for r in runs)
    return result(resets_identical and all(routes_identical.values()), resets_identical_100=resets_identical,
                  routes_identical=routes_identical)


def random_episodes(session: Session, episodes: int, disabled: tuple[str, ...], seed_base: int, record_first: int = 0,
                    trajectory_hook=None) -> dict:
    env = session.env(disabled=disabled)
    endings, lengths, faults, recorded = Counter(), [], [], []
    previous = None
    for episode in range(episodes):
        seed = seed_base + episode
        actions = random_actions(seed, DEADLINE_FRAMES, disabled)
        session.recorder.record = episode < record_first
        try:
            run = play(env, actions)
        except BridgeFault as fault:
            faults.append({"seed": seed, "previous_ending": previous, "error": str(fault)[:300]})
            previous = "fault"
            continue
        finally:
            session.recorder.record = False
        previous = run["ending"]
        endings[run["ending"]] += 1
        lengths.append(run["step"])
        if episode < record_first:
            recorded.append({"seed": seed, "actions": [to_line(a) for a in actions[:run["step"]]],
                             "trace": session.recorder.trace, "ending": run["ending"]})
    return {"endings": dict(endings), "faults": faults, "median_length": statistics.median(lengths) if lengths else None,
            "recorded": recorded}


def probe_p7(session, episodes):
    menus_off = random_episodes(session, episodes, MENU_INPUTS, 10_000)
    menus_on = random_episodes(session, episodes, (), 20_000)
    reproductions = []
    for fault in menus_on["faults"]:
        env = session.env(disabled=())
        try:
            play(env, random_actions(fault["seed"], DEADLINE_FRAMES, ()))
            reproductions.append({"seed": fault["seed"], "reproduced": False})
        except BridgeFault as again:
            reproductions.append({"seed": fault["seed"], "reproduced": True, "error": str(again)[:300]})
    return result(not menus_off["faults"] and all(r["reproduced"] for r in reproductions),
                  menus_disabled={k: v for k, v in menus_off.items() if k != "recorded"},
                  all_inputs={k: v for k, v in menus_on.items() if k != "recorded"}, fault_reproductions=reproductions)


def probe_p8(session, episodes):
    env = session.env(disabled=MENU_INPUTS)
    rec = session.recorder
    samples = mismatches = violations = 0
    examples = []
    for episode in range(episodes):
        actions = random_actions(30_000 + episode, DEADLINE_FRAMES, MENU_INPUTS)
        obs, _ = env.reset()
        for step, action in enumerate(actions, 1):
            try:
                obs, _, terminated, _, _ = env.step(action)
            except BridgeFault as fault:
                violations += 1
                examples.append({"episode": episode, "step": step, "fault": str(fault)[:200]})
                break
            if not all(np.all(np.isfinite(v)) for v in obs.values()):
                violations += 1
            state, extras = rec.last.state, rec.last.extras
            if state is not None and step % 10 == 0 and not terminated:
                samples += 1
                bad = check_collision(rec.bridge, state, extras, obs["grid"][0])
                mismatches += len(bad)
                examples += [{"episode": episode, "step": step, **b} for b in bad[:3]]
            if terminated:
                break
    return result(samples > 0 and mismatches == 0 and violations == 0, samples=samples, mismatches=mismatches,
                  violations=violations, examples=examples[:20])


def check_collision(bridge: LockstepBridge, state: dict, extras: dict, solid_channel: np.ndarray) -> list[dict]:
    """Compare the grid's tile-solid cells with the game's collision query, and check the player's collider."""
    b = state["Level"]["Bounds"]
    collider = extras["player"]["Collider"]
    px, py = state["Player"]["Position"]["X"], state["Player"]["Position"]["Y"]
    col0 = math.floor((px + collider["X"] + collider["W"] / 2 - b["X"]) / CELL_SIZE) - GRID_ANCHOR
    row0 = math.floor((py + collider["Y"] + collider["H"] / 2 - b["Y"]) / CELL_SIZE) - GRID_ANCHOR
    rows = state["SolidsData"].replace("\r", "").split("\n")
    cells, rects = [], []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            row, col = row0 + i, col0 + j
            if not (0 <= row < len(rows) and 0 <= col < len(rows[row])):
                continue
            # Inset by 1 px so the check is about the cell itself, not shared edges.
            rects.append((b["X"] + col * CELL_SIZE + 1, b["Y"] + row * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2))
            cells.append((i, j, rows[row][col] != "0"))
    player_rect = (int(px + collider["X"]), int(py + collider["Y"]), int(collider["W"]), int(collider["H"]))
    answers = bridge.query_solids(rects + [player_rect])
    bad = []
    for (i, j, tile_solid), collides in zip(cells, answers):
        in_grid = bool(solid_channel[i, j])
        if tile_solid and not collides:
            bad.append({"cell": [i, j], "problem": "grid marks a solid tile but the game reports no collision"})
        if not in_grid and collides:
            bad.append({"cell": [i, j], "problem": "grid cell empty but the game reports a collision"})
    if answers[-1]:
        bad.append({"cell": None, "problem": f"player collider {player_rect} overlaps a solid"})
    return bad


def probe_p9(session, steps: int = 5000, repeats: int = 3):
    actions = random_actions(40_000, steps, MENU_INPUTS)
    env = session.env(disabled=MENU_INPUTS)
    rec = session.recorder

    # First pass through the environment finds the reset points, so the raw bridge can reset at the same steps.
    boundaries = []
    env.reset()
    for i, action in enumerate(actions):
        if env.step(action)[2]:
            boundaries.append(i)
            env.reset()
    boundary_set = set(boundaries)

    def env_run():
        rec.bridge_seconds.clear()
        env.reset()
        overhead = []
        start = time.perf_counter()
        for i, action in enumerate(actions):
            t0 = time.perf_counter()
            done = env.step(action)[2]
            overhead.append(time.perf_counter() - t0 - rec.bridge_seconds[-1])
            if done:
                env.reset()
        return steps / (time.perf_counter() - start), overhead

    def raw_run():
        bridge = session.lockstep
        bridge.reset()
        start = time.perf_counter()
        for i, action in enumerate(actions):
            bridge.step(*to_parts(action))
            if i in boundary_set:
                bridge.reset()
        return steps / (time.perf_counter() - start)

    env_rates, raw_rates, overhead = [], [], []
    rec.reset_seconds.clear()
    for _ in range(repeats):
        raw_rates.append(raw_run())
        rate, per_step = env_run()
        env_rates.append(rate)
        overhead += per_step
    overhead_us = np.array(overhead) * 1e6
    ratio = statistics.median(env_rates) / statistics.median(raw_rates)
    p50, p95, p99 = (float(np.percentile(overhead_us, q)) for q in (50, 95, 99))
    resets_ms = np.array(rec.reset_seconds) * 1000
    return result(ratio >= 0.8 and p95 < 200, env_steps_per_s=[round(r) for r in env_rates],
                  raw_steps_per_s=[round(r) for r in raw_rates], median_ratio=round(ratio, 3),
                  env_overhead_us={"p50": round(p50, 1), "p95": round(p95, 1), "p99": round(p99, 1)},
                  resets_per_run=len(boundaries) + 1,
                  reset_ms={"p50": round(float(np.percentile(resets_ms, 50)), 1),
                            "p95": round(float(np.percentile(resets_ms, 95)), 1)} if len(resets_ms) else None)


def play_plain_lines(client, tas_path: Path, export_path: Path, lines: list[str], warmup_frames: int) -> None:
    content = ["console load 1", str(warmup_frames), f"StartExportGameInfo {export_path.as_posix()}", *lines,
               "FinishExportGameInfo"]
    tas_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    total = warmup_frames + len(lines)
    client.play_tas(tas_path)
    deadline, started = time.perf_counter() + 120, False
    while True:
        info = client.info()
        if not started:
            started = info.running and info.current_frame < warmup_frames
        elif info.paused_at_end and info.current_frame == total:
            if not export_path.exists():
                raise RuntimeError("Plain TAS playback finished but wrote no export")
            return
        elif not info.running:
            raise RuntimeError(f"Plain TAS playback stopped early: {info}")
        if time.perf_counter() > deadline:
            raise RuntimeError(f"Plain TAS playback did not finish: {info}")
        time.sleep(0.01)


def probe_p10(session, random_count: int = 3):
    rec = session.recorder
    runs = []
    env = session.env()
    for route in ("room1_exit_dash_route", "room1_spike_death_route"):
        actions, _ = route_vectors(route)
        rec.record = True
        run = play(env, actions)
        rec.record = False
        runs.append({"name": route, "lines": [to_line(a) for a in actions[:run["step"]]], "trace": rec.trace})
    episodes = random_episodes(session, random_count, MENU_INPUTS, 50_000, record_first=random_count)
    runs += [{"name": f"random_seed{r['seed']}", "lines": r["actions"], "trace": r["trace"]} for r in episodes["recorded"]]

    # Plain playback uses the HTTP side; the lockstep session is closed first and reopened by the next reset.
    session.lockstep._socket.close()
    session.lockstep._socket = None
    comparisons = {}
    for index, run in enumerate(runs):
        export_path = session.output_dir / f"plain-export-{index}.txt"
        play_plain_lines(session.http.client, session.output_dir / f"plain-{index}.tas", export_path, run["lines"],
                         session.http.warmup_frames)
        comparisons[run["name"]] = compare_files(run["trace"], export_path).summary()
    return result(all(c["passed"] for c in comparisons.values()),
                  comparisons={k: {key: v[key] for key in ("passed", "expected_frames", "matched", "mismatched", "first_mismatch")}
                               for k, v in comparisons.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--probes", default=",".join(ALL_PROBES))
    parser.add_argument("--episodes", type=int, default=1000, help="P7 episodes per input setting")
    parser.add_argument("--collision-episodes", type=int, default=30, help="P8 episodes")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (exploratory only)")
    parser.add_argument("--allow-runtime-mismatch", action="store_true",
                        help="run although the runtime differs from config/pinned_runtime.json")
    args = parser.parse_args()
    selected = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
    unknown = set(selected) - set(ALL_PROBES)
    if unknown:
        parser.error(f"unknown probes {sorted(unknown)}")

    git = runtime.git_state()
    if git["uncommitted_changes"] and not args.allow_dirty:
        print(runtime.DIRTY_MESSAGE + "\n  " + "\n  ".join(git["changed_paths"]))
        return 2
    output_dir = REPO / "runs" / "env-probes" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    results = {**git, "args": {**vars(args), "game_dir": str(args.game_dir)}, "probes": {}}

    def save():
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    runners = {
        "P1": probe_p1, "P2": probe_p2, "P3": probe_p3, "P4": probe_p4, "P5": probe_p5, "P6": probe_p6,
        "P7": lambda s: probe_p7(s, args.episodes), "P8": lambda s: probe_p8(s, args.collision_episodes),
        "P9": probe_p9, "P10": probe_p10,
    }
    session = Session(args.game_dir, output_dir)
    try:
        results["runtime"] = runtime.collect(args.game_dir, session.http._prefix_lines())
        results["runtime_problems"] = runtime.check(results["runtime"], runtime.load_pins())
        if results["runtime_problems"]:
            print("Runtime differs from the pins:\n  " + "\n  ".join(results["runtime_problems"]))
            if not args.allow_runtime_mismatch:
                results["refused"] = "runtime mismatch"
                return 2
        for name in selected:
            start = time.perf_counter()
            try:
                outcome = runners[name](session)
            except (BridgeError, BridgeFault, RuntimeError, AssertionError, StopIteration, KeyError, TypeError) as error:
                outcome = result(False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            outcome["seconds"] = round(time.perf_counter() - start, 1)
            results["probes"][name] = outcome
            save()
            summary = {k: v for k, v in outcome.items() if k not in ("traceback", "checks", "examples")}
            print(f"{name} {'PASS' if outcome['passed'] else 'FAIL'} {json.dumps(summary, default=str)[:600]}")
            if "checks" in outcome:
                for check, value in outcome["checks"].items():
                    print(f"     {check}: {'PASS' if value['passed'] else 'FAIL'} {json.dumps(value, default=str)[:300]}")
    finally:
        session.close()
        save()
    passed = all(p["passed"] for p in results["probes"].values())
    attributable = not results["uncommitted_changes"] and not results["runtime_problems"]
    results["attributable"] = attributable
    save()
    print(("All probes passed." if passed else "SOME PROBES FAILED.")
          + (f" Commit {results['commit'][:7]}, pinned runtime." if attributable
             else " NOT ATTRIBUTABLE: uncommitted changes or a runtime differing from the pins."))
    print(f"Results: {output_dir / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
