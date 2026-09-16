"""Check the lockstep mod's session rules against the real game, using raw sockets that misbehave on purpose.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/lockstep_fault_check.py

1. A connection cannot step before it has completed a reset.
2. A reset with the wrong start frame is rejected immediately, not left waiting.
3. A second connection replaces the first: the first is closed, and the second must reset before stepping.
4. A client that disconnects right after sending a step does not leak that step's reply to the next client.
5. A connection replaced at random points during its reset never strands the replacement (100 rounds).
6. A client that sends requests but never reads replies is closed, and a new client can recover.
7. After all of the above, the normal client still resets and steps correctly.
"""
from __future__ import annotations

import json
import random
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import DEFAULT_LOCKSTEP_PORT, LockstepBridge  # noqa: E402


class RawClient:
    def __init__(self):
        self.socket = socket.create_connection(("127.0.0.1", DEFAULT_LOCKSTEP_PORT), timeout=5)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = b""

    def send(self, message: dict) -> None:
        self.socket.sendall((json.dumps(message) + "\n").encode())

    def receive(self, timeout: float = 5) -> dict | None:
        """Next reply, or None if the server closed the connection."""
        self.socket.settimeout(timeout)
        while b"\n" not in self.buffer:
            chunk = self.socket.recv(1 << 16)
            if not chunk:
                return None
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def request(self, message: dict) -> dict | None:
        self.send(message)
        return self.receive()

    def close(self) -> None:
        self.socket.close()


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if passed else 'FAIL'} {name}{': ' + detail if detail else ''}")
    return passed


def main() -> int:
    process = game_process.launch(Path("C:/Projects/celeste-research-scratch/game-probe"), focus=False)
    http = CelesteBridge(REPO / "runs" / "lockstep-fault-check" / "episode.tas")
    results = []
    try:
        http.reset()  # write the prefix file, create the savestate, pause at the episode start
        start = http.start_frame

        print("1. Step before reset")
        a = RawClient()
        reply = a.request({"id": 1, "cmd": "step", "line": "1,R"})
        results.append(check("step refused on a fresh connection", "reset" in (reply or {}).get("error", ""), str(reply)[:120]))

        print("2. Reset with the wrong start frame")
        began = time.perf_counter()
        reply = a.request({"id": 2, "cmd": "reset", "start_frame": start - 1})
        elapsed = time.perf_counter() - began
        results.append(check("wrong prefix rejected", "expected a prefix" in (reply or {}).get("error", ""), str(reply)[:120]))
        results.append(check("rejected quickly", elapsed < 1.0, f"{elapsed * 1000:.0f} ms"))

        reply = a.request({"id": 3, "cmd": "reset", "start_frame": start})
        results.append(check("correct reset accepted", (reply or {}).get("frame") == start, str(reply)[:80]))
        reply = a.request({"id": 4, "cmd": "step", "line": "1,R"})
        results.append(check("step accepted after reset", (reply or {}).get("frame") == start + 1))

        print("3. Second connection replaces the first")
        b = RawClient()
        time.sleep(0.2)
        try:
            a.send({"id": 5, "cmd": "step", "line": "1,R"})
            closed = a.receive(timeout=2) is None
        except OSError:
            closed = True
        results.append(check("first connection closed by the server", closed))
        reply = b.request({"id": 1, "cmd": "step", "line": "1,L"})
        results.append(check("second connection must reset before stepping", "reset" in (reply or {}).get("error", ""), str(reply)[:80]))
        reply = b.request({"id": 2, "cmd": "reset", "start_frame": start})
        results.append(check("second connection resets", (reply or {}).get("frame") == start and (reply or {}).get("id") == 2))
        b.close()

        print("4. Disconnect right after sending a step")
        c = RawClient()
        c.request({"id": 1, "cmd": "reset", "start_frame": start})
        c.send({"id": 7777, "cmd": "step", "line": "1,R"})
        c.close()
        d = RawClient()
        reply = d.request({"id": 1, "cmd": "reset", "start_frame": start})
        results.append(check("next connection receives only its own reply",
                             reply is not None and reply.get("id") == 1 and reply.get("frame") == start, str(reply)[:80]))
        try:
            extra = d.receive(timeout=0.5)
        except TimeoutError:
            extra = "nothing"
        results.append(check("no leftover reply afterwards", extra == "nothing", str(extra)[:80]))
        d.close()

        print("5. Replace a connection at random points during its reset (100 rounds)")
        rng = random.Random(0)
        stranded = []
        worst = 0.0
        for round_number in range(100):
            a = RawClient()
            a.request({"id": 1, "cmd": "reset", "start_frame": start})
            a.send({"id": 2, "cmd": "reset", "start_frame": start})
            time.sleep(rng.uniform(0, 0.015))
            a.close()
            b = RawClient()
            began = time.perf_counter()
            try:
                reset = b.request({"id": 1, "cmd": "reset", "start_frame": start})
                step = b.request({"id": 2, "cmd": "step", "line": "1,R"})
                ok = (reset or {}).get("id") == 1 and (reset or {}).get("frame") == start and \
                    (step or {}).get("id") == 2 and (step or {}).get("frame") == start + 1
            except (TimeoutError, OSError) as error:
                ok, reset = False, str(error)
            worst = max(worst, time.perf_counter() - began)
            if not ok:
                stranded.append((round_number, str(reset)[:80]))
            b.close()
        results.append(check("replacement connection always resets and steps", not stranded,
                             f"worst reset+step {worst * 1000:.0f} ms, failures {stranded[:3]}"))

        print("6. A client that stops reading replies")
        flood = RawClient()
        flood.request({"id": 1, "cmd": "reset", "start_frame": start})
        began = time.perf_counter()
        try:
            # Steps are answered as fast as the game can run, so unread replies fill the socket buffer and
            # the mod's 2 s send timeout has to close the connection.
            for i in range(2, 200_000):
                flood.send({"id": i, "cmd": "step", "line": "1"})
        except OSError:
            pass  # the server closed the connection or the send buffer filled after it stopped reading
        flooded_for = time.perf_counter() - began
        recover = RawClient()
        began = time.perf_counter()
        reply = recover.request({"id": 1, "cmd": "reset", "start_frame": start})
        recovered_in = time.perf_counter() - began
        results.append(check("new client recovers after a non-reading client", (reply or {}).get("frame") == start,
                             f"flood ran {flooded_for:.1f} s, recovery reset took {recovered_in * 1000:.0f} ms"))
        recover.close()
        flood.close()

        print("7. Normal client afterwards")
        bridge = LockstepBridge(http)
        observation = bridge.reset()
        steps = [bridge.step("R").tas_frame for _ in range(10)]
        results.append(check("reset and 10 steps", observation.tas_frame == start and steps == list(range(start + 1, start + 11))))
        bridge.close()
    finally:
        game_process.stop(process)

    passed = all(results)
    print(f"{sum(results)}/{len(results)} checks passed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
