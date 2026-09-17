"""The progress potential rew-v2 shapes with, pinned to the offline validation it passed.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest

`scripts/potential_check.py` accepted the spike-weighted tile distance by replaying the recorded room 1 clear
and a recorded death route (runs/potential-check/20260917-230728). These tests hold `celeste_rl/potential.py`
to the numbers that check reported, so a later change to the geometry, the costs or the parameters cannot
quietly move the potential the reward is built on.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from celeste_rl.potential import RoomPotential, hazard_potential, room, tile_distances, tile_potential

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "env_replies.json").read_text(encoding="utf-8"))
START = next(s for s in FIXTURE["room1_exit_dash_route"]["samples"] if "start" in s["reasons"])["state"]

# From runs/potential-check/20260917-230728, candidate tiles_hazard, rounded to 3 decimals as reported there.
VALIDATED_CELLS = {
    (19, 144): 0.225,   # the canonical start, on the ledge
    (39, 144): 0.234,   # the ledge edge
    (48, 144): 0.205,   # over the first pit
    (60, 152): 0.180,   # falling into the pit
    (60, 164): 0.143,   # the pit floor beside the spikes
    (84, 128): 0.365,   # up on the platform past the pit
    (261, 60): 0.934,   # under the exit
    (261, 4): 1.000,    # the exit gap
}


class RoomGeometryTests(unittest.TestCase):
    def test_room_1_geometry(self):
        geometry = room(START)
        self.assertEqual((geometry["rows"], geometry["cols"]), (23, 40))
        # The exit is the open stretch of the top row, columns 31 to 35 (x 248 to 288).
        self.assertEqual([col for _, col in geometry["exits"]], [31, 32, 33, 34, 35])
        self.assertTrue(geometry["spikes"], "room 1 has spikes and the hazard cost needs them")


class HazardPotentialTests(unittest.TestCase):
    def setUp(self):
        self.potential = hazard_potential(room(START))

    def test_matches_the_validated_check(self):
        for (x, y), expected in VALIDATED_CELLS.items():
            with self.subTest(cell=(x, y)):
                self.assertAlmostEqual(self.potential(x, y), expected, places=3)

    def test_the_exit_is_the_maximum_and_the_start_is_far_from_it(self):
        self.assertEqual(self.potential(261, 4), 1.0)
        self.assertLess(self.potential(19, 144), 0.3)

    def test_it_does_not_pay_the_agent_to_enter_the_spike_pit(self):
        """The trap the plain tile distance falls into: the pit must be worth less than the ledge it leaves."""
        ledge, over_pit, pit_floor = self.potential(39, 144), self.potential(48, 144), self.potential(60, 164)
        self.assertLess(over_pit, ledge)
        self.assertLess(pit_floor, over_pit)

    def test_the_plain_tile_distance_still_falls_into_it(self):
        """Kept as the reason the plain distance is rejected: if this ever stops being true, re-run the check."""
        geometry = room(START)
        plain = tile_potential(geometry, tile_distances(geometry))
        self.assertGreater(plain(48, 144), plain(19, 144))


class RoomPotentialTests(unittest.TestCase):
    def setUp(self):
        self.potential = RoomPotential(START)

    def test_value_follows_the_player_position(self):
        state = json.loads(json.dumps(START))
        state["Player"]["Position"] = {"X": 84, "Y": 128}
        self.assertAlmostEqual(self.potential.value(state), VALIDATED_CELLS[(84, 128)], places=3)

    def test_the_terminal_potential_is_zero(self):
        """No player, or a player in another room: both are endings, and shaping needs them at 0."""
        self.assertEqual(self.potential.value(None), 0.0)
        elsewhere = json.loads(json.dumps(START))
        elsewhere["RoomName"] = "2"
        self.assertEqual(self.potential.value(elsewhere), 0.0)

    def test_matches_recognises_the_same_room_and_rejects_a_different_one(self):
        self.assertTrue(self.potential.matches(json.loads(json.dumps(START))))
        self.assertFalse(self.potential.matches(None))
        changed = json.loads(json.dumps(START))
        changed["SolidsData"] = changed["SolidsData"].replace("0", "1", 1)
        self.assertFalse(self.potential.matches(changed))


if __name__ == "__main__":
    unittest.main()
