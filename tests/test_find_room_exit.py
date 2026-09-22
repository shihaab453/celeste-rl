"""The route search converts world positions to room-local tile coordinates."""
from __future__ import annotations

import unittest

from types import SimpleNamespace

from scripts.find_room_exit import live_in_room, player_tile, tile_map


class PlayerTileTests(unittest.TestCase):
    def test_nonzero_room_bounds_are_removed_from_the_world_position(self) -> None:
        state = {
            "Player": {"Position": {"X": 261, "Y": 1}},
            "Level": {"Bounds": {"X": 240, "Y": -184, "W": 320, "H": 180}},
        }

        self.assertEqual(player_tile(state), (2, 22))

    def test_spike_bounds_are_converted_to_room_local_tiles(self) -> None:
        state = {
            "SolidsData": "\n".join(["0" * 40] * 23),
            "Spikes": [{"Bounds": {"X": 280, "Y": -16, "W": 8, "H": 3}}],
            "Level": {"Bounds": {"X": 240, "Y": -184, "W": 320, "H": 184}},
        }

        _, blocked = tile_map(state)

        self.assertEqual(blocked, {(5, 21)})

    def test_a_continuation_must_remain_in_the_room_it_entered(self) -> None:
        self.assertTrue(live_in_room(SimpleNamespace(state={}, room="3"), "3"))
        self.assertFalse(live_in_room(SimpleNamespace(state={}, room="4"), "3"))
        self.assertFalse(live_in_room(SimpleNamespace(state=None, room="3"), "3"))


if __name__ == "__main__":
    unittest.main()
