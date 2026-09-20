"""The route search converts world positions to room-local tile coordinates."""
from __future__ import annotations

import unittest

from scripts.find_room_exit import player_tile


class PlayerTileTests(unittest.TestCase):
    def test_nonzero_room_bounds_are_removed_from_the_world_position(self) -> None:
        state = {
            "Player": {"Position": {"X": 261, "Y": 1}},
            "Level": {"Bounds": {"X": 240, "Y": -184, "W": 320, "H": 180}},
        }

        self.assertEqual(player_tile(state), (2, 22))


if __name__ == "__main__":
    unittest.main()
