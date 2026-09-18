"""The start archive: what it keeps, what it refuses, and how it chooses (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from celeste_rl.starts import Start, StartArchive, cell_of


def start(frames: int, x: float, y: float, room: str = "1") -> Start:
    return Start(tuple("1,R" for _ in range(frames)), (x, y), room, dashes=1)


class CellTests(unittest.TestCase):
    def test_positions_in_the_same_tile_share_a_cell(self):
        self.assertEqual(cell_of((19, 144)), cell_of((23, 147)))
        self.assertNotEqual(cell_of((19, 144)), cell_of((27, 144)))


class ArchiveTests(unittest.TestCase):
    def test_keeps_the_shortest_prefix_for_a_cell(self):
        archive = StartArchive(seed=0)
        self.assertTrue(archive.offer(start(100, 80, 120)))
        self.assertTrue(archive.offer(start(40, 82, 122)), "a shorter route to the same cell should replace it")
        self.assertFalse(archive.offer(start(60, 81, 121)), "a longer route to the same cell should be refused")
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.starts[cell_of((80, 120))].frames, 40)
        self.assertEqual((archive.offered, archive.replaced), (3, 1))

    def test_would_keep_agrees_with_offer(self):
        """The environment asks would_keep every step, so it must never disagree with what offer does."""
        archive = StartArchive(max_frames=600, seed=0)
        cases = [(100, 80, 120), (40, 82, 122), (60, 81, 121), (10, 140, 96), (900, 200, 60)]
        for frames, x, y in cases:
            with self.subTest(frames=frames, position=(x, y)):
                predicted = archive.would_keep((x, y), frames)
                self.assertEqual(predicted, archive.offer(start(frames, x, y)))

    def test_different_cells_are_kept_separately(self):
        archive = StartArchive(seed=0)
        archive.offer(start(10, 80, 120))
        archive.offer(start(10, 140, 96))
        self.assertEqual(len(archive), 2)

    def test_a_prefix_longer_than_the_limit_is_refused(self):
        """A start that costs most of the 30 second deadline to reach leaves no episode to learn from."""
        archive = StartArchive(max_frames=600, seed=0)
        self.assertFalse(archive.offer(start(900, 80, 120)))
        self.assertEqual(len(archive), 0)

    def test_an_empty_archive_always_gives_the_canonical_start(self):
        self.assertIsNone(StartArchive(seed=0).sample())

    def test_the_canonical_fraction_is_respected(self):
        archive = StartArchive(canonical_fraction=0.25, seed=0)
        for i in range(8):
            archive.offer(start(10, 80 + 8 * i, 120))
        draws = [archive.sample() for _ in range(4000)]
        canonical = sum(1 for d in draws if d is None) / len(draws)
        self.assertAlmostEqual(canonical, 0.25, delta=0.03)

    def test_rarely_used_cells_are_preferred(self):
        """Inverse use count, so the archive spreads rather than re-running whatever came first."""
        archive = StartArchive(canonical_fraction=0.0, seed=0)
        for i in range(4):
            archive.offer(start(10, 80 + 8 * i, 120))
        counts = Counter(cell_of(s.position) for s in (archive.sample() for _ in range(2000)) if s)
        self.assertEqual(len(counts), 4, "every cell should be reachable by sampling")
        self.assertLess(max(counts.values()) - min(counts.values()), 0.2 * max(counts.values()),
                        f"use counts should even out, got {counts}")

    def test_a_canonical_fraction_of_one_never_uses_the_archive(self):
        archive = StartArchive(canonical_fraction=1.0, seed=0)
        archive.offer(start(10, 80, 120))
        self.assertTrue(all(archive.sample() is None for _ in range(50)))

    def test_an_impossible_canonical_fraction_is_refused(self):
        for value in (-0.1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StartArchive(canonical_fraction=value)

    def test_coverage_describes_what_is_held(self):
        archive = StartArchive(seed=0)
        self.assertIsNone(archive.coverage()["prefix_frames"])
        archive.offer(start(10, 80, 120))
        archive.offer(start(30, 140, 96))
        archive.sample()
        coverage = archive.coverage()
        self.assertEqual((coverage["cells"], coverage["furthest_x"], coverage["highest_y"]), (2, 140, 96))
        self.assertEqual(coverage["prefix_frames"]["min"], 10)

    def test_a_saved_archive_reloads_identically(self):
        archive = StartArchive(cell_size=8, canonical_fraction=0.3, max_frames=600, seed=0)
        archive.offer(start(10, 80, 120))
        archive.offer(start(30, 140, 96))
        for _ in range(10):
            archive.sample()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive.save(path)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [], "the temporary file should be renamed away")
            loaded = StartArchive.load(path, seed=0)
        self.assertEqual(loaded.starts, archive.starts)
        self.assertEqual(loaded.uses, archive.uses)
        self.assertEqual((loaded.cell_size, loaded.canonical_fraction, loaded.max_frames), (8, 0.3, 600))

    def test_a_start_round_trips_through_json(self):
        original = start(3, 80, 120)
        self.assertEqual(Start.from_json(json.loads(json.dumps(original.to_json()))), original)


if __name__ == "__main__":
    unittest.main()
