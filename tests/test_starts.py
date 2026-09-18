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

from celeste_rl.starts import MINIMUM_ATTEMPTS, OUTCOME_WINDOW, Start, StartArchive, cell_of


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


class ReverseCurriculumTests(unittest.TestCase):
    """`sampling="success"`: weight cells by how often the agent clears them, so the band it is learning moves
    backwards from the exit without anything measuring distance to the exit."""

    def archive(self, cells=4) -> StartArchive:
        archive = StartArchive(canonical_fraction=0.0, seed=0, sampling="success")
        for i in range(cells):
            archive.offer(start(10 + i, 80 + 8 * i, 120))
        return archive

    def test_an_unmeasured_cell_reads_as_half(self):
        """So the frontier is tried first rather than last."""
        archive = self.archive()
        self.assertEqual(archive.success_rate((99, 99)), 0.5)
        self.assertAlmostEqual(archive.weight((99, 99)), 0.25)

    def test_mastered_and_hopeless_cells_are_both_down_weighted(self):
        archive = self.archive()
        mastered, hopeless, learning = start(10, 80, 120), start(11, 88, 120), start(12, 96, 120)
        for _ in range(OUTCOME_WINDOW):
            archive.record_outcome(mastered, True)
            archive.record_outcome(hopeless, False)
        for i in range(OUTCOME_WINDOW):
            archive.record_outcome(learning, i % 2 == 0)
        weights = {name: archive.weight(cell_of(s.position))
                   for name, s in (("mastered", mastered), ("hopeless", hopeless), ("learning", learning))}
        self.assertGreater(weights["learning"], 3 * weights["mastered"])
        self.assertGreater(weights["learning"], 3 * weights["hopeless"])
        self.assertGreaterEqual(min(weights.values()), 0.01, "no cell should be excluded outright")

    def test_the_curriculum_moves_backwards_as_cells_are_mastered(self):
        """The behaviour the whole change exists for: once the exit-side cells are reliable, sampling shifts
        to the ones behind them."""
        archive = self.archive()
        near_exit = [start(10, 80, 120), start(11, 88, 120)]
        for s in near_exit:
            for _ in range(OUTCOME_WINDOW):
                archive.record_outcome(s, True)
        def mastered_share(a):
            drawn = Counter(cell_of(s.position) for s in (a.sample() for _ in range(4000)) if s)
            return sum(drawn[cell_of(s.position)] for s in near_exit) / sum(drawn.values())

        # Coverage sampling would keep drawing them: four cells, so half the draws.
        coverage = StartArchive(canonical_fraction=0.0, seed=0)
        for i in range(4):
            coverage.offer(start(10 + i, 80 + 8 * i, 120))
        self.assertAlmostEqual(mastered_share(coverage), 0.5, delta=0.05)
        self.assertLess(mastered_share(archive), 0.2, "mastered cells should fall away under the curriculum")

    def test_only_recent_attempts_count(self):
        archive = self.archive()
        s = start(10, 80, 120)
        for _ in range(OUTCOME_WINDOW):
            archive.record_outcome(s, False)
        for _ in range(OUTCOME_WINDOW):
            archive.record_outcome(s, True)
        self.assertGreater(archive.success_rate(cell_of(s.position)), 0.9,
                           "a window of 20 should have forgotten the early failures")

    def test_coverage_sampling_is_unchanged_and_is_the_default(self):
        self.assertEqual(StartArchive().sampling, "coverage")
        with self.assertRaises(ValueError):
            StartArchive(sampling="whatever")

    def test_sampling_does_not_count_a_use(self):
        """A stale entry that never replays must not be recorded as having been tried."""
        archive = self.archive()
        for _ in range(20):
            archive.sample()
        self.assertEqual(archive.uses, {})
        archive.record_use(start(10, 80, 120))
        self.assertEqual(archive.uses, {cell_of((80, 120)): 1})

    def test_outcomes_survive_a_save_and_load(self):
        archive = self.archive()
        s = start(10, 80, 120)
        for i in range(5):
            archive.record_outcome(s, i % 2 == 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive.save(path)
            loaded = StartArchive.load(path, seed=0)
        self.assertEqual(loaded.sampling, "success")
        self.assertEqual(list(loaded.outcomes[cell_of((80, 120))]), [True, False, True, False, True])
        self.assertEqual(loaded.success_rate(cell_of((80, 120))), archive.success_rate(cell_of((80, 120))))

    def test_a_barely_tried_cell_is_not_counted_as_a_learning_cell(self):
        """The metric flaw found in the curriculum run: one attempt and no successes estimates 0.33 and would
        have been reported as a cell halfway to being learned."""
        archive = self.archive()
        archive.record_outcome(start(10, 80, 120), False)
        report = archive.coverage()
        self.assertEqual(report["cells_measured"], 1)
        self.assertEqual((report["cells_tried_enough"], report["cells_learning"]), (0, 0))
        for _ in range(MINIMUM_ATTEMPTS):
            archive.record_outcome(start(10, 80, 120), True)
        self.assertEqual(archive.coverage()["cells_tried_enough"], 1)

    def test_sampling_still_favours_untried_cells(self):
        """The minimum applies to the reported bands, not to sampling: an untried cell should be tried."""
        archive = self.archive()
        self.assertAlmostEqual(archive.weight((99, 99)), 0.25)

    def test_coverage_reports_where_the_learning_band_is(self):
        archive = self.archive()
        for _ in range(OUTCOME_WINDOW):
            archive.record_outcome(start(10, 80, 120), True)
        for i in range(OUTCOME_WINDOW):
            archive.record_outcome(start(12, 96, 120), i % 2 == 0)
        report = archive.coverage()
        self.assertEqual((report["cells_mastered"], report["cells_learning"]), (1, 1))
        self.assertEqual(report["nearest_x_mastered"], 80)
        self.assertEqual(report["sampling"], "success")


if __name__ == "__main__":
    unittest.main()
