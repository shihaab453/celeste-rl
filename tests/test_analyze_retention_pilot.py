"""Offline checks of the retention pilot analyzer's helpers."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_retention_pilot import explained_variance, retained_share  # noqa: E402


class RetainedShareTests(unittest.TestCase):
    def test_full_kept_back_at_floor_and_below_it(self):
        self.assertEqual(retained_share(0.76, 0.76, 0.2), 1.0)
        self.assertEqual(retained_share(0.2, 0.76, 0.2), 0.0)
        self.assertLess(retained_share(0.09, 0.769, 0.2046), 0)

    def test_matches_a_recorded_value(self):
        self.assertAlmostEqual(retained_share(0.2137, 0.7643, 0.142), 0.1152)


class ExplainedVarianceTests(unittest.TestCase):
    def test_skips_rows_without_an_update_and_keeps_the_first_ten(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        rows = ["accepted_steps,explained_variance", "2048,"] + [f"{2048 * (i + 2)},{i / 10}" for i in range(12)]
        (folder / "progress.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.assertEqual(explained_variance(folder), [i / 10 for i in range(10)])


if __name__ == "__main__":
    unittest.main()
