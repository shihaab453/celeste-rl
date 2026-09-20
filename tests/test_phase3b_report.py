"""The published Phase 3B numeric tables are generated from the committed analysis data."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.render_heldout_ab_markdown import primary_table, route_table, seed_table

REPO = Path(__file__).resolve().parents[1]


def generated_block(document: str, name: str) -> str:
    begin = f"<!-- BEGIN GENERATED {name} -->"
    end = f"<!-- END GENERATED {name} -->"
    return document.split(begin, 1)[1].split(end, 1)[0].strip()


class Phase3BReportTests(unittest.TestCase):
    def test_numeric_tables_match_the_machine_readable_analysis(self) -> None:
        analysis = json.loads(
            (REPO / "docs" / "results" / "phase3b-heldout-ab-200.json").read_text(encoding="utf-8")
        )
        report = (REPO / "docs" / "phase3b-demonstration-comparison.md").read_text(encoding="utf-8")

        self.assertEqual(generated_block(report, "PRIMARY TABLE"), primary_table(analysis))
        self.assertEqual(generated_block(report, "SEED TABLE"), seed_table(analysis))
        self.assertEqual(generated_block(report, "ROUTE TABLE"), route_table(analysis))


if __name__ == "__main__":
    unittest.main()
