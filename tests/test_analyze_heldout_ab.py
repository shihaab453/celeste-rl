"""Offline tests for the predeclared paired held-out analysis."""
from __future__ import annotations

import unittest

from scripts.analyze_heldout_ab import analyse, crossed_cluster_interval, exact_sign_flip_pvalue, same_repo_path


class PairedInferenceTests(unittest.TestCase):
    def test_repository_paths_compare_by_identity_across_slash_styles(self) -> None:
        self.assertTrue(same_repo_path("runs/train/checkpoint.zip", r"runs\train\checkpoint.zip"))

    def test_six_differences_all_in_one_direction_reach_exact_minimum(self) -> None:
        self.assertEqual(exact_sign_flip_pvalue([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 0.03125)

    def test_crossed_bootstrap_resamples_seed_pairs_and_routes_together(self) -> None:
        lower, upper = crossed_cluster_interval([[0.2, 0.2], [0.2, 0.2]], samples=100, seed=7)

        self.assertAlmostEqual(lower, 0.2)
        self.assertAlmostEqual(upper, 0.2)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def rows(outcomes: dict[str, str]) -> list[dict]:
        return [{"route": route, "ending": ending} for route, ending in outcomes.items()]

    def test_primary_result_equally_weights_matched_seeds_and_routes(self) -> None:
        run_rows = {
            (1, "A"): self.rows({"r1": "death", "r2": "death"}),
            (1, "B"): self.rows({"r1": "success", "r2": "success"}),
            (2, "A"): self.rows({"r1": "success", "r2": "death"}),
            (2, "B"): self.rows({"r1": "success", "r2": "success"}),
        }

        result = analyse(run_rows, [1, 2], ["r1", "r2"], bootstrap_samples=100, bootstrap_seed=3)

        self.assertEqual(result["primary"]["A_route_macro_success_rate"], 0.25)
        self.assertEqual(result["primary"]["B_route_macro_success_rate"], 1.0)
        self.assertEqual(result["primary"]["B_minus_A"], 0.75)
        self.assertEqual(result["primary"]["pair_differences"], {"1": 1.0, "2": 0.5})
        self.assertEqual(result["secondary"]["arms"]["A"]["successes"], 1)
        self.assertEqual(result["secondary"]["arms"]["B"]["successes"], 4)


if __name__ == "__main__":
    unittest.main()
