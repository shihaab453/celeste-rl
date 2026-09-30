"""Check signed success-point loss and cancellation of a shared baseline."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from describe_clone_near_far import loss_row


class CloneLossTests(unittest.TestCase):
    def test_improvement_is_negative_and_metrics_are_distinct(self):
        result = loss_row({'success_rate':.5,'route_macro':.6},{'success_rate':.7,'route_macro':.4})
        self.assertAlmostEqual(result['success_rate_loss_percentage_points'],-20)
        self.assertAlmostEqual(result['route_macro_loss_percentage_points'],20)

    def test_shared_baseline_cancels_in_between_arm_loss_comparison(self):
        for baseline in (.2,.8):
            a1 = loss_row({'success_rate':baseline,'route_macro':baseline},{'success_rate':.6,'route_macro':.65})
            e0 = loss_row({'success_rate':baseline,'route_macro':baseline},{'success_rate':.5,'route_macro':.55})
            self.assertAlmostEqual(e0['success_rate_loss_percentage_points']-a1['success_rate_loss_percentage_points'],10)
            self.assertAlmostEqual(e0['route_macro_loss_percentage_points']-a1['route_macro_loss_percentage_points'],10)


if __name__ == '__main__':
    unittest.main()
