"""Check route weighting, same-dash distance and refusal of corrupted episode identities."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from describe_tiebreak_near_far import aggregate, distances, safe_path, validate_episodes


class NearFarTests(unittest.TestCase):
    def test_route_macro_does_not_weight_by_episode_count(self):
        rows = [{'route': 'a', 'ending': 'success'}] * 3 + [{'route': 'b', 'ending': 'death'}]
        result = aggregate(rows)
        self.assertEqual(result['success_rate'], .75)
        self.assertEqual(result['route_macro'], .5)
        self.assertIsNone(aggregate([])['route_macro'])

    def test_same_dash_distance_and_four_pixel_boundary(self):
        fitted = np.array([[0, 0, 0], [100, 0, 1]])
        starts = np.array([[4, 0, 0], [5, 0, 0], [0, 0, 1]])
        d, fallbacks = distances(fitted, starts)
        self.assertEqual(d.tolist(), [4, 5, 100])
        self.assertEqual((d <= 4).tolist(), [True, False, False])
        self.assertEqual(fallbacks, 0)

    def test_fresh_sets_and_external_paths_refuse_before_read(self):
        for name in ('config/heldout_starts-room1-v2.json', 'config/heldout_starts-room2-v2.json', '../elsewhere.json'):
            with self.assertRaises(ValueError):
                safe_path(name)

    def test_duplicate_state_cannot_replace_missing_state(self):
        entries, episodes = self.fixture()
        validate_episodes(episodes, entries, 'pin')
        with self.assertRaises(ValueError):
            validate_episodes([episodes[0], episodes[0]], entries, 'pin')

    def test_fault_repeat_checkpoint_or_metadata_mismatch_refuses(self):
        entries, episodes = self.fixture()
        for key, value in [('problem', 'fault'), ('repeat', 1), ('checkpoint_sha256', 'other'), ('route', 'other'), ('start_dashes', 1)]:
            corrupted = copy.deepcopy(episodes)
            corrupted[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_episodes(corrupted, entries, 'pin')

    @staticmethod
    def fixture():
        entries = [{'state_id': str(i), 'route': 'a', 'position': [19,144], 'dashes': 0, 'frames': 20, 'room': '1'} for i in range(2)]
        episodes = [{'state_id': e['state_id'], 'route': 'a', 'start_position': [19,144], 'start_dashes': 0, 'start_frames': 20,
                     'start_room': '1', 'repeat': 0, 'problem': None, 'checkpoint_sha256': 'pin', 'ending': 'success'} for e in entries]
        return entries, episodes


if __name__ == '__main__':
    unittest.main()
