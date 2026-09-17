"""act-v1 action mapping: canonical lines, round trips, the mod's regex, disabled inputs.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import itertools
import random
import re
import unittest
from pathlib import Path

import numpy as np

from celeste_rl.actions import INDEX, apply_disabled, disabled_mask, parse_line, to_line
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS

NO_MASK = np.zeros(len(ACTION_INPUTS), dtype=bool)


def mod_pattern() -> re.Pattern:
    source = (Path(__file__).resolve().parents[1] / "mod" / "CelesteRLLockstep" / "Source" / "LockstepDriver.cs").read_text()
    return re.compile(re.search(r'InputLine = new\(@"(.+?)", RegexOptions', source)[1].replace(r"\z", r"\Z"))


class CanonicalLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pattern = mod_pattern()

    def check(self, vector: np.ndarray) -> str:
        line = to_line(apply_disabled(vector, NO_MASK))
        self.assertRegex(line, self.pattern)
        np.testing.assert_array_equal(parse_line(line, canonical_only=True), vector)
        return line

    def test_every_single_input(self):
        lines = set()
        for i, name in enumerate(ACTION_INPUTS):
            vector = np.zeros(len(ACTION_INPUTS), dtype=np.int8)
            vector[i] = 1
            line = self.check(vector)
            self.assertEqual(line, f"1,{name}")
            lines.add(line)
        self.assertEqual(len(lines), len(ACTION_INPUTS))
        self.assertEqual(to_line(np.zeros(len(ACTION_INPUTS), dtype=np.int8)), "1")

    def test_all_dash_only_and_move_only_combinations(self):
        rng = random.Random(0)
        lines = set()
        a_indices = [INDEX[f"A{d}"] for d in "LRUD"]
        m_indices = [INDEX[f"M{d}"] for d in "LRUD"]
        for a_bits, m_bits in itertools.product(itertools.product((0, 1), repeat=4), repeat=2):
            vector = np.zeros(len(ACTION_INPUTS), dtype=np.int8)
            vector[a_indices] = a_bits
            vector[m_indices] = m_bits
            lines.add(self.check(vector))
            # The same directions with random buttons held.
            vector[:16] = [rng.random() < 0.3 for _ in range(16)]
            self.check(vector)
        self.assertEqual(len(lines), 256)

    def test_random_vectors_are_one_to_one_with_canonical_lines(self):
        rng = np.random.default_rng(0)
        vectors = rng.integers(0, 2, size=(5000, len(ACTION_INPUTS)), dtype=np.int8)
        seen: dict[str, bytes] = {}
        for vector in vectors:
            line = self.check(vector)
            self.assertEqual(seen.setdefault(line, vector.tobytes()), vector.tobytes(), f"two vectors give {line}")

    def test_non_canonical_lines_are_canonicalised_or_rejected(self):
        np.testing.assert_array_equal(parse_line("1,R,R"), parse_line("1,R"))
        np.testing.assert_array_equal(parse_line("1,J,R"), parse_line("1,R,J"))
        np.testing.assert_array_equal(parse_line("1,AUL"), parse_line("1,ALU"))
        for line in ("1,R,R", "1,J,R", "1,AUL"):
            with self.assertRaisesRegex(ValueError, "Not canonical"):
                parse_line(line, canonical_only=True)
        for line in ("2,R", "1,F", "1,AX", "1,A", "R", "1,R,,J"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_line(line)


class DisabledInputTests(unittest.TestCase):
    def test_menu_inputs_are_forced_off_and_reported(self):
        mask = disabled_mask(MENU_INPUTS)
        self.assertEqual([ACTION_INPUTS[i] for i in np.flatnonzero(mask)], ["S", "Q", "N"])
        requested = np.ones(len(ACTION_INPUTS), dtype=np.int8)
        applied = apply_disabled(requested, mask)
        self.assertEqual(applied.dtype, np.int8)
        self.assertEqual(applied.sum(), len(ACTION_INPUTS) - 3)
        self.assertNotIn(",S", to_line(applied))
        self.assertNotIn(",Q", to_line(applied))
        self.assertNotIn(",N", to_line(applied))
        self.assertEqual(requested.sum(), len(ACTION_INPUTS), "the requested action must not be modified")

    def test_invalid_actions_and_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown input"):
            disabled_mask(["P"])
        with self.assertRaisesRegex(ValueError, "shape"):
            apply_disabled(np.zeros(23), NO_MASK)
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            apply_disabled(np.full(len(ACTION_INPUTS), 2), NO_MASK)


if __name__ == "__main__":
    unittest.main()
