"""Line-ending-safe hashes: a recorded hash keeps verifying when git changes a file's line endings."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from celeste_rl.texthash import matches_text_hash, text_hash_forms, text_sha256


class TextHashTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.lf, self.crlf = self.folder / "lf.json", self.folder / "crlf.json"
        self.lf.write_bytes(b'{\n  "a": 1\n}\n')
        self.crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')

    def test_the_text_hash_ignores_line_endings(self):
        self.assertEqual(text_sha256(self.lf), text_sha256(self.crlf))
        self.assertEqual(text_sha256(self.lf), hashlib.sha256(self.lf.read_bytes()).hexdigest())

    def test_a_hash_recorded_from_either_form_matches_either_checkout(self):
        for recorded_from in (self.lf, self.crlf):
            recorded = hashlib.sha256(recorded_from.read_bytes()).hexdigest()
            for checkout in (self.lf, self.crlf):
                self.assertTrue(matches_text_hash(checkout, recorded))

    def test_other_content_or_no_record_never_matches(self):
        other = self.folder / "other.json"
        other.write_bytes(b'{\n  "a": 2\n}\n')
        self.assertFalse(matches_text_hash(other, hashlib.sha256(self.lf.read_bytes()).hexdigest()))
        self.assertFalse(matches_text_hash(self.lf, None))
        self.assertEqual(len(text_hash_forms(self.lf)), 2)  # raw == LF form here, plus the CRLF form


if __name__ == "__main__":
    unittest.main()
