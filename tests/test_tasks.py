"""Room-task definitions are complete, hash-pinned, and converted into replayable starts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from celeste_rl.tasks import (
    TaskDefinitionError,
    canonical_task_identity,
    load_task_definition,
    manifest_task_identity,
    require_task_identity,
    task_identity,
)


class TaskDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.route = self.root / "route.json"
        self.route.write_text(json.dumps({
            "start_room": "1",
            "next_room": "2",
            "transition_step": 1,
            "actions": ["R", "UX", ""],
        }), encoding="utf-8")

    def definition(self, **changes) -> Path:
        data = {
            "format_version": 1,
            "name": "room-2",
            "start_room": "2",
            "target_room": "3",
            "source_route": "route.json",
            "source_route_sha256": hashlib.sha256(self.route.read_bytes()).hexdigest(),
            "prefix_frames": 2,
            "position": [261, 1],
            "dashes": 0,
        }
        data.update(changes)
        path = self.root / "task.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def load(self, path: Path):
        with patch("celeste_rl.tasks.REPO", self.root):
            return load_task_definition(path)

    def test_route_prefix_becomes_a_canonical_replay_start(self) -> None:
        definition = self.load(self.definition())

        self.assertEqual((definition.task.start_room, definition.task.target_room), ("2", "3"))
        self.assertEqual(definition.start.lines, ("1,R", "1,U,X"))
        self.assertEqual((definition.start.position, definition.start.room, definition.start.dashes),
                         ((261, 1), "2", 0))
        identity = task_identity(definition)
        self.assertEqual(identity["task_definition"], "task.json")
        self.assertEqual(identity["source_route_sha256"], hashlib.sha256(self.route.read_bytes()).hexdigest())
        self.assertEqual((identity["start_room"], identity["target_room"]), ("2", "3"))

    def test_source_route_hash_is_enforced(self) -> None:
        path = self.definition()
        self.route.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(TaskDefinitionError, "SHA-256"):
            self.load(path)

    def test_prefix_must_end_on_the_first_target_room_frame(self) -> None:
        with self.assertRaisesRegex(TaskDefinitionError, "first frame"):
            self.load(self.definition(prefix_frames=3))

    def test_a_later_room_cannot_omit_its_source_route(self) -> None:
        path = self.definition(source_route=None)

        with self.assertRaisesRegex(TaskDefinitionError, "needs a source route"):
            self.load(path)

    def test_a_legacy_manifest_is_room_1_and_refuses_a_later_task(self) -> None:
        definition = self.load(self.definition())

        legacy = manifest_task_identity({})

        self.assertEqual((legacy["start_room"], legacy["target_room"]), ("1", "2"))
        with self.assertRaisesRegex(TaskDefinitionError, "task identity does not match"):
            require_task_identity({}, task_identity(definition), "manifest")

    def test_later_room_identity_requires_both_hash_pins(self) -> None:
        definition = self.load(self.definition())
        identity = task_identity(definition)

        with self.assertRaisesRegex(TaskDefinitionError, "source-route SHA-256"):
            canonical_task_identity({**identity, "source_route_sha256": None})
        with self.assertRaisesRegex(TaskDefinitionError, "task definition"):
            canonical_task_identity({**identity, "task_definition": None,
                                     "task_definition_sha256": None})


if __name__ == "__main__":
    unittest.main()
