"""Offline synthetic prefix replay; no Native, model, network or subprocess."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robofate_pipeline import collection


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.run = self.root / "new"
        self.run.mkdir()
        rows = [{"index": i, "native_count_after": i + 1, "actual_control_steps": 10 * (i + 1),
                 "role": "DM0.5", "action": {"synthetic_delta": 1},
                 "native_success": False, "terminated": False, "truncated": False} for i in range(2)]
        (self.source / "native_actions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows) + "POISON_FUTURE_SUFFIX\n")
        self.events, self.failed = [], []
        self.task = SimpleNamespace(control_steps=0, max_steps=4, done=False)
        self.recorder = SimpleNamespace(task=self.task, run=self.run,
                                        writer=SimpleNamespace(fail=self.failed.append))

    def execute(self, task, recorder, publish, action, **kwargs):
        self.events.append(("action", task.control_steps))
        task.control_steps += 1

    def gate(self, name):
        def check():
            self.events.append((name, self.task.control_steps))
        return check

    def replay(self, **overrides):
        args = dict(source_run=self.source, t1=2, budget=4, admission=self.gate("admission"),
                    owner_guard=contextlib.nullcontext, verify_initial=self.gate("initial"),
                    verify_restored=self.gate("restored"))
        args.update(overrides)
        with patch("robofate_pipeline.recording.common.execute_action", side_effect=self.execute), \
             patch("robofate_pipeline.recording.core.wait_for_native_publication"):
            collection.replay_prefix(self.task, self.recorder, lambda: None, **args)

    def test_only_prefix_is_read_before_restored_gate(self):
        self.replay()
        self.assertEqual(self.task.control_steps, 2)
        self.assertEqual([value for value in self.events if value[0] == "action"],
                         [("action", 0), ("action", 1)])
        self.assertEqual(self.events[-1], ("restored", 2))

    def test_initial_gate_failure_prevents_all_actions(self):
        def deny():
            raise RuntimeError("SYNTHETIC initial mismatch")
        with self.assertRaises(RuntimeError):
            self.replay(verify_initial=deny)
        self.assertEqual(self.task.control_steps, 0)
        self.assertEqual(len(self.failed), 1)

    def test_missing_restore_gate_is_not_an_approval(self):
        with self.assertRaises(ValueError):
            self.replay(verify_restored=None)
        self.assertEqual(self.task.control_steps, 0)

    def test_boolean_success_is_not_a_restore_proof(self):
        with self.assertRaises(ValueError):
            self.replay(verify_restored=lambda: True)
        self.assertEqual(len(self.failed), 1)

    def test_source_is_never_replay_destination(self):
        self.recorder.run = self.source
        with self.assertRaises(ValueError):
            self.replay()
        self.assertEqual(self.task.control_steps, 0)

    def test_budget_is_never_renewed(self):
        self.task.max_steps = 99
        with self.assertRaises(ValueError):
            self.replay()
        self.assertEqual(self.task.control_steps, 0)

    def test_changed_boundary_after_restore_gate_is_failure(self):
        def change():
            self.task.control_steps += 1
        with self.assertRaises(ValueError):
            self.replay(verify_restored=change)
        self.assertEqual(len(self.failed), 1)

    def test_unknown_action_outcome_stops_without_retry(self):
        def fail(*args, **kwargs):
            self.task.control_steps += 1
            raise RuntimeError("SYNTHETIC uncertain outcome")
        self.execute = fail
        with self.assertRaises(RuntimeError):
            self.replay()
        self.assertEqual(self.task.control_steps, 1)
        self.assertEqual(len(self.failed), 1)
