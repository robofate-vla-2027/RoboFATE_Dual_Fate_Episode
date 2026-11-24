"""Original restore, persistence and readback on synthetic CPU-only fixtures.

Only temporary files and in-memory fake counters are changed. No sockets,
subprocesses, signals, real model calls, robot or simulator operations are used.
"""
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robofate_pipeline import evidence
from robofate_pipeline.persistence import CameraBudget, camera
from robofate_pipeline.persistence.files import Files
from robofate_pipeline.persistence.plain import plain
from robofate_pipeline.readback import settle, reader
from robofate_pipeline.restoration import RestoreBinding
from synthetic_production import Fixture, physical, observation, cameras, put


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.addCleanup(self.f.close)

    def compare(self, state=None):
        f = self.f
        return f.api.comparator.compare_states(
            physical(1), physical(1) if state is None else state,
            f.restore_preparation["profile"], context=f.context)

    def test_original_context_and_profile_linkage(self):
        f = self.f
        self.assertEqual((f.context.total, f.context.t1, f.context.remaining(1)), (4, 1, 3))
        changed = copy.deepcopy(f.restore_preparation["profile"])
        changed["thresholds"]["position_tol_m"] *= 10
        with self.assertRaises(ValueError):
            f.api.context.check_profile(f.context, changed)

    def test_equality_never_qualifies_hidden_state(self):
        result = self.compare()
        self.assertTrue(result["all_compared_fields_within_candidate_rules"])
        self.assertFalse(result["restore_qualified"])
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["comparison_status"], "indeterminate")

    def test_original_comparator_detects_physics_rng_and_missing_state(self):
        for mutate, expected in (
            (lambda s: s["objects"]["synthetic_object"]["world_pose"][0].__setitem__(0, .5), False),
            (lambda s: s["rng"]["synthetic_rng"].append(3), False),
            (lambda s: s["task"].__setitem__("native_count", [2]), False),
        ):
            state = physical(1)
            mutate(state)
            self.assertEqual(self.compare(state)["all_compared_fields_within_candidate_rules"], expected)
        state = physical(1)
        del state["objects"]["synthetic_object"]["sleeping"]
        result = self.compare(state)
        self.assertTrue(result["coverage"]["missing_or_unhandled_paths"])

    def test_quaternion_sign_invariance(self):
        result = self.f.api.comparator.quaternion_angle_rad([1, 0, 0, 0], [-1, 0, 0, 0])
        self.assertEqual(result, (0, 0, 0))

    def test_source_change_is_not_readmitted(self):
        f = self.f
        put(f.source / "result.json", {"native_success": True})
        with self.assertRaises(ValueError):
            f.binding.source_material(f.source_context_ref, f.manifest_ref, f.summary_ref, t1=1)

    def test_excluded_source_rejected(self):
        f = self.f
        policy = copy.deepcopy(f.policy)
        policy["excluded_source_ids"] = ["SYNTHETIC_SOURCE"]
        binding = RestoreBinding(put(f.config / "excluded-policy.json", policy))
        with self.assertRaises(ValueError):
            binding.source_material(f.source_context_ref, f.manifest_ref, f.summary_ref, t1=1)

    def test_consumed_source_rejected(self):
        f = self.f
        admission = evidence.document(f.admission_ref["path"])
        admission["sources"]["SYNTHETIC_SOURCE"]["consumed"] = True
        new_ref = put(f.config / "consumed-admission.json", admission)
        policy = dict(f.policy, admission_ref=new_ref)
        binding = RestoreBinding(put(f.config / "consumed-policy.json", policy))
        with self.assertRaises(ValueError):
            binding.source_material(f.source_context_ref, f.manifest_ref, f.summary_ref, t1=1)

    def test_synthetic_preparation_cannot_enable_execution(self):
        f = self.f
        with self.assertRaises(ValueError):
            f.binding.prepare_context(f.material, run=f.run, owner=f.identity, core_modules=f.modules, enabled=True)
        with self.assertRaises(RuntimeError):
            f.api.support.replay_prefix(f.task, f.recorder, f.run, lambda *_: None,
                                        f.restore_preparation["config"], put, context=f.context, main_execute=True)

    def test_original_prefix_parser_does_not_read_future(self):
        f = self.f
        first = (f.source / "native_actions.jsonl").read_bytes().splitlines(keepends=True)[0]
        stream = io.BytesIO(first + b"POISON_FUTURE_SUFFIX\n")
        self.assertEqual(len(f.api.support.prefix_rows(stream, 1, total=4)), 1)
        self.assertEqual(stream.readline(), b"POISON_FUTURE_SUFFIX\n")
        for bad in (b"{}\n", b'{"index":0,"index":1}\n', b'{"a":NaN}\n', first.rstrip(b"\n")):
            with self.assertRaises((ValueError, RuntimeError, KeyError)):
                f.api.support.prefix_rows(io.BytesIO(bad), 1, total=4)

    def test_original_recorded_prefix_revalidates_all_bindings(self):
        f = self.f
        result = f.api.support.recorded_prefix(f.source, f.restore_preparation["config"], context=f.context)
        self.assertEqual(result[0]["action"], {"delta": 1})
        f.task.native.step_lim = 5
        with self.assertRaises(ValueError):
            f.api.context.check_native(f.context, f.api.context.native_snapshot(f.task))

    def test_no_bool_as_native_budget(self):
        with self.assertRaises(ValueError):
            self.f.binding.budget("synthetic_task", True, 1)
        with self.assertRaises(ValueError):
            CameraBudget("synthetic_task", 4, True, 3, (("synthetic_task", 4),))
        with self.assertRaises(ValueError):
            CameraBudget("synthetic_task", 4, 0, 4, (("synthetic_task", 4),))

    def test_journal_preserves_raw_rgb_and_action_pair(self):
        f = self.f
        journal, reference = f.journal_action()
        self.assertIsNone(journal.pending)
        self.assertEqual(journal.sequence, 1)
        path, digest = reference.split("#sha256=")
        self.assertEqual(evidence.file_ref(path)["sha256"], digest)
        ipc = f.run / "teacher_ipc"
        for name in ("cam_head.png", "cam_left_wrist.png", "cam_right_wrist.png", "arrays.npz"):
            self.assertEqual((ipc / "artifacts/public_000001" / name).read_bytes(),
                             (f.run / "macro_obs_000002" / name).read_bytes())
        transition = evidence.document(path)
        self.assertEqual(transition["actual_control_range"], [11, 20])
        self.assertEqual(transition["actual_event"]["action"], {"delta": 2})

    def test_journal_rejects_unpaired_or_stale_proposal(self):
        f = self.f
        journal = f.journal()
        with self.assertRaises(ValueError):
            journal.committed(f.recorder, before={"native_used": 1}, after={"native_used": 2}, event={})
        with self.assertRaises(ValueError):
            journal.proposal(dict(kind="action", action={"delta": 1}, refate=dict(
                source_id="SYNTHETIC_SOURCE", sequence=2, native_before=1)))

    def test_journal_destination_is_exclusive(self):
        self.f.journal()
        with self.assertRaises(FileExistsError):
            self.f.journal()

    def test_camera_enrichment_is_public_and_does_not_change_rgb(self):
        f = self.f
        raw = observation(1)
        enriched = camera.enrich(raw, cameras()["cameras"], 1, budget=f.budget)
        self.assertNotIn("additional_info", raw)
        for c in camera.CAMERAS:
            before, after = raw["vision"][c]["color"], enriched["vision"][c]["color"]
            self.assertEqual(before.dtype, after.dtype)
            self.assertEqual(before.shape, after.shape)
            self.assertTrue(np.array_equal(before, after))
        with self.assertRaises(ValueError):
            camera.enrich(raw, cameras()["cameras"], 0, budget=f.budget)

    def test_plain_does_not_accept_object_arrays(self):
        with self.assertRaises(RuntimeError):
            plain(np.array([object()], dtype=object))

    def test_persistence_rejects_escape_links_and_overwrite(self):
        f = self.f
        journal = f.journal()
        files = journal.host
        for name in ("../escape", "/absolute"):
            with self.assertRaises(RuntimeError):
                files.path(name)
        (f.run / "teacher_ipc/linked").symlink_to(f.source, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            files.path("linked/result.json")
        files.publish("response_000001.json", {"data_kind": "synthetic"})
        with self.assertRaises(FileExistsError):
            files.publish("response_000001.json", {"data_kind": "synthetic"})
        with self.assertRaises(RuntimeError):
            Files(f.run / "teacher_ipc", authorized_root=f.source)

    def test_readback_decodes_complete_synthetic_branch_without_credit(self):
        refs = self.f.prepare_readback()
        summary = settle(*refs)
        self.assertEqual(summary["data_status"], "synthetic_validation_only")
        self.assertFalse(summary["qualified_positive"])
        self.assertEqual(summary["new_positive"], 0)
        self.assertIsNone(summary["publication"])
        report = evidence.document(summary["readback_ref"]["path"])
        self.assertEqual(report["decoded_teacher_observations"], 3)
        self.assertEqual(report["decoded_teacher_RGB_images"], 9)
        self.assertEqual(len(report["diagnostics"]["constant_rgb_frames"]), 9)
        self.assertTrue(all(v["status"] == "passed" for v in report["conditions"].values()))
        self.assertEqual(settle(*refs), summary)
        self.assertFalse((self.f.data / "published").exists())

    def test_failed_readback_cannot_blindly_retry(self):
        f = self.f
        refs = f.prepare_readback()
        (f.run / "macro_obs_000002/cam_head.png").unlink()
        with self.assertRaises((OSError, ValueError)):
            settle(*refs)
        self.assertTrue((f.run / "teacher_readback_001/FAILED.json").is_file())
        with self.assertRaisesRegex(ValueError, "no blind retry"):
            settle(*refs)

    def test_changing_previously_reviewed_source_invalidates_reuse(self):
        f = self.f
        refs = f.prepare_readback()
        settle(*refs)
        put(f.run / "physics_000002.json", {"data_kind": "synthetic", "changed": True})
        with self.assertRaises(ValueError):
            settle(*refs)

    def test_non_success_is_not_qualified(self):
        f = self.f
        refs = f.prepare_readback()
        result = evidence.document(f.run / "result.json")
        result["native_success"] = False
        put(f.run / "result.json", result)
        summary = settle(*refs)
        self.assertFalse(summary["qualified_positive"])
        self.assertFalse(summary["complete_positive_reader_run"])
        self.assertEqual(summary["new_positive"], 0)

    def test_original_reader_rejects_budget_gap_wrong_teacher_action(self):
        f = self.f
        refs = f.prepare_readback()
        path = f.run / "teacher_ipc/response_000001.json"
        value = evidence.document(path)
        value["action"] = {"delta": 99}
        put(path, value)
        with self.assertRaises(ValueError):
            settle(*refs)

    def test_evidence_pins_detect_changes_and_duplicate_json(self):
        f = self.f
        pin = put(f.config / "evidence.json", {"data_kind": "synthetic"})
        Path(pin["path"]).write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):
            evidence.read_pin(pin)
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                evidence.decode(raw)

    def test_continuation_retains_synthetic_execution_hold(self):
        f = self.f
        continuation = f.initialized_continuation()
        with self.assertRaises(RuntimeError):
            continuation.restore()
        self.assertEqual(f.task.control_steps, 0)
        self.assertEqual(f.events, [])
        self.assertTrue(continuation.owner.failed)
        self.assertIsNotNone(f.writer.first_error)

    def restored_synthetic_continuation(self):
        f = self.f
        value = f.initialized_continuation()
        # Only the execution-enable gate is mocked for in-memory synthetic counters.
        # Original profile, counter, prefix, state, auxiliary and RGB checks execute.
        with patch.object(value.api.support, "bound_run", return_value=value.context):
            value.restore()
        return value

    def test_original_replay_and_all_comparisons_are_composed(self):
        f = self.f
        continuation = self.restored_synthetic_continuation()
        self.assertEqual(f.task.control_steps, 1)
        self.assertEqual(len(f.events), 1)
        self.assertEqual(f.events[0]["role"], "recorded_DM0.5_prefix_replay")
        self.assertEqual(set(continuation.proof), {"state", "observation", "auxiliary", "boundary"})
        self.assertFalse(evidence.document(continuation.proof["boundary"]["path"])["restore_qualified"])
        with self.assertRaises(ValueError):
            continuation.restore()

    def test_continuation_binds_original_journal_before_teacher_takeover(self):
        from robofate_pipeline.recording.predict import ExistingEnvTeacher
        from robofate_pipeline.recording import takeover
        f = self.f
        continuation = self.restored_synthetic_continuation()
        fence = dict(f.identity, dm_submissions_disabled=True, dm_inflight=0, dm_queue_discarded=True)
        continuation.owner.snapshot = lambda: copy.deepcopy(fence)
        config = dict(schema="robofate.runtime.v1", identity=f.identity, source_id="SYNTHETIC_SOURCE",
                      task="synthetic_task", native_limit=4, t1=1, actor_wall_s=3600,
                      run_dir=str(f.data / "sessions/SYNTHETIC_SESSION"))
        calls = []
        model = SimpleNamespace(worker=None, connection=None,
                                cfg={"runtime_binding": config, "refate_takeover": dict(
                                    fence, native_used=1, native_limit=4, native_terminal=False)},
                                takeover=lambda obs, history, remaining: calls.append((len(history), remaining)),
                                get_takeover_request=lambda: None)
        runtime = SimpleNamespace(binding=config, ExistingEnvTeacher=ExistingEnvTeacher,
                                  takeover=SimpleNamespace(
                                      policy_observation=lambda obs: {k: v for k, v in obs.items() if k in takeover.PUBLIC_KEYS}))
        teacher = continuation.bind_teacher(runtime, model)
        self.assertTrue(teacher.bound)
        self.assertEqual(calls, [(2, 3)])
        self.assertTrue((f.run / "teacher_ipc/bootstrap.json").is_file())
        self.assertIs(teacher._common_teacher["committed_dense_ref"].__self__, continuation.journal)

    def test_original_publisher_exclusive_transaction_on_labeled_synthetic_report(self):
        from robofate_pipeline.readback.settlement import publisher
        f = self.f
        refs = f.prepare_readback()
        summary = settle(*refs)
        publish = publisher(context=evidence.document(refs[0]["path"]), config=evidence.document(refs[1]["path"]))
        first = publish(summary["readback_ref"])
        second = publish(summary["readback_ref"])
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(first["manifest"], second["manifest"])
        manifest = evidence.document(first["manifest"]["path"])
        self.assertTrue(manifest["synthetic_fixture"])
        self.assertFalse(manifest["counted_in_quota"])
        self.assertFalse(manifest["ledger_registration_done"])

    def test_original_publisher_rejects_partial_slot(self):
        from robofate_pipeline.readback.settlement import publisher
        f = self.f
        refs = f.prepare_readback()
        summary = settle(*refs)
        (f.data / "published/synthetic_task/SYNTHETIC_SOURCE").mkdir(parents=True)
        publish = publisher(context=evidence.document(refs[0]["path"]), config=evidence.document(refs[1]["path"]))
        with self.assertRaisesRegex(ValueError, "incomplete existing publication"):
            publish(summary["readback_ref"])
