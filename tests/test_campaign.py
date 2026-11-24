"""Synthetic CPU-only sources, ledger/CAS and composition tests.

Only temporary files and locks are real. No network, processes, signals, model,
simulator or GPU calls; no fixture is a production admission or scientific result.
"""
import copy
from contextlib import ExitStack
import datetime
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robofate_pipeline import evidence
from robofate_pipeline.baseline import frozen_impl, runtime as baseline_runtime, reader as baseline_reader
from robofate_pipeline.baseline import foreground as bp
from robofate_pipeline import foreground as owned
from robofate_pipeline.campaign.ledger import now
from robofate_pipeline.campaign.runner import Campaign
from robofate_pipeline.campaign.store import Store
from robofate_pipeline.campaign.family import family
from robofate_pipeline.runtime_policy import RuntimePolicy
from synthetic_campaign import CampaignFixture
from synthetic_baseline import BaselineFixture


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.f = CampaignFixture()
        self.addCleanup(self.f.close)

    def test_source_preparation_budget_raw_family_and_order(self):
        prep = self.f.catalog.prepare("synthetic_task", 0, self.f.split_ref)
        self.assertEqual(prep["native_step_limit"], 4)
        self.assertEqual(prep["source"]["required_accounted_prefix"], [])
        self.assertFalse(prep["source_admitted"])
        self.f.catalog.validate(prep)
        wrong = copy.deepcopy(prep)
        wrong["source"]["encoded_seed"] = 1
        with self.assertRaises(ValueError):
            self.f.catalog.validate(wrong)

    def test_task_budget_pin_and_bool_rejected(self):
        task = Path(self.f.catalog.tasks["synthetic_task"]["task_ref"]["path"])
        task.write_text(task.read_text().replace("= 4", "= True"))
        with self.assertRaises(ValueError):
            self.f.catalog.prepare("synthetic_task", 0, self.f.split_ref)

    def test_pose_only_family_normalization_preserves_unknown_fields(self):
        a = evidence.document(self.f.layout_refs[0]["path"])
        b = evidence.document(self.f.layout_refs[1]["path"])
        self.assertEqual(family(a), family(b))
        b["Rigid"]["synthetic"][0]["SYNTHETIC_unknown_field"] = 8
        self.assertNotEqual(family(a), family(b))

    def test_append_is_contiguous_pinned_and_outcome_blind(self):
        ref = self.f.catalog.append("synthetic_task", self.f.split_ref, self.f.layout_refs[2], now())
        split = self.f.catalog.lineage(ref, "synthetic_task")
        self.assertEqual(split["initial_source_sequence"], [0, 1, 2])
        self.assertEqual(split["layouts"]["0"], self.f.split["layouts"]["0"])
        self.assertFalse(split["collection_append"]["outcome_used"])
        self.f.catalog.prepare("synthetic_task", 2, ref)
        with self.assertRaises(ValueError):
            self.f.catalog.append("synthetic_task", ref, self.f.layout_refs[2], now())

    def test_reserved_initial_prefix_cannot_be_skipped(self):
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.f.reserve(ordinal=1)

    def test_reserve_start_and_second_launch_rejection(self):
        ref, c = self.f.reserve()
        d, _ = self.f.ledger.store.snapshot()
        n = d["nodes"]["SYNTHETIC_ALIAS"]
        self.assertTrue(d["source_index"][c["source_id"]]["source_started"])
        self.assertEqual(n["active_real_simulators"], 1)
        self.assertEqual(n["baseline_accounting"]["baseline_attempts_running"], 1)
        self.assertEqual(d["qualified_count"], 0)
        with self.assertRaisesRegex(ValueError, "active"):
            self.f.ledger.select_task("synthetic_task", "SYNTHETIC_SECOND")

    def test_original_baseline_failure_settlement_no_quota(self):
        ref, c = self.f.reserve()
        api, before, e, (out, receipt) = self.f.pure_settlement(c)
        self.assertEqual(receipt["negative_delta"], 0)
        self.assertTrue(out["source_index"][c["source_id"]]["reusable_DM_failure"])
        self.assertEqual(out["nodes"]["SYNTHETIC_ALIAS"]["active_real_simulators"], 0)
        reused, again = api.settle(out, e, now())
        self.assertEqual(reused, out)
        self.assertTrue(again["already_settled"])
        e["new_negative"] = 1
        with self.assertRaises(ValueError):
            api.settle(before, e, now())

    def test_original_success_negative_not_teacher_and_cleanup_separate(self):
        ref, c = self.f.reserve()
        _, before, e, (out, receipt) = self.f.pure_settlement(c, success=True, safe=False)
        self.assertEqual(receipt["negative_delta"], 1)
        self.assertFalse(out["source_index"][c["source_id"]]["reusable_DM_failure"])
        self.assertEqual(out["nodes"]["SYNTHETIC_ALIAS"]["active_real_simulators"], 1)
        self.assertEqual(out["negative_count"], 1)
        self.assertEqual(out["qualified_count"], 0)

    def test_canonical_recount_rejects_both_fates(self):
        ref, c = self.f.reserve()
        api, d, _, _ = self.f.pure_settlement(c)
        d["source_index"][c["source_id"]].update(counted_in_quota=True, counted_in_negative_quota=True)
        with self.assertRaises(ValueError):
            api.export_current_counts(d, "synthetic_task")

    def test_actual_wrapper_rejects_synthetic_settlement(self):
        ref, c = self.f.reserve()
        summary = evidence.exclusive(self.f.metadata / "no-authority.json", {"data_kind": "synthetic"})
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            self.f.ledger.settle_baseline(ref, summary, safe=True)

    def test_metadata_transaction_idempotence_and_request_conflict(self):
        store = self.f.ledger.store
        def change(d, inode):
            d["revision"] += 1
            return d, {"synthetic_value": 7}
        self.assertEqual(store.transact("SYNTHETIC", {"v": 1}, change), {"synthetic_value": 7})
        never = Mock(side_effect=AssertionError("must not recompute"))
        self.assertEqual(store.transact("SYNTHETIC", {"v": 1}, never), {"synthetic_value": 7})
        never.assert_not_called()
        with self.assertRaises(ValueError):
            store.transact("SYNTHETIC", {"v": 2}, never)

    def test_interrupted_metadata_commit_reconciles_not_recomputes(self):
        store = self.f.ledger.store
        def change(d, inode):
            d["revision"] += 1
            return d, {"synthetic_value": 9}
        with patch.object(store, "_replace", side_effect=OSError("SYNTHETIC disk fault")):
            with self.assertRaises(OSError):
                store.transact("SYNTHETIC_INTERRUPTED", {}, change)
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            store.snapshot()
        self.assertEqual(store.reconcile("SYNTHETIC_INTERRUPTED"), {"synthetic_value": 9})
        self.assertEqual(store.snapshot()[0]["revision"], 1)

    def test_reconcile_rejects_diverged_ledger(self):
        store = self.f.ledger.store
        with patch.object(store, "_replace", side_effect=OSError("SYNTHETIC")):
            with self.assertRaises(OSError):
                store.transact("SYNTHETIC_DIVERGED", {}, lambda d, i: (dict(d, revision=1), {}))
        value = evidence.document(store.state)
        value["revision"] = 3
        store.state.write_text(evidence.canonical(value))
        with self.assertRaisesRegex(ValueError, "diverged"):
            store.reconcile("SYNTHETIC_DIVERGED")

    def test_campaign_clocks_are_preserved(self):
        with self.assertRaisesRegex(ValueError, "clock"):
            self.f.ledger.store.transact("SYNTHETIC_BAD_CLOCK", {}, lambda d, i:
                (dict(d, revision=d["revision"]+1, started_at="SYNTHETIC_CHANGED"), {}))

    def test_lock_replacement_rejected(self):
        store = self.f.ledger.store
        old = store.lock.with_name("old-synthetic-lock")
        store.lock.rename(old)
        store.lock.touch()
        with self.assertRaisesRegex(ValueError, "lock"):
            store.snapshot()

    def test_large_blob_is_streamed_and_hash_checked(self):
        path = self.f.metadata / "SYNTHETIC_BLOB"
        path.write_bytes(b"synthetic" * 200000)
        ref = evidence.file_ref(path)
        self.assertEqual(evidence.verify_blob(ref), path.stat().st_size)
        with self.assertRaises(ValueError):
            evidence.verify_blob(dict(ref, sha256="0"*64))

    def test_synthetic_campaign_cannot_launch(self):
        campaign = Campaign(self.f.policy_ref)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            campaign.prepare_baseline({"baseline_run_id": "200001010001_SYNTHETIC"})

    def test_baseline_context_synthetic_readonly_and_host_guard(self):
        ref, c = self.f.reserve()
        current, p = baseline_runtime.load_context(ref)
        self.assertEqual(current["source_id"], c["source_id"])
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            p.host_guard()

    def test_failed_infrastructure_source_is_quarantined_not_reusable(self):
        ref, c = self.f.reserve()
        failure = evidence.exclusive(Path(c["run_dir"]) / "failure.json", dict(
            qualified=False, type="SYNTHETIC_FAILURE", native_failure_proven=False))
        self.f.ledger.close_failed_baseline(ref, failure, safe=True)
        d, _ = self.f.ledger.store.snapshot()
        s = d["source_index"][c["source_id"]]
        self.assertFalse(s["reusable_DM_failure"])
        self.assertEqual((d["qualified_count"], d["negative_count"]), (0, 0))
        self.assertIsNone(d["nodes"]["SYNTHETIC_ALIAS"]["active_run_id"])

    def positive_fixture(self):
        ref, c = self.f.reserve()
        _, _, _, (d, _) = self.f.pure_settlement(c)
        sid = c["source_id"]
        source = d["source_index"][sid]
        source.update(complete_readback=True, source_owner="SYNTHETIC_EXECUTOR")
        run_id = "200001010002_SYNTHETIC_TEACHER"
        file_ref = evidence.exclusive(self.f.metadata / "SYNTHETIC_EVIDENCE.json", dict(synthetic_fixture=True))
        refs = {k: file_ref for k in (
            "publication", "manifest", "acceptance", "context", "source_report", "teacher_report", "split_manifest")}
        d["reservations"][sid].update(quota_reserved=True, negative_quota_reserved=False,
            current_attempt_run_id=run_id, active_attempt_owner="SYNTHETIC_EXECUTOR",
            execution_owner="SYNTHETIC_EXECUTOR", source_owner="SYNTHETIC_EXECUTOR",
            current_attempt_context=file_ref, split="student_train", split_manifest=file_ref, positive_slots=1)
        request = dict(schema="refate.positive.ledger_commit_input.v1", data_kind="synthetic",
            campaign_id="SYNTHETIC_CAMPAIGN", owner="SYNTHETIC_EXECUTOR", source_owner="SYNTHETIC_EXECUTOR",
            host="SYNTHETIC_ALIAS", source_id=sid, task="synthetic_task", run_id=run_id,
            teacher="EmbodiedEvo public-recipe primitive baseline", split="student_train",
            group_id=source["group_id"], native_dm_success=False, native_teacher_success=True,
            conditions={"1.1": True, "1.2": True, "1.3": True}, refs=refs)
        api = self.f.ledger._bind("positive_impl.py", LOCK_INODE=17, SOURCE=sid, TASK="synthetic_task",
            SOURCE_OWNER="SYNTHETIC_EXECUTOR", _evidence_path=lambda p: Path(p).is_relative_to(self.f.data),
            _collection_split_compat=lambda row: False)
        return api, d, request

    def test_original_positive_transaction_and_exact_idempotence(self):
        api, d, request = self.positive_fixture()
        out, receipt = api.apply_commit(d, request, now=now(), lock_inode=17, synthetic=True)
        self.assertEqual((out["qualified_count"], out["negative_count"]), (1, 0))
        self.assertEqual((d["qualified_count"], d["negative_count"]), (0, 0))
        self.assertEqual(receipt["positive_delta"], 1)
        same, again = api.apply_commit(out, request, now=now(), lock_inode=17, synthetic=True)
        self.assertEqual(same, out)
        self.assertTrue(again["idempotent_reuse"])
        changed = copy.deepcopy(request)
        changed["refs"]["manifest"]["sha256"] = "f"*64
        with self.assertRaises(ValueError):
            api.apply_commit(out, changed, now=now(), lock_inode=17, synthetic=True)

    def test_original_positive_conditions_quota_reservation_and_group_fail_closed(self):
        api, d, request = self.positive_fixture()
        for condition in ("1.1", "1.2", "1.3"):
            changed = copy.deepcopy(request)
            changed["conditions"][condition] = False
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                api.apply_commit(d, changed, now=now(), lock_inode=17, synthetic=True)
        for mode in ("quota", "reservation", "group"):
            changed = copy.deepcopy(d)
            if mode == "quota":
                changed["tasks"]["synthetic_task"]["positive_quota"] = 0
            elif mode == "reservation":
                changed["reservations"][request["source_id"]]["quota_reserved"] = False
            else:
                changed["source_index"]["SYNTHETIC_OTHER"] = dict(group_id=request["group_id"], split="student_dev")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                api.apply_commit(changed, request, now=now(), lock_inode=17, synthetic=True)

    def test_unjournaled_metadata_change_is_not_trusted(self):
        store = self.f.ledger.store
        d = evidence.document(store.state)
        d["qualified_count"] = 1
        store.state.write_text(evidence.canonical(d) + "\n")
        with self.assertRaisesRegex(ValueError, "journal"):
            store.snapshot()

    def test_teacher_reservation_produces_real_restore_admission_inputs(self):
        ref, c = self.f.reserve()
        _, d, _, (settled, _) = self.f.pure_settlement(c)
        sid = c["source_id"]
        settled["source_index"][sid]["complete_readback"] = True
        self.f.ledger.store.transact("SYNTHETIC_PURE_SETTLEMENT", {}, lambda before, inode: (settled, {}))
        dispatch = self.f.ledger.reserve_teacher(sid, "200001010002_SYNTHETIC_TEACHER",
            "200001010002_SYNTHETIC_SESSION", 1, "SYNTHETIC_RESERVE_TEACHER", protocol={
                "source_commit": "SYNTHETIC_REVISION", "outline_sha256": "a"*64, "directive_sha256": "b"*64})
        self.assertEqual(dispatch["source_context_ref"], ref)
        self.assertEqual((dispatch["positive_slots"], dispatch["negative_slots"]), (1, 0))
        with self.assertRaises(ValueError):
            self.f.ledger.reserve_teacher(sid, "200001010003_SYNTHETIC_OTHER",
                "200001010003_SYNTHETIC_SESSION", 1, "SYNTHETIC_REPEAT", protocol={})


class MixedSourceTests(unittest.TestCase):
    def test_exact_mixed_membership_and_append_keep_limitations(self):
        f = CampaignFixture(mixed=True)
        self.addCleanup(f.close)
        original = f.catalog.prepare("synthetic_task", 1, f.split_ref)
        self.assertFalse(original["collection_group_compatibility"]["independent_evaluation_claimed"])
        ref = f.catalog.append("synthetic_task", f.split_ref, f.layout_refs[2], now())
        prepared = f.catalog.prepare("synthetic_task", 2, ref)
        self.assertEqual(prepared["source"]["split"], "student_train")
        self.assertFalse(prepared["collection_append_provenance"]["original_memberships_modified"])
        self.assertEqual(f.catalog.lineage(ref, "synthetic_task")["layouts"]["1"], f.split["layouts"]["1"])


class FrozenDMTests(unittest.TestCase):
    def test_fixed_config_and_unsupported_loading_rejected(self):
        with patch.object(frozen_impl, "MODEL_PATH", Path("/SYNTHETIC/model"), create=True), \
             patch.object(frozen_impl, "NATIVE_ROOT", Path("/SYNTHETIC/native"), create=True), \
             patch.object(frozen_impl, "EXPERIMENT", Path("/SYNTHETIC/native/XPolicyLab/policy/OpenDM/scripts/robodojo_dm05_history.py"), create=True):
            cfg = {"policy_name": "OpenDM", "model_path": "/SYNTHETIC/model"}
            good = frozen_impl.validate_config(cfg, 0)
            self.assertEqual((good["history_slots"], good["action_chunk_size"], good["action_steps"]), (20, 50, 25))
            for extra in ({"history_slots": 1}, {"action_steps": 1}, {"model_path": "/SYNTHETIC/other"},
                          {"gpu_id": 0}, {"history_enabled": 1}, {"unknown": 1}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    frozen_impl.validate_config(dict(cfg, **extra), 0)
            with self.assertRaises(ValueError):
                frozen_impl.validate_config(cfg, 0, construct_on_device=True)

    def test_freeze_requires_nonempty_fp32_cuda_parameters(self):
        parameter = SimpleNamespace(device="cuda:0", dtype="torch.float32", requires_grad=False, numel=lambda: 4)
        network = SimpleNamespace(training=False, requires_grad_=Mock(), eval=Mock(), parameters=lambda: [parameter])
        model = SimpleNamespace(_inference=SimpleNamespace(model=network))
        self.assertEqual(frozen_impl.freeze_model(model)["trainable_parameters"], 0)
        for field, value in (("requires_grad", True), ("device", "cpu"), ("dtype", "torch.float16")):
            original = getattr(parameter, field)
            setattr(parameter, field, value)
            with self.assertRaises(RuntimeError):
                frozen_impl.freeze_model(model)
            setattr(parameter, field, original)

    def test_native_action_shape_finite_and_scope(self):
        state = {k: np.zeros(v) for k, v in frozen_impl.STATE_DIMS.items()}
        self.assertEqual(set(frozen_impl.native14(state)), set(state))
        state["left_ee_joint_state"][0] = np.nan
        with self.assertRaises(ValueError):
            frozen_impl.native14(state)
        with self.assertRaises(ValueError):
            frozen_impl.scope_id({})


class BaselineReaderTests(unittest.TestCase):
    def setUp(self):
        self.f = CampaignFixture()
        self.addCleanup(self.f.close)
        self.run = self.f.data / "runs/200001010001_SYNTHETIC_READER"
        self.run.mkdir()
        self.fixture = BaselineFixture(self.run)
        self.c = dict(run_dir=str(self.run), source_id="SYNTHETIC_SOURCE", task="synthetic_task")

    def change(self, name, mutate):
        path = self.run / name
        value = evidence.document(path)
        mutate(value)
        path.write_text(evidence.canonical(value) + "\n")

    def test_full_original_baseline_reader_and_actual_decoder(self):
        report = baseline_reader.audit(self.f.runtime, self.c)
        self.assertTrue(report["file_recording_readback_passed"])
        self.assertEqual(report["actual_controls"], 10)
        self.assertEqual(report["decoded_RGB_images"], 6)
        self.assertFalse(report["qualified"])
        self.assertFalse(report["boundary_index_provenance"]["claims_actual_DM_API_calls"])

    def test_control_gap_and_physical_drift_rejected(self):
        path = self.run / "dense/control_steps.jsonl"
        original = path.read_bytes()
        rows = original.splitlines()
        path.write_bytes(b"\n".join(rows[:-1]) + b"\n")
        with self.assertRaises(ValueError):
            baseline_reader.audit(self.f.runtime, self.c)
        path.write_bytes(original)
        bad = evidence.decode(rows[0])
        bad["simulation_time_after"] = 1
        rows[0] = evidence.canonical(bad).encode()
        path.write_bytes(b"\n".join(rows) + b"\n")
        with self.assertRaises(ValueError):
            baseline_reader.audit(self.f.runtime, self.c)

    def test_conflicting_source_alias_rejected(self):
        self.change("source_identity.json", lambda d: d.update(source_id="SYNTHETIC_OTHER"))
        with self.assertRaises(ValueError):
            baseline_reader.audit(self.f.runtime, self.c)

    def test_bad_rgb_and_pt_counter_rejected(self):
        self.change("initial/image_stats.json", lambda d: d["cam_head"].update(sha256="0"*64))
        with self.assertRaises(ValueError):
            baseline_reader.audit(self.f.runtime, self.c)

    def test_pt_counter_drift_rejected(self):
        self.change("camera_000001.json", lambda d: d["rtx_profile"]["actual_settings"].update(
            {"/rtx/externalFrameCounter": 99}))
        with self.assertRaises(ValueError):
            baseline_reader.audit(self.f.runtime, self.c)

    def test_once_only_synthetic_summary_no_quota_and_tamper_rejected(self):
        identity = dict(source_id="SYNTHETIC_SOURCE", task="synthetic_task", group_id="SYNTHETIC_FAMILY",
            split="student_train", layout_ref=self.f.layout_refs[0], initialization_content_sha256="a"*64)
        self.change("source_identity.json", lambda d: (d.clear(), d.update(identity)))
        self.change("result.json", lambda d: d.update(source_id="SYNTHETIC_SOURCE", task="synthetic_task", native_step_limit=2))
        c = dict(self.c, schema="robofate.baseline-context.v1", data_kind="synthetic",
            campaign="SYNTHETIC_CAMPAIGN", execution_owner="SYNTHETIC_EXECUTOR", source_owner="SYNTHETIC_EXECUTOR",
            run_id=self.run.name, native_total_steps=2, preparation={"source": identity})
        ref = evidence.exclusive(self.run / "CONTEXT_001.json", c)
        q = baseline_reader.settle(ref, self.f.runtime)
        self.assertFalse(q["qualified_negative"])
        self.assertFalse(q["counted_in_quota"])
        self.assertIn("source_inventory_before.json", q["files"])
        self.assertEqual(baseline_reader.settle(ref, self.f.runtime), q)
        self.change("result.json", lambda d: d.update(native_steps=2))
        with self.assertRaises(ValueError):
            baseline_reader.settle(ref, self.f.runtime)

    def test_terminal_flags_are_not_inferred_from_exit(self):
        with self.assertRaises(ValueError):
            baseline_reader.native_label(dict(status="native_terminal", native_success=True,
                                              terminated=False, truncated=True))
        self.assertIsNone(baseline_reader.native_label(dict(status="native_terminal",
            native_success=False, terminated=False, truncated=True)))


class BaselineParentTests(unittest.TestCase):
    def setUp(self):
        self.f = CampaignFixture()
        self.addCleanup(self.f.close)
        self.ref, self.context = self.f.reserve()
        self.run = Path(self.context["run_dir"])
        self.context["self_ref"] = self.ref

    def fake_parent(self, *, startup_error=None):
        events = []
        processes = {}
        def launch(command, **kwargs):
            role = command[4]
            events.append("launch." + role)
            p = Mock(pid=731 if role == "native" else 732)
            p.poll.return_value = None
            p.wait.return_value = 0
            processes[role] = p
            self.assertEqual(kwargs["pass_fds"], (17,))
            return p
        def wait(path, children, deadline):
            events.append("wait." + path.name)
            if startup_error is not None and path.name == "native_boundary_ready.json":
                raise startup_error
            if path.name == "native_started.json":
                return evidence.document(path)
            if path.name == "native_boundary_ready.json":
                return dict(native_used=0, control_ticks=0, phase="actual_initial_boundary_saved_before_policy_load")
            return {"synthetic_fixture": True}
        with ExitStack() as stack:
            for target, name, value in (
                (bp, "identity", lambda pid: dict(pid=pid, start_ticks=1, process_group=pid, boot_id="SYNTHETIC_BOOT")),
                (self.f.runtime, "host_guard", lambda: None),
                (bp.deployment, "prepare", lambda p, c: ("SYNTHETIC_PYTHON", {})),
                (bp.service, "environment", lambda p, e: ("SYNTHETIC_DM_PYTHON", {})),
                (bp, "inspect_resources", lambda p, r: {"synthetic_fixture": True}),
                (owned, "lease", lambda *a: {"path": "SYNTHETIC_LEASE", "sha256": "a"*64}),
                (bp.subprocess, "Popen", launch), (bp, "wait_file", wait),
                (bp, "validate_ready", lambda *a: None), (owned, "stop_owned", lambda *a: 0),
                (owned, "original_absent", lambda *a: True), (owned, "group_members", lambda *a: []),
            ):
                stack.enter_context(patch.object(target, name, value))
            callback = lambda *a: events.append("actual_start_accounted")
            if startup_error:
                with self.assertRaises(type(startup_error)):
                    bp._foreground(self.ref, self.context, self.f.runtime, self.run, 17, on_started=callback)
            else:
                self.assertEqual(bp._foreground(self.ref, self.context, self.f.runtime, self.run, 17,
                                               on_started=callback), 0)
        return events, processes

    def test_native_saved_boundary_precedes_dm_and_both_owned_groups_clean(self):
        events, processes = self.fake_parent()
        self.assertLess(events.index("wait.native_boundary_ready.json"), events.index("launch.policy"))
        self.assertLess(events.index("actual_start_accounted"), events.index("launch.policy"))
        receipt = evidence.document(self.run / "process_cleanup.json")
        self.assertTrue(receipt["owned_cleanup_complete"])
        self.assertEqual(set(receipt) & {"native", "policy"}, {"native", "policy"})
        self.assertLessEqual(processes["native"].wait.call_args.kwargs["timeout"], 2400)

    def test_startup_failure_never_loads_dm_and_preserves_failure(self):
        events, processes = self.fake_parent(startup_error=TimeoutError("SYNTHETIC_NATIVE_STARTUP"))
        self.assertNotIn("launch.policy", events)
        self.assertFalse(evidence.document(self.run / "foreground_error.json")["qualified"])
        self.assertTrue(evidence.document(self.run / "process_cleanup.json")["owned_cleanup_complete"])
