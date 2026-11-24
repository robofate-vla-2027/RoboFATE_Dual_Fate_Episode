"""Original worker/Native logic on synthetic CPU peers only.

Targets are temporary files and in-memory fake peers. No real processes, signals,
network connections, models, simulator, provider request or GPU operation runs.
The outer runner's network/process/signaling audit guard remains enabled.
"""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robofate_pipeline import evidence
from robofate_pipeline.native import NativeRuntime
from robofate_pipeline.native.diagnostics import reset_preserving_primary
from robofate_pipeline.native.initialization import wrap_class
from robofate_pipeline.native.resources import validate_allocation
from robofate_pipeline.teacher import install, model
from robofate_pipeline.teacher import provider_safety
from robofate_pipeline.runtime_policy import PACKAGE
from synthetic_runtime import RuntimeFixture


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.f = RuntimeFixture()
        self.addCleanup(self.f.close)

    def runtime(self):
        return install(self.f.binding)

    def model(self, runtime):
        f = self.f
        return model(runtime, main_binding_ref=f.main_ref, context=f.context,
                     observation=f.observation, causal_history=f.history)

    def test_production_model_constructs_without_worker(self):
        runtime = self.runtime()
        original = self.model(runtime)
        self.assertIsNone(original.worker)
        self.assertIsNone(original.connection)
        self.assertEqual(original.cfg["seconds"], 3600)
        self.assertEqual(original.cfg["task_config"]["agent"]["model"], "gt-6-as-a")
        self.assertEqual(original.cfg["task_config"]["test"]["budget"], 100)
        original.takeover(self.f.observation, self.f.history, 3)
        self.assertEqual(original.step_limit, 4)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            original.get_action()
        self.assertIsNone(original.worker)

    def test_takeover_and_original_model_action_receipt(self):
        runtime = self.runtime()
        original = self.model(runtime)
        original.takeover(self.f.observation, self.f.history, 3)
        request = dict(kind="action", action={"robot_arm_joint_state": [1.]},
                       refate=runtime.takeover.request_metadata(self.f.context, 1, 1))
        pipe = Mock()
        pipe.poll.return_value = True
        pipe.recv.return_value = request
        original.connection, original.worker = pipe, Mock()
        actions = original.get_action()
        self.assertEqual(actions, [request["action"]])
        self.assertTrue(original.pending)
        with self.assertRaisesRegex(RuntimeError, "receipt"):
            original.get_action()
        transition = dict(kind="transition", observation=self.f.observation,
                          success=False, terminated=False, truncated=False,
                          refate=dict(request["refate"], native_after=2, dense_trace_ref="SYNTHETIC_COMMITTED"))
        original.accept_transition(transition)
        pipe.send.assert_called_once_with(transition)
        self.assertFalse(original.pending)
        self.assertIsNone(original.takeover_request)

    def test_original_bridge_latches_unknown_action(self):
        runtime = self.runtime()
        c = runtime.takeover.validate_context(self.f.context, self.f.observation, self.f.history, 3)
        connection = Mock()
        connection.recv.side_effect = RuntimeError("SYNTHETIC lost receipt")
        bridge = runtime.takeover.TakeoverBridgeTask(connection, self.f.observation, 4, context=c)
        bridge.reset()
        with self.assertRaisesRegex(RuntimeError, "lost receipt"):
            bridge.control_step({"robot_arm_joint_state": [1.]})
        self.assertTrue(bridge.invalid)
        with self.assertRaises(RuntimeError):
            bridge.control_step({"robot_arm_joint_state": [1.]})
        self.assertEqual(connection.send.call_count, 1)

    def test_worker_budget_identity_history_and_once_only(self):
        runtime = self.runtime()
        for key, value in (("task", "DIFFERENT_SYNTHETIC_TASK"), ("native_used", 2),
                           ("remaining_steps", True), ("epoch", 2)):
            context = dict(self.f.context, **{key: value})
            with self.assertRaises((ValueError, RuntimeError)):
                model(runtime, main_binding_ref=self.f.main_ref, context=context,
                      observation=self.f.observation, causal_history=self.f.history)
        original = self.model(runtime)
        original.takeover(self.f.observation, self.f.history, 3)
        with self.assertRaises(RuntimeError):
            original.takeover(self.f.observation, self.f.history, 3)
        with self.assertRaises(RuntimeError):
            runtime.takeover.validate_worker_spec(original.cfg, 1)
        future = self.f.history + [self.f.history[-1]]
        with self.assertRaises(RuntimeError):
            runtime.takeover.validate_context(self.f.context, self.f.observation, future, 3)

    def test_original_worker_provider_checks_not_removed(self):
        runtime = self.runtime()
        path = Path(runtime.original_worker.__file__)
        calls = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "require_open"]
        self.assertEqual(len(calls), 2)

    def test_main_pin_and_runtime_change_rejected(self):
        runtime = self.runtime()
        runtime.load_binding(self.f.main_ref, context=self.f.context)
        evidence.exclusive(self.f.data / "other.json", {"synthetic": True})
        path = Path(self.f.main_ref["path"])
        path.write_text("{}")
        with self.assertRaises(ValueError):
            runtime.load_binding(self.f.main_ref)
        Path(self.f.policy_ref["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            runtime.validate_binding()

    def test_policy_rejects_linked_root_and_noncanonical_output(self):
        binding = copy.deepcopy(self.f.binding)
        binding["run_dir"] += "/../" + self.f.session
        with self.assertRaises(ValueError):
            install(binding)

    def test_binding_rejects_synthetic_relabel_and_native_bool(self):
        for changes in (dict(data_kind="actual"), dict(native_limit=True), dict(t1=True),
                        dict(actor_wall_s=60), dict(source_id="OTHER_SYNTHETIC")):
            with self.assertRaises(ValueError):
                install(dict(self.f.binding, **changes))

    def test_original_prompt_and_optional_tool_setting(self):
        runtime = self.runtime()
        from orchestrator.config import config_from_dict
        command = runtime.stage.original_command(config_from_dict(runtime.task_config()))
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn('web_search="disabled"', command)
        self.assertIn('model="gt-6-as-a"', command)
        self.assertIn("never open raw_cam_*", command[-1])
        self.assertNotIn("--search", command)

    def test_original_oci_mounts_keep_private_sources_out(self):
        runtime = self.runtime()
        run = self.f.run
        config = runtime.stage.configuration(run, run / "runtime/socket", ["SYNTHETIC"])
        mounts = config["mounts"]
        self.assertEqual(config["root"], dict(path="rootfs", readonly=True))
        self.assertEqual({r["type"] for r in config["linux"]["namespaces"]},
                         {"user", "pid", "network", "ipc", "uts", "mount"})
        self.assertTrue(config["process"]["noNewPrivileges"])
        self.assertTrue(all(not value for value in config["process"]["capabilities"].values()))
        self.assertIn("CUDA_VISIBLE_DEVICES=", config["process"]["env"])
        for name in ("primitives", "scripts", "observations", "instruction.md"):
            row = next(row for row in mounts if row["destination"] == "/workspace/" + name)
            self.assertIn("ro", row["options"])
        self.assertFalse(any(str(PACKAGE.parent) in r["source"] for r in mounts
                             if r["destination"] != "/inner.py"))
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            runtime.stage.stage(None, run, run, owner=self.f.identity["active_chat_id"], synthetic=True)

    def test_provider_structured_stop_persists_and_precedes_transport(self):
        path = self.f.data / "synthetic-provider-stop.json"
        gate = provider_safety.Gate(path)
        parser = provider_safety.Parser(gate)
        value = dict(type="response.failed", response=dict(error=dict(code="cyber_policy", message="timeout")))
        payload = b"data: " + json.dumps(value).encode() + b"\n"
        parser.feed(payload[:20])
        self.assertFalse(gate.stopped.is_set())
        parser.feed(payload[20:])
        with self.assertRaises(provider_safety.ProviderSafetyStop):
            gate.require_open()
        receipt = evidence.document(path)
        self.assertFalse(receipt["automatic_replay"])
        self.assertTrue(receipt["original_response_not_rewritten"])
        self.assertIsNone(provider_safety.refusal(dict(error=dict(code="timeout"))))
        gate.observe(value)
        self.assertEqual(evidence.document(path), receipt)

    def test_provider_stop_blocks_before_forwarding(self):
        class Handler:
            def forward(self):
                raise AssertionError("No actual forward is permitted in this synthetic test")
            def send_error(self, code, text):
                self.sent = code
        class Connection:
            pass
        proxy = SimpleNamespace(http=SimpleNamespace(client=SimpleNamespace(HTTPConnection=Connection)))
        gate = provider_safety.Gate(self.f.data / "synthetic-stop.json")
        gate.observe({"error": {"code": "safety_refusal"}})
        factory = provider_safety.install(proxy, lambda _: Handler,
                                         SimpleNamespace(local=SimpleNamespace()), gate)
        handler = factory("model")()
        handler.forward()
        self.assertEqual(handler.sent, 409)

    def test_native_creation_materializes_original_hooks_without_simulator(self):
        native = NativeRuntime(self.f.policy_ref)
        hooks, source = native.creation_source()
        ast.parse(source)
        self.assertIn('_make_app(AppLauncher, args)', source)
        self.assertIn("True, True, \"cuda:1\"", source)
        self.assertIn("device_id=1", source)
        self.assertIn("_node_prepare_native(self.native)", source)
        self.assertIn("cfg.sim.device = \"cpu\"", source)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            native.install()
        self.assertFalse(native.installed)

    def test_original_initial_binding_checks_budget_and_layout(self):
        hooks, _ = NativeRuntime(self.f.policy_ref).creation_source()
        layout = self.f.data / "synthetic-layout.json"
        ref = evidence.exclusive(layout, {"SYNTHETIC": True})
        prep = dict(source=dict(task="synthetic_task", native_bank=0, native_ordinal=0,
                    resolved_layout_path=str(layout), layout_ref=ref), native_step_limit=4)
        task = SimpleNamespace(task_name="synthetic_task", official_action_limit=True,
            max_steps=4, native_step_limit=4, control_steps=0, bank=0, ordinal=0, layout_path=layout,
            native=SimpleNamespace(step_lim=4, take_action_cnt=[0], dt=.004,
                                   obs_manager=SimpleNamespace(collect_interval=10)))
        result = hooks.verify_initial(task, prep)
        self.assertFalse(result["actual_render_device_verified"])
        task.max_steps = 5
        with self.assertRaises(ValueError):
            hooks.verify_initial(task, prep)

    def test_native_layout_rejects_unpinned_or_wrong_seed(self):
        native = NativeRuntime(self.f.policy_ref)
        path = self.f.data / "synthetic-layout.json"
        ref = evidence.exclusive(path, {"SYNTHETIC": True})
        preparation = dict(native_step_limit=4, source=dict(task="synthetic_task",
            source_id="SYNTHETIC_SOURCE", native_bank=0, native_ordinal=0, encoded_seed=0,
            resolved_layout_path=str(path), layout_ref=ref))
        with self.assertRaisesRegex(ValueError, "not pinned"):
            native.layout(preparation)
        preparation["source"]["encoded_seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed/bank/ordinal"):
            native.layout(preparation)

    def test_primary_exception_is_preserved_and_diagnosed(self):
        primary = RuntimeError("SYNTHETIC primary")
        client = Mock()
        client.call.side_effect = ValueError("SYNTHETIC cleanup")
        try:
            raise primary
        except RuntimeError:
            reset_preserving_primary(client, {"synthetic": True}, self.f.data)
        receipt = evidence.document(self.f.data / "final_reset_secondary_error.json")
        self.assertTrue(receipt["primary_rethrown"])
        self.assertFalse(receipt["normal_shutdown_claimed"])
        with self.assertRaises(ValueError):
            reset_preserving_primary(client, {}, self.f.data)

    def test_initialization_receipt_is_not_success(self):
        error = ValueError("SYNTHETIC initialization")
        class Task:
            def _create_native(self):
                raise error
        cls = wrap_class(Task, self.f.data, evidence_kind="synthetic_initialization_exception")
        with self.assertRaises(ValueError) as raised:
            cls()._create_native()
        self.assertIs(raised.exception, error)
        self.assertFalse(evidence.document(self.f.data / "initialization_failure.json")["scientific_qualification"])

    def test_empty_code_pin_does_not_permit_empty_metadata(self):
        path = self.f.data / "synthetic-empty.py"
        path.touch()
        reference = evidence.file_ref(path, allow_empty=True)
        self.assertEqual(evidence.read_pin(reference, allow_empty=True), b"")
        with self.assertRaises(ValueError):
            evidence.read_pin(reference)

    def test_rng_adapter_preserves_simulation_generator(self):
        runtime = self.runtime()
        rng = runtime.policy.load("native/transport_rng.py")
        path = self.f.data / "synthetic-ping.py"
        path.write_text("import random\ndef ping():\n    return random.getrandbits(32)\n")
        connection = SimpleNamespace(__file__=str(path), random=random)
        before = random.getstate()
        receipt = rng.install(connection, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(random.getstate(), before)
        self.assertIsNot(connection.random, random)
        self.assertFalse(receipt["simulation_rng_changed"])
        self.assertEqual(rng.install(connection, receipt["source"]["sha256"]), receipt)
        connection.random = random
        with self.assertRaisesRegex(ValueError, "identity changed"):
            rng.install(connection, receipt["source"]["sha256"])

    def test_gpu_admission_respects_scheduler_identity_and_headroom(self):
        devices = dict(visible_gpu_indices=[0, 1], physical_gpu_ids=[4, 5],
                       gpu_uuids=["SYNTHETIC_A", "SYNTHETIC_B"])
        snapshot = "0, SYNTHETIC_A, 50000, 80000, 30\n1, SYNTHETIC_B, 40000, 80000, 20\n"
        receipt = validate_allocation(devices, snapshot, "SYNTHETIC_PEERS",
                                     {"SLURM_JOB_GPUS": "4,5"})
        self.assertTrue(receipt["peer_processes_are_not_managed"])
        for env in ({"SLURM_JOB_GPUS": "4"}, {"PBS_JOBID": "SYNTHETIC"}):
            with self.assertRaises(ValueError):
                validate_allocation(devices, snapshot, "", env)
        with self.assertRaises(ValueError):
            validate_allocation(devices, snapshot.replace("50000", "40000"), "", {})
        with self.assertRaises(ValueError):
            validate_allocation(devices, snapshot.replace("SYNTHETIC_A", "SYNTHETIC_OTHER"), "", {})

    def test_resources_never_signal_a_changed_owned_identity(self):
        runtime = self.runtime()
        cls = runtime.original_worker.AgentResources
        resources = cls.__new__(cls)
        process = Mock(pid=123)
        process.poll.return_value = None
        resources.process = process
        resources.created = {"SYNTHETIC": "old"}
        namespace = cls.stop_agent_client.__globals__
        with patch.dict(namespace, identity=lambda _: {"SYNTHETIC": "changed"}):
            with self.assertRaises(ValueError):
                resources.stop_agent_client(process)
        process.terminate.assert_not_called()
        with patch.dict(namespace, identity=lambda _: {"SYNTHETIC": "old"}):
            resources.stop_agent_client(process)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=25)

    def test_cleanup_requires_real_lifecycle_receipt(self):
        runtime = self.runtime()
        cls = runtime.original_worker.AgentResources
        resources = cls.__new__(cls)
        resources.process = Mock(pid=123)
        resources.process.poll.return_value = 0
        resources.process.wait.return_value = 0
        resources.run = self.f.run
        resources.config_sha = "SYNTHETIC"
        resources.server = resources.thread = resources.socket_identity = resources.timing_sink = None
        with self.assertRaisesRegex(ValueError, "lifecycle missing"):
            resources.cleanup()
        self.assertFalse((self.f.run / "agent/worker_wait.json").exists())

    def test_terminal_result_is_not_a_qualification(self):
        from robofate_pipeline.execution import terminal_result
        task = SimpleNamespace(control_steps=2, native=SimpleNamespace(step_lim=4), done=True,
                               success=True, terminated=True, truncated=False)
        recorder = SimpleNamespace(control_index=20)
        context = dict(native_limit=4, t1=1, source_id="SYNTHETIC_SOURCE", task="synthetic_task")
        with patch("robofate_pipeline.execution.core.wait_for_native_publication") as wait:
            value = terminal_result(task, recorder, context)
        wait.assert_called_once_with(recorder, 2)
        self.assertTrue(value["native_success"])
        self.assertFalse(value["qualified"])
        self.assertFalse(value["counted_in_quota"])
        self.assertEqual(value["teacher_actions"], 1)

    def test_original_cleanup_order_and_failure_retention(self):
        from robofate_pipeline.execution import cleanup
        calls = []
        recorder = SimpleNamespace(closed_attempted=False, phases={},
                                   writer=SimpleNamespace(metrics={}))
        def close_recorder():
            calls.append("recorder")
            raise RuntimeError("SYNTHETIC writer-close failure")
        recorder.close = close_recorder
        model = SimpleNamespace(worker=None, close=lambda: calls.append("model"))
        task = SimpleNamespace(close=lambda: calls.append("task"))
        errors = cleanup(recorder, model, task, self.f.run, {}, 0)
        self.assertEqual(calls, ["recorder", "model", "task"])
        self.assertEqual(len(errors), 1)
        receipt = evidence.document(self.f.run / "timing_summary.json")
        self.assertIn("SYNTHETIC", receipt["cleanup_errors"][0])

    def test_synthetic_dispatch_cannot_prepare_an_episode(self):
        from robofate_pipeline.execution import prepare_teacher
        ref = evidence.exclusive(self.f.data / "synthetic-dispatch.json",
                                 dict(schema="robofate.teacher-dispatch.v1", data_kind="synthetic"))
        with self.assertRaisesRegex(ValueError, "Actual"):
            prepare_teacher(ref, None, self.f.policy_ref)
        self.assertFalse(list((self.f.data / "runs").iterdir()))
