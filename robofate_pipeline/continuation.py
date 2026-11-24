"""Compose original replay, camera, journal and teacher gates on an owned Native.

The separate execution module constructs Native and the production worker.
This component can also operate on explicitly supplied caller-owned objects.
"""
import copy
from pathlib import Path

from . import collection, evidence, persistence
from .evidence import canonical, need
from .recording import core, teacher as common_teacher
from .recording.ownership import SoleOwner


class SourceContinuation:
    def __init__(self, binding, material, preparation, *, task, recorder, publish, identity):
        self.binding = binding
        self.material, self.preparation = copy.deepcopy(material), copy.deepcopy(preparation)
        self.task, self.recorder, self.publish = task, recorder, publish
        self.modules = collection.recording_modules()
        self.api = binding.load(material, core_modules=self.modules)
        self.context = binding.bind_actual(self.api, preparation, task)
        need(Path(recorder.run) == Path(preparation["run"]) and recorder.task is task
             and canonical(identity) == canonical(preparation["owner"]),
             "Owned recorder/run/identity differs from pinned preparation")
        self.owner = SoleOwner(task, recorder, identity)
        self.proof = None
        self.teacher = self.journal = None

    def source_gate(self):
        m = self.material
        current = self.binding.source_material(m["source_context"], m["source_manifest"],
                                               m["source_readback"], t1=m["t1"])
        need(canonical(current) == canonical(m), "Source changed after admission")

    def restore_gate(self):
        need(self.proof is not None and self.task.control_steps == self.material["t1"]
             and not self.task.done, "No current nonterminal restored boundary")
        for ref in self.proof.values():
            self.binding.read_pin(ref)
        comparison = evidence.decode(self.binding.read_pin(self.proof["state"]))
        need(comparison["all_compared_fields_within_candidate_rules"] is True
             and comparison["no_new_missing_fields"] is True,
             "Original restored-state checks did not pass")
        self.api.context.check_native(self.context, self.api.context.native_snapshot(self.task),
                                      expected_count=self.material["t1"])

    def restore(self):
        need(self.proof is None and self.teacher is None and self.task.control_steps == 0,
             "Prefix restore cannot be repeated")
        run = Path(self.preparation["run"])
        try:
            with self.owner.guard():
                self.source_gate()
                self.api.support.replay_prefix(
                    self.task, self.recorder, run, self.publish, self.preparation["config"],
                    core.write, context=self.context, main_execute=True)
                self.proof = {key: self.binding.ref(run / name) for key, name in {
                    "state": "restore_state_comparison.json", "observation": "restore_observation_comparison.json",
                    "auxiliary": "restore_auxiliary_render_comparison.json", "boundary": "restore_boundary.json",
                }.items()}
                self.restore_gate()
        except BaseException as error:
            self._fail(error)
            raise
        return dict(self.proof)

    def bind_teacher(self, runtime, model):
        """Attach original persistence before the existing teacher can dispatch."""
        need(self.teacher is None and self.journal is None, "Teacher may only be attached once")
        m = self.material
        config = runtime.binding
        need(config["identity"] == self.preparation["owner"] and config["source_id"] == m["source_id"]
             and config["task"] == m["task"] and config["native_limit"] == m["native_limit"]
             and config["t1"] == m["t1"] and model.cfg["runtime_binding"] == config,
             "Teacher/runtime differs from original prepared source")
        need(model.worker is None and model.connection is None, "Use a fresh unstarted teacher model")
        session = Path(config["run_dir"])
        need(session.is_absolute() and session.resolve() == session
             and session.parent == self.binding.root / "sessions"
             and session.name == config["identity"]["session"], "Teacher must use its own session")
        def provider_gate():
            marker = session / "agent/provider_safety.json"
            if marker.exists() or marker.is_symlink():
                raise RuntimeError("Provider stop retained; no further prediction or control")
        try:
            with self.owner.guard():
                self.source_gate()
                self.restore_gate()
                provider_gate()
                budget = persistence.CameraBudget(m["task"], m["native_limit"], m["t1"],
                                                  m["native_limit"]-m["t1"],
                                                  tuple(self.binding.policy["task_totals"].items()))
                persistence.camera.install(self.task, self.preparation["run"], m["t1"], budget=budget)
                history = common_teacher.load_causal_history(
                    source_run=m["source"], t1=m["t1"], budget=m["native_limit"], core_modules=self.modules,
                    policy_observation=runtime.takeover.policy_observation)
                self.journal = persistence.Journal(
                    persistence.api(), run=self.preparation["run"], identity=config["identity"], task=self.task,
                    recorder=self.recorder, core=core, causal_history=history, source_run=m["source"],
                    t1=m["t1"], total=m["native_limit"], authorized_root=self.binding.root, camera_budget=budget)
                self.journal.attach(model)
            self.teacher = common_teacher.bind_existing_teacher(
                runtime, model, self.task, self.recorder, source_run=m["source"], task_name=m["task"],
                t1=m["t1"], budget=m["native_limit"], config=config, causal_history=history,
                core_modules=self.modules, owner_guard=self.owner.guard, owner_snapshot=self.owner.snapshot,
                check_source_admission=self.source_gate, check_restored_t1=self.restore_gate,
                check_provider_safety=provider_gate, publish=self.publish, committed_dense_ref=self.journal.committed)
        except BaseException as error:
            self._fail(error)
            raise
        return self.teacher

    def create_teacher(self, runtime, *, main_binding_ref):
        """Build the original unstarted model from the actual restored boundary."""
        from .teacher.specification import model
        need(self.teacher is None and self.journal is None, "Teacher already attached")
        m = self.material
        try:
            with self.owner.guard():
                self.source_gate()
                self.restore_gate()
                context = dict(self.owner.snapshot(), native_used=self.task.control_steps,
                    native_limit=self.task.native.step_lim, remaining_steps=m["native_limit"] - m["t1"],
                    boundary_receipt_ref=str(Path(self.preparation["run"]) / "restore_state_comparison.json"),
                    budget_mapping_ref=str(Path(self.preparation["run"]) / "restore_request.json"),
                    native_terminal=self.task.done)
                observation = common_teacher.public_observation(
                    self.task.public_observation(), runtime.takeover.policy_observation)
                history = common_teacher.load_causal_history(
                    source_run=m["source"], t1=m["t1"], budget=m["native_limit"],
                    core_modules=self.modules, policy_observation=runtime.takeover.policy_observation)
                fresh = model(runtime, main_binding_ref=main_binding_ref, context=context,
                              observation=observation, causal_history=history)
            return self.bind_teacher(runtime, fresh)
        except BaseException as error:
            self._fail(error)
            raise

    def run(self):
        need(self.teacher is not None, "Bind the teacher before running")
        # Data readback and caller-owned cleanup remain separate from model exit.
        return collection.run_teacher(self.teacher, wall_seconds=3600)

    def _fail(self, error):
        self.owner.failed = True
        try:
            self.recorder.writer.fail(error)
        except BaseException as secondary:
            if secondary is not error:
                error.add_note("Writer failure latch also failed: " + repr(secondary))
