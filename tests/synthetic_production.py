"""Small explicitly synthetic disk fixtures. No real data, Native or model."""
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import numpy as np

from robofate_pipeline import evidence
from robofate_pipeline.collection import recording_modules
from robofate_pipeline.persistence import CameraBudget, Journal, api, camera
from robofate_pipeline.recording import core, observation_io, teacher
from robofate_pipeline.restoration.binding import HERE, IMPLEMENTATIONS, RestoreBinding
from robofate_pipeline.readback import reader

CAMERAS = teacher.CAMERAS
PROFILE = "pt_spatial4_collection_candidate_v1"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return evidence.file_ref(path)


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))


def observation(index):
    return dict(instruction="SYNTHETIC CPU FIXTURE ONLY",
                state={"synthetic_position": np.array([index], dtype=np.float64)},
                vision={c: {"color": np.full((480, 640, 3), index, dtype=np.uint8)} for c in CAMERAS})


def cameras():
    return {"cameras": {c: {"intrinsic": [[100, 0, 320], [0, 100, 240], [0, 0, 1]],
                           "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                           "shape": [480, 640, 3], "crop": None, "resize": None} for c in CAMERAS}}


def physical(index, terminal=False, success=False):
    robot = dict(joint_names=["synthetic_arm", "synthetic_grip"], arm_joint_indices=[0],
                 gripper_joint_indices=[1], root_state_w=[[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    for key in ("joint_pos", "joint_vel", "joint_pos_target", "joint_vel_target", "joint_effort_target"):
        robot[key] = [[0.0, 0.0]]
    return {
        "robots": {"synthetic_robot": robot},
        "objects": {"synthetic_object": dict(
            **{"class": "SyntheticObject"}, prim_path="/Synthetic/Object",
            world_pose=[[0, 0, 0], [1, 0, 0, 0]], linear_velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0], sleeping=False)},
        "control": {"queues": [], "prev_control": []},
        "task": {"native_count": [index], "end_flag": [terminal], "success_flag": [success],
                 "step_limit": 4, "dt": .004},
        "reward": {"synthetic_reward": 0}, "rng": {"synthetic_rng": [0, 1]},
        "unexposed": ["SYNTHETIC hidden state remains unknown"],
    }


class Fixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robofate-synthetic-production-")
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "data"
        self.config = self.root / "private-config"
        self.source = self.data / "runs/SYNTHETIC_SOURCE_RUN"
        self.run = self.data / "runs/SYNTHETIC_TEACHER_RUN"
        for path in (self.source, self.run, self.config):
            path.mkdir(parents=True)
        self.identity = dict(campaign="SYNTHETIC_CAMPAIGN", chat="synthetic", active_chat_id="synthetic-owner",
                             session="SYNTHETIC_SESSION", task="synthetic_task", source_id="SYNTHETIC_SOURCE",
                             owner="synthetic-owner", epoch=1)
        self.source_identity = dict(source_id="SYNTHETIC_SOURCE", task="synthetic_task",
                                    source_owner="synthetic-owner", encoded_seed=0, native_ordinal=0)
        self.preparation = dict(source=self.source_identity, native_step_limit=4, renderer_profile=PROFILE,
                                prospective_teacher_renderer_profile=PROFILE)
        self.source_context = dict(data_kind="synthetic", task="synthetic_task", native_total_steps=4,
                                   source_id="SYNTHETIC_SOURCE", run_id=self.source.name,
                                   run_dir=str(self.source), preparation=self.preparation)
        self.source_context_ref = put(self.source / "context.json", self.source_context)
        self.manifest = dict(source_id="SYNTHETIC_SOURCE", run_id=self.source.name, native_success=False)
        self.manifest_ref = put(self.config / "manifest.json", self.manifest)
        self.make_run(self.source, success=False)
        self.make_run(self.run, success=True)
        source_result = evidence.document(self.source / "result.json")
        rb = put(self.config / "source-readback.json", dict(
            file_recording_readback_passed=True, source_id="SYNTHETIC_SOURCE", result=source_result,
            data_kind="synthetic"))
        inventory = {str(p.relative_to(self.source)): reader.sig(p) for p in self.source.rglob("*") if p.is_file()}
        inventory_ref = put(self.config / "source-inventory.json", inventory)
        # The original reader scopes its retained readback evidence to data_root.
        for name, ref in (("source-readback.json", rb), ("source-inventory.json", inventory_ref)):
            (self.data / name).write_bytes(Path(ref["path"]).read_bytes())
        rb = evidence.file_ref(self.data / "source-readback.json")
        inventory_ref = evidence.file_ref(self.data / "source-inventory.json")
        self.summary = dict(source_id="SYNTHETIC_SOURCE", run=str(self.source),
                            file_recording_readback_passed=True, native_result=source_result,
                            files={"readback.json": rb, "source_inventory_before.json": inventory_ref},
                            data_kind="synthetic")
        self.summary_ref = put(self.data / "source-summary.json", self.summary)
        state = physical(0)
        robot = state["robots"]["synthetic_robot"]
        layout = {k: copy.deepcopy(robot[k]) for k in ("joint_names", "arm_joint_indices", "gripper_joint_indices")}
        layout.update(joint_array_shape=[1, 2], root_state_shape=[1, 13], fields=list(robot))
        rules = dict(schema_contract={"top_keys": list(state), "robots": {"synthetic_robot": layout}},
                     thresholds=dict(position_tol_m=.001, orientation_tol_rad=.001, linear_velocity_tol_mps=.001,
                                     angular_velocity_tol_radps=.001, quaternion_norm_error_max=.000001,
                                     joint_position_tol={"revolute_rad": .001, "prismatic_m": .001},
                                     joint_velocity_tol={"revolute_radps": .001, "prismatic_mps": .001}),
                     joint_units={"mapping_status": "SYNTHETIC TEST UNITS"})
        template = put(self.config / "rules.json", rules)
        invariants = put(self.config / "invariants.json", dict(
            required_discrete_exact=["task", "control", "rng"],
            coverage_requirements=["SYNTHETIC hidden state not established"]))
        admission = dict(schema="robofate.source-admission.v1", data_kind="synthetic",
                         execution_owner="synthetic-owner", sources={"SYNTHETIC_SOURCE": dict(
                             source_id="SYNTHETIC_SOURCE", task="synthetic_task", status="reserved",
                             consumed=False, refused=False, context_filename="context.json",
                             context_ref=self.source_context_ref, manifest_ref=self.manifest_ref,
                             readback_ref=self.summary_ref)})
        self.admission_ref = put(self.config / "admission.json", admission)
        self.policy = dict(schema="robofate.restore-policy.v1", data_kind="synthetic",
                           data_root=str(self.data), config_root=str(self.config), execution_owner="synthetic-owner",
                           partition="synthetic", task_totals={"synthetic_task": 4}, excluded_source_ids=[],
                           admission_ref=self.admission_ref, rules={"profile_template": template, "invariants": invariants},
                           code_pins={n: evidence.file_ref(HERE / n) for n in IMPLEMENTATIONS})
        self.policy_ref = put(self.config / "restore-policy.json", self.policy)
        self.binding = RestoreBinding(self.policy_ref)
        self.modules = recording_modules()
        self.material = self.binding.source_material(self.source_context_ref, self.manifest_ref, self.summary_ref, t1=1)
        self.api = self.binding.load(self.material, core_modules=self.modules)
        self.restore_preparation = self.binding.prepare_context(
            self.material, run=self.run, owner=self.identity, core_modules=self.modules)
        self.task = SimpleNamespace(task_name="synthetic_task", native_step_limit=4, max_steps=4,
                                    official_action_limit=True, control_steps=1, done=False,
                                    success=False, terminated=False, truncated=False, _failure=None)
        self.task.native = SimpleNamespace(take_action_cnt=[1], step_lim=4, dt=.004,
                                          obs_manager=SimpleNamespace(collect_interval=10))
        self.context = self.binding.bind_actual(self.api, self.restore_preparation, self.task)
        self.budget = CameraBudget("synthetic_task", 4, 1, 3, (("synthetic_task", 4),))
        self.task.public_observation = lambda: camera.enrich(
            observation(self.task.control_steps), cameras()["cameras"], self.task.control_steps, budget=self.budget)
        self.recorder = core.DenseRecorder.__new__(core.DenseRecorder)
        self.recorder.__dict__.update(task=self.task, run=self.run, control_index=10, capture_count=2,
                                     closed_attempted=False)
        self.writer = observation_io.OrderedObservationWriter.__new__(observation_io.OrderedObservationWriter)
        self.writer.__dict__.update(condition=threading.Condition(), first_error=None,
                                    last_committed=1, committed=2, submitted=2, outstanding=0,
                                    publication_in_flight=None, closed=False, closing=False, cleanup_errors=[])
        self.recorder.writer = self.writer
        self.history = [dict(native_step=i, observation=observation(i), executed_action=None if i == 0 else {"delta": 1},
                             feedback=dict(success=False, terminated=False, truncated=False)) for i in range(2)]

    def initialized_continuation(self):
        from robofate_pipeline.continuation import SourceContinuation
        self.task.control_steps = self.task.native.take_action_cnt[0] = 0
        self.task._frames = []
        self.task.public_observation = lambda: observation(self.task.control_steps)
        self.task.native.task_name = "synthetic_task"
        self.task.native.sim = SimpleNamespace(sim=SimpleNamespace(current_time=0.0))
        camera_values = list(cameras()["cameras"].values())
        self.task.native.camera_manager = SimpleNamespace(
            num_cams=3, camera_names=[CAMERAS],
            get_camera_intrinsics=lambda i, _: camera_values[i]["intrinsic"],
            get_camera_extrinsics=lambda i, _: camera_values[i]["extrinsic"])
        self.recorder.control_index, self.recorder.capture_count = 0, 1
        self.recorder.last_observation_time, self.recorder.controls = 0.0, []
        self.writer.last_committed, self.writer.committed, self.writer.submitted = 0, 1, 1
        self.events = []
        def control(action):
            self.task.control_steps += 1
            used = self.task.control_steps
            self.task.native.take_action_cnt[0] = used
            self.task.native.sim.sim.current_time = used * .04
            self.recorder.last_observation_time = used * .04
            self.recorder.control_index, self.recorder.capture_count = used*10, used+1
        def publish(event):
            self.events.append(event)
            used = self.task.control_steps
            self.writer.last_committed, self.writer.committed, self.writer.submitted = used, used+1, used+1
        self.task.control_step = control
        return SourceContinuation(self.binding, self.material, self.restore_preparation,
                                  task=self.task, recorder=self.recorder, publish=publish, identity=self.identity)

    def close(self):
        self.temp.cleanup()

    def make_run(self, run, *, success):
        put(run / "source_identity.json", self.source_identity)
        put(run / "ready.json", dict(actual_shared_recorder=True, native_used=0, native_step_limit=4))
        put(run / "initial_physics_state.json", physical(0))
        put(run / "initial_task_auxiliary.json", {"synthetic_aux": 0})
        put(run / "camera_profile.json", {"rtx_profile": {"configuration": {"profile": PROFILE}}})
        put(run / "recording_contract.json", dict(
            schema="refate-native25-v1", visual_hz=25, low_level_ticks_per_native=10, physical_dt_s=.004,
            camera_rgb_shape=[480, 640, 3], camera_aliases=list(CAMERAS)))
        actions, dense, observations = [], [], []
        for i in range(3):
            folder = core.recorded_observation_path(run, i)
            observation_io.save_observation(folder, observation(i))
            put(run / f"physics_{i:06d}.json", physical(i, i == 2, success and i == 2))
            put(run / f"task_auxiliary_{i:06d}.json", {"synthetic_aux": i})
            put(run / f"camera_{i:06d}.json", cameras())
            observations.append(dict(index=i, control_index=10*i, observation_dir=str(folder),
                                     terminal=i == 2, simulation_time=i*.04))
            if not i:
                continue
            actions.append(dict(index=i-1, native_count_after=i, actual_control_steps=10*i, action={"delta": i},
                                role="recorded_DM0.5_prefix_replay" if success and i == 1 else
                                "EmbodiedEvo_teacher" if success else "DM0.5",
                                original_native_event=str(self.source / "native_actions.jsonl"),
                                native_success=success and i == 2, terminated=i == 2, truncated=False))
            for j in range((i-1)*10+1, i*10+1):
                dense.append(dict(index=j, macro_index=i-1, native_action_count=i,
                                  obs_before=str(core.recorded_observation_path(run, i-1)),
                                  obs_after=str(folder), visual_interval=[i-1, i],
                                  executed_actuator_targets={"synthetic_target": 0}, robot_feedback={"synthetic_joint": 0},
                                  simulation_time_before=(j-1)*.004, simulation_time_after=j*.004,
                                  wall_before_ns=j*2, wall_after_ns=j*2+1, physical_dt_s=.004,
                                  fresh_rgb_capture=j % 10 == 0))
        rows(run / "native_actions.jsonl", actions)
        rows(run / "dense/control_steps.jsonl", dense)
        rows(run / "native_observations.jsonl", observations)
        put(run / "result.json", dict(native_steps=2, native_step_limit=4, native_success=success,
                                    terminated=True, truncated=False, actual_control_steps=20,
                                    mode="EmbodiedEvo_teacher_suffix" if success else "independent_baseline",
                                    rtx_profile=PROFILE))

    def journal(self):
        self.journal_value = Journal(
            api(), run=self.run, identity=self.identity, task=self.task, recorder=self.recorder, core=core,
            causal_history=self.history, source_run=self.source, t1=1, total=4,
            authorized_root=self.data, camera_budget=self.budget)
        return self.journal_value

    def journal_action(self):
        journal = self.journal()
        request = dict(kind="action", action={"delta": 2}, refate=dict(self.identity, sequence=1,
                       native_before=1, native_limit=4))
        model = SimpleNamespace(get_takeover_request=lambda: copy.deepcopy(request))
        journal.attach(model)
        model.get_takeover_request()
        self.task.control_steps = 2
        self.recorder.control_index, self.recorder.capture_count = 20, 3
        self.writer.last_committed, self.writer.committed, self.writer.submitted = 2, 3, 3
        event = evidence.document(self.run / "result.json")
        event["action"] = {"delta": 2}
        reference = journal.committed(self.recorder, before={"native_used": 1}, after={"native_used": 2}, event=event)
        return journal, reference

    def prepare_readback(self):
        # This fixture invokes real comparison/readback algorithms, not real controls.
        result = self.api.comparator.compare_files(
            self.source / "physics_000001.json", self.run / "physics_000001.json",
            self.restore_preparation["config"]["profile_path"], context=self.context)
        result["no_new_missing_fields"] = not result["coverage"]["missing_or_unhandled_paths"]
        put(self.run / "restore_state_comparison.json", result)
        put(self.run / "restore_boundary.json", dict(native_count=1, remaining_steps=3, restore_qualified=False))
        put(self.run / "restore_observation_comparison.json", dict(
            t1=1, instruction_exact=True, reference_state={"synthetic": 1}, restored_state={"synthetic": 1}))
        self.journal_action()
        context = dict(data_kind="synthetic", source_id="SYNTHETIC_SOURCE", task="synthetic_task",
                       run_dir=str(self.run), source_root=str(self.source), identity=self.identity,
                       execution_owner="synthetic-owner", campaign="SYNTHETIC_CAMPAIGN",
                       native_limit=4, t1=1, remaining_steps=3, source_readback_ref=self.summary_ref)
        self.readback_context_ref = put(self.run / "context.json", context)
        package = Path(reader.__file__).parent
        policy = dict(schema="robofate.readback-policy.v1", data_kind="synthetic", data_root=str(self.data),
                      manifest_root=str(self.data / "published"), execution=dict(
                          host="SYNTHETIC", hostname="SYNTHETIC", owner_chat="synthetic-owner", campaign="SYNTHETIC_CAMPAIGN"),
                      code_pins={n: evidence.file_ref(package / n) for n in
                                 ("reader.py", "decoder.py", "contracts.py", "content.py", "publisher_impl.py", "settlement.py")},
                      protocol_refs={n: put(self.config / (n + ".json"), {"data_kind": "synthetic", "name": n})
                                     for n in ("outline", "directive")})
        self.readback_policy_ref = put(self.config / "readback-policy.json", policy)
        return self.readback_context_ref, self.readback_policy_ref
