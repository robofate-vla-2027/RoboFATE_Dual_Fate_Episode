"""Original one-step fixture; all values are synthetic, no Native/model calls."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robofate_pipeline.baseline import reader_impl as constants


def encoded(value):
    return (json.dumps(value, allow_nan=False) + "\n").encode()


class BaselineFixture:
    def __init__(self, run):
        self.run_dir = Path(run)
        self.sid = "SYNTHETIC_SOURCE"
        self.fs = {}
        self.fixed = dict(constants.PT_FIXED, **{"/rtx/pathtracing/spp": 4, "/rtx/pathtracing/totalSpp": 4})
        self.counter, self.profile = constants.PT_COUNTER, constants.PT4_COLLECTION
        self.cameras = sorted(constants.NATIVE25_CAMERAS)
        self.image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        png = io.BytesIO()
        Image.fromarray(self.image).save(png, format="PNG")
        self.png = png.getvalue()
        archive = io.BytesIO()
        np.savez_compressed(archive, array_0000=self.image)
        self.npz = archive.getvalue()
        archive = io.BytesIO()
        np.savez_compressed(archive)
        self.empty_npz = archive.getvalue()
        first, last = self.capture_boundary(0), self.capture_boundary(1)
        self.put("recording_contract.json", {"schema": "refate-native25-v1", "visual_hz": 25,
            "physical_dt_s": .004, "low_level_ticks_per_native": 10, "synthetic_fixture": True})
        self.put("result.json", {"status": "native_terminal", "native_steps": 1, "actual_control_steps": 10,
            "terminated": True, "truncated": False, "native_success": True,
            "qualified": False, "counted_in_quota": False, "synthetic_fixture": True})
        self.put("ready.json", {"native_step_limit": 2})
        self.put("initial_capture.json", {"native_count": 0, "physics_rng_exact": True, "boundary": first})
        self.put("source_identity.json", {"candidate_source_id": self.sid, "task": "synthetic_task"})
        self.put("failure.json", {"status": "SYNTHETIC_FAILED_DIAGNOSTIC", "failed": True})
        self.put("writer_close.json", {"status": "failed", "failed": True, "committed_boundaries": 0})
        self.put("timing_summary.json", {"status": "failed", "phase_counts": {"native_actions": 0}})
        self.put("collector_outcome.json", {"collector_outcome": "SYNTHETIC_failed", "container_exit_code": 1})
        self.fs[self.run_dir / "stdout.log"] = b"SYNTHETIC [Error] [rtx.plugin] diagnostic\n"
        self.actions = [{"index": 0, "native_count_after": 1, "actual_control_steps": 10,
            "action": [0.1], "terminated": True, "truncated": False, "native_success": True}]
        self.dense = []
        for i in range(1, 11):
            row = {"index": i, "macro_index": 0, "native_action_count": 1, "owner": "SYNTHETIC_DM",
                   "visual_interval": [0, 1], "obs_before": str(self.run_dir / "initial"),
                   "obs_after": str(self.run_dir / "macro_obs_000001"), "fresh_rgb_capture": i == 10,
                   "executed_actuator_targets": [0.1], "robot_feedback": {"SYNTHETIC": 1},
                   "decimation": 1, "physical_dt_s": .004, "simulation_time_before": (i - 1) * .004,
                   "simulation_time_after": i * .004, "wall_before_ns": 100 + i * 2, "wall_after_ns": 101 + i * 2}
            if i == 10: row["rtx_accumulation_boundary"] = last
            self.dense.append(row)
        observations, history = [], []
        configuration = {"profile": self.profile, "qualified": False,
                         "requested_carb_settings": {**self.fixed, self.counter: 0}}
        for i in (0, 1):
            folder = "initial" if i == 0 else "macro_obs_000001"
            observations.append({"index": i, "control_index": i * 10, "terminal": bool(i),
                "observation_dir": str(self.run_dir / folder), "simulation_time": i * .04,
                "rtx_accumulation_boundary": None if i == 0 else last})
            history.append({"native_index": i, "terminal": bool(i), "observation_dir": str(self.run_dir / folder)})
            vision, stats = {}, {}
            for camera in self.cameras:
                vision[camera] = {"color": {"shape": [480, 640, 3], "dtype": "uint8",
                    **({"npz_key": "array_0000"} if i == 0 else {"png_key": camera + ".png"})}}
                stats[camera] = {"shape": [480, 640, 3], "sha256": hashlib.sha256(self.image.tobytes()).hexdigest()}
                self.fs[self.run_dir / folder / (camera + ".png")] = self.png
            self.put(folder + "/observation.json", {"vision": vision, "state": {"SYNTHETIC": 1}, "instruction": "SYNTHETIC fixture only"})
            self.put(folder + "/image_stats.json", stats)
            self.fs[self.run_dir / folder / "arrays.npz"] = self.npz if i == 0 else self.empty_npz
            self.put(f"physics_{i:06d}.json", {"robots": {}, "objects": {}, "control": {"prev_control": [], "queues": []},
                "task": {"native_count": [i], "step_limit": 2, "dt": .004, "end_flag": [bool(i)], "success_flag": [True]},
                "reward": {}, "rng": {"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": []}, "unexposed": []})
            self.put(f"task_auxiliary_{i:06d}.json", {"pre_state": {}, "robot_origin_endpose": [], "joint_ratio_transition_state": {}})
            profile = {"configuration": configuration, "qualification": "unverified", "attached_hydra_engine_types": ["rtx"],
                       "actual_settings": {**self.fixed, self.counter: i + 1}}
            self.put(f"camera_{i:06d}.json", {"cameras": {c: {"shape": [480, 640, 3], "intrinsic": [], "extrinsic": [],
                "actual_usd_cameras": []} for c in self.cameras}, "renderer_settings": {"/rtx/rendermode": "PathTracing"}, "rtx_profile": profile})
        self.put("camera_profile.json", {"SYNTHETIC": True})
        raw = [{"observation_index": i, "profile": self.profile, "read_errors": {}, "configuration": configuration,
                "actual_settings": {**self.fixed, self.counter: i}, "wall_time_ns": 1000 + i,
                "attached_hydra_engines": {"status": "observed", "names": ["rtx"], "uids": [1],
                    "descriptions": [{"uid": 1, "engine_type_name": "rtx"}]}} for i in (0, 1, 2)]
        self.jsonl("renderer_raw_observations.jsonl", raw)
        self.jsonl("native_actions.jsonl", self.actions); self.jsonl("dense/control_steps.jsonl", self.dense)
        self.jsonl("native_observations.jsonl", observations); self.legacy_history = history

        for path, raw in self.fs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    def put(self, name, value):
        self.fs[self.run_dir / name] = encoded(value)

    def get(self, name):
        return json.loads(self.fs[self.run_dir / name])

    def jsonl(self, name, rows):
        self.fs[self.run_dir / name] = b"".join(encoded(row) for row in rows)

    def capture_boundary(self, index):
        settings = {key: {"status": "observed", "value": value} for key, value in (
            (self.counter, index + 1), ("/rtx/rendermode", "PathTracing"),
            ("/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", True))}
        value = {"previous_external_counter": index, "requested_external_counter": index + 1,
                 "physical_reset": False, "rtx_rng_reset_claimed": False, "actual_mode": "PathTracing",
                 "visual_index": index, "low_level_control_index": 10 * index}
        for offset, side in enumerate(("before", "after")):
            value[side] = {"schema": "v007.read_only_capture.1", "phase": side + "_existing_capture",
                           "capture_index": index, "native_count": 0, "physics_time": index * .04,
                           "carb_settings": settings,
                           "dispatcher_reference_time": {"status": "stable_read", "first": [index + offset, 1], "second": [index + offset, 1]}}
        loading = {"stage_loading_status": [0, 1, 1], "stage_streaming_busy": False}
        value["loading_before"] = value["loading_after"] = loading if index == 0 else None
        return value
