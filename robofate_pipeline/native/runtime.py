"""Original creation hooks with pinned layout/assets and explicit device roles."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path

from .. import evidence
from ..evidence import canonical, need
from ..runtime_policy import RuntimePolicy
from .initialization import wrap_class


class NativeRuntime:
    def __init__(self, policy_ref):
        self.policy = RuntimePolicy(policy_ref)
        self.p = self.policy.value
        self.installed = False

    def layout(self, preparation):
        """Re-read ordered raw layout, never regenerate it from context metadata."""
        source = preparation["source"]
        need(source["task"] in self.p["task_totals"]
             and type(preparation["native_step_limit"]) is int
             and preparation["native_step_limit"] == self.p["task_totals"][source["task"]],
             "Unbound Native task budget")
        need(source["source_id"] not in self.p["excluded_source_ids"], "Excluded source")
        for key in ("native_bank", "native_ordinal", "encoded_seed"):
            need(type(source[key]) is int and source[key] >= 0, "Invalid layout coordinate")
        path = Path(source["resolved_layout_path"])
        need(path == Path(source["layout_ref"]["path"]).resolve(strict=True)
             and source["encoded_seed"] == source["native_bank"] * 1_000_000 + source["native_ordinal"],
             "Source seed/bank/ordinal mismatch")
        self.policy.pin(path)
        need(self.policy.pins[str(path)]["sha256"] == source["layout_ref"]["sha256"],
             "Layout is not pinned by this execution release")
        raw = evidence.read_pin({"path": str(path), "sha256": source["layout_ref"]["sha256"]})
        return evidence.decode(raw)

    def device_guard(self):
        self.policy.host_guard()
        devices = self.p["native_devices"]
        need(type(devices["cuda_visible_devices"]) is str
             and len(devices["cuda_visible_devices"].split(",")) == 2
             and len(set(devices["cuda_visible_devices"].split(","))) == 2
             and os.environ.get("CUDA_VISIBLE_DEVICES") == devices["cuda_visible_devices"]
             and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
             "Original two-device planner/simulation binding required")
        need(type(devices["nvidia_visible_devices"]) is str and devices["nvidia_visible_devices"]
             and os.environ.get("NVIDIA_VISIBLE_DEVICES") == devices["nvidia_visible_devices"],
             "Explicit device visibility binding required")
        need(type(devices["renderer_index"]) is int and devices["renderer_index"] == 1,
             "Original isolated renderer logical index must remain one")

    def creation_source(self):
        """Materialize the original creation function without importing Isaac Sim."""
        source = self.policy.teacher / "simulators/robodojo.py"
        raw = self.policy.pin(source)
        hooks = self.policy.load("native/hooks_impl.py", {
            "UPSTREAM_SHA": hashlib.sha256(raw).hexdigest(), "need": need,
            "NATIVE": self.policy.native, "_planner_refs": self.p["planner_refs"],
            "_read_pin": evidence.read_pin})
        return hooks, hooks.creation_source(raw)

    def make_app(self, launcher, args):
        self.device_guard()
        class BoundLauncher(launcher):
            def _create_app(self):
                self._sim_app_config.update(active_gpu=1, physics_gpu=1, multi_gpu=False)
                return super()._create_app()
        return BoundLauncher(args, multi_gpu=False, active_gpu=1, physics_gpu=1,
                             fast_shutdown=True, renderer="RaytracedLighting")

    def install(self):
        self.device_guard()
        need(not self.installed, "Native runtime cannot be installed twice")
        self.policy.verify_release()
        self.policy.install_import_paths()
        experience = self.policy.native / "third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit"
        self.policy.pin(experience)
        self.policy.pin(self.policy.native / "third_party/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py")
        for root in (self.policy.native / "env", self.policy.native / "src",
                     self.policy.native / "task", self.policy.native / "env_cfg"):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".yml", ".yaml", ".json"):
                    self.policy.pin(path)
        for ref in self.p["planner_refs"].values():
            self.policy.pin(ref["path"])
            need(self.policy.pins[ref["path"]] == ref, "Planner release mismatch")
        material_root = self.policy.native / "deployment/offline_materials"
        manifest = evidence.decode(self.policy.pin(material_root / "manifest.json"))
        for row in manifest["files"]:
            path = material_root / row["path"]
            need(path.resolve(strict=True).is_relative_to(material_root), "Offline material escapes its root")
            self.policy.pin(path)
            need(self.policy.pins[str(path)]["sha256"] == row["sha256"], "Offline material pin mismatch")
        hooks, source = self.creation_source()
        def remap(native):
            from pxr import Sdf
            import omni.usd
            api = self.policy.load("native/offline_impl.py", dict(ROOT=self.policy.native, Sdf=Sdf, omni=omni))
            return api.remap_offline(native)
        hooks._remap_offline = remap
        layout = importlib.import_module("simulators.robodojo_layouts")
        module = importlib.import_module("simulators.robodojo")
        need(Path(module.__file__).resolve() == self.policy.teacher / "simulators/robodojo.py"
             and not getattr(module, "_robofate_runtime_installed", False),
             "Wrong or previously bound Native implementation")
        layout.ROOT = module.ROOT = self.policy.native
        module.decode_scene = layout.decode_scene
        namespace = dict(module.__dict__, _node_adjust_config=hooks._adjust_config,
                         _node_prepare_native=hooks._prepare_native, _experience=experience, _make_app=self.make_app)
        exec(compile(source, str(module.__file__) + "::runtime", "exec"), namespace)
        module.RoboDojoTask._create_native = namespace["_create_native"]
        module._robofate_runtime_installed = True
        self.installed = True
        return module.RoboDojoTask, hooks, layout

    def create(self, preparation, run, *, source_context_ref=None):
        """Actual construction only; admission must be checked by the caller first."""
        expected = self.layout(preparation)
        from . import deployment
        source = preparation["source"]
        need(source_context_ref is not None, "Pinned source context required for task extension binding")
        deployment.verify_environment(self.policy, dict(
            run_dir=str(run), source_context_ref=source_context_ref, source_id=source["source_id"],
            task=source["task"], native_limit=preparation["native_step_limit"]))
        run = Path(run)
        need(run.parent == self.policy.root / "runs" and run.resolve(strict=True) == run,
             "Canonical owned run required")
        cls, hooks, layout = self.install()
        source = preparation["source"]
        bank, ordinal, path = layout.decode_scene(source["task"], source["encoded_seed"])
        need((bank, ordinal, path.resolve()) ==
             (source["native_bank"], source["native_ordinal"], Path(source["resolved_layout_path"])),
             "Original numeric layout decoder differs")
        task = None
        try:
            from ..recording import rtx_profile
            evidence.exclusive(run / "nvoptix_container_asset.json", rtx_profile.nvoptix_asset_metadata())
            task = wrap_class(cls, run)(source["task"], seed=source["encoded_seed"])
            rtx_profile.configure(task, hooks.PROFILE, evidence_path=run / "renderer_raw_observations.jsonl")
            task.reset()
            evidence.exclusive(run / "actual_initial_binding.json", hooks.verify_initial(task, preparation))
            actual = task.native.seed_manager.get_seed_scene_info(task.ordinal)
            need(canonical(actual) == canonical(expected), "Actual initialized layout differs from pinned raw layout")
            evidence.exclusive(run / "initialized_layout.json", actual)
            return task
        except BaseException as primary:
            if task is not None:
                try:
                    task.close()
                except BaseException as cleanup:
                    primary.add_note("Native close also failed: " + repr(cleanup))
            raise
