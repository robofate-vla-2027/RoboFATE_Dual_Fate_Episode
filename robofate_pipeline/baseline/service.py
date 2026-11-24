"""Owned loopback service around the unchanged frozen-DM inference implementation."""
import asyncio
import os
from pathlib import Path
import random
import signal
import sys
import time

from .. import evidence
from ..evidence import need
from ..native.deployment import executable


def binding(policy):
    d = policy.value["frozen_dm"]
    model = policy.directory(d["model_root"])
    experiment = policy.native / "XPolicyLab/policy/OpenDM/scripts/robodojo_dm05_history.py"
    # The checkpoint is external, but every weight/config byte is release-pinned.
    files = [p for p in model.rglob("*") if p.is_file()]
    need(files and not any(p.is_symlink() for p in model.rglob("*")), "Regular external checkpoint required")
    manifest = evidence.decode(evidence.read_pin(d["checkpoint_manifest_ref"]))
    refs = manifest["files"]
    need(manifest["schema"] == "robofate.checkpoint-files.v1" and len(refs) == len(files)
         and {r["path"] for r in refs} == {str(p) for p in files}, "Exact checkpoint inventory differs")
    for ref in refs:
        evidence.verify_blob(ref)
    for path in (model / "config.json", model / "norm_stats.json", experiment):
        if path == experiment:
            policy.pin(path)
        else:
            need(str(path) in {r["path"] for r in refs}, "Missing checkpoint metadata")
    return policy.load("baseline/frozen_impl.py", dict(
        NATIVE_ROOT=policy.native, MODEL_PATH=model, EXPERIMENT=experiment))


def environment(policy, native_env):
    d = policy.value["frozen_dm"]
    command = executable(policy, d["python"])
    env = dict(native_env)
    for key in ("LD_PRELOAD", "VK_LAYER_PATH", "VK_INSTANCE_LAYERS", "VK_ICD_FILENAMES",
                "EMBODIEDEVO_ROBODOJO_KIT_ARGS", "REFATE_VK_COPY_LOG"):
        env.pop(key, None)
    need(d["cuda_visible_devices"] == "0", "Original isolated policy logical device required")
    env.update(CUDA_VISIBLE_DEVICES=d["cuda_visible_devices"],
        LD_LIBRARY_PATH=":".join(str(policy.directory(v)) for v in d["library_dirs"]),
        WANDB_MODE="offline", NO_ALBUMENTATIONS_UPDATE="1",
        HF_HOME=str(Path(env["XDG_CACHE_HOME"]) / "hf"),
        TRITON_CACHE_DIR=str(Path(env["XDG_CACHE_HOME"]) / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(Path(env["XDG_CACHE_HOME"]) / "inductor"))
    return command, env


def run(context_ref, *, lease_ref, lock_fd):
    from .runtime import load_context
    from ..foreground import validate_lease, parent_death_guard
    context, policy = load_context(context_ref)
    policy.host_guard()
    inherited = validate_lease(policy, context_ref, lease_ref, lock_fd)
    parent_death_guard(inherited["parent"])
    try:
        return _run(context, policy)
    finally:
        os.close(lock_fd)


def _run(context, policy):
    d = policy.value["frozen_dm"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == d["cuda_visible_devices"]
         and os.environ.get("LOCAL_RANK", "0") == "0", "Wrong frozen policy device")
    need(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Checkpoint network fallback is forbidden")
    api = binding(policy)
    config = evidence.decode(evidence.read_pin(d["config_ref"]))
    cfg = api.validate_config(config, 0)
    policy.verify_release()
    policy.install_import_paths()
    for path in (policy.native / "XPolicyLab", policy.native):
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from XPolicyLab.policy.OpenDM import model as native
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    need(Path(native.__file__).resolve() == policy.native / "XPolicyLab/policy/OpenDM/model.py",
         "Wrong native OpenDM adapter imported")
    run_dir = Path(context["run_dir"]) / "server"
    run_dir.mkdir(mode=0o700)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    with (run_dir / "policy_io.jsonl").open("x", buffering=1) as audit:
        started = time.monotonic()
        with api.direct_gpu_loading(native):
            model = api.audited_model_class(native, audit, cuda_metrics=lambda: {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            })(cfg)
        ready = dict(status="model_loaded_not_episode_qualified", pid=os.getpid(),
            execution_policy_ref=policy.reference, context_ref=context["self_ref"],
            alias=policy.value["ssh_alias"], hostname=policy.value["expected_hostname"],
            physical_gpu=policy.value["native_devices"]["physical_gpu_ids"][0],
            model_identity="DM0.5 / OpenDM mm-robodojo v1",
            checkpoint=str(api.MODEL_PATH), native_adapter=str(Path(native.__file__).resolve()),
            config=cfg, **model.freeze_receipt, seed=0, initial_random_seed=0,
            action_shape=[25, 14], prediction_shape=[50, 14], native_action_type="joint",
            model_action_mode="absolute", native_state_action_order=list(api.STATE_DIMS),
            history_slots=20, history_fps=1, runtime_fps=25,
            loading_strategy=api.LOADING_STRATEGY, checkpoint_device_map={"": "cuda:0"},
            rng_reset_after_load=False, load_seconds=time.monotonic() - started,
            deployment_teacher_calls=0, hidden_state_inputs=False, qualified=False)

        async def serve():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, stop.set)
            server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0,
                ws_ping_interval_s=120.0, ws_ping_timeout_s=120.0))
            try:
                await server.start()
                ready["url"] = server.url
                evidence.exclusive(run_dir / "model_ready.json", ready)
                await stop.wait()
            finally:
                try:
                    await server.stop()
                finally:
                    for sig in (signal.SIGTERM, signal.SIGINT):
                        loop.remove_signal_handler(sig)
            evidence.exclusive(run_dir / "model_stopped.json", dict(
                pid=os.getpid(), normal_service_stop=True, model_identity="DM0.5", trainable_parameters=0))
        asyncio.run(serve())
